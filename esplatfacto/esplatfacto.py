"""ESplatfacto: Gaussian splatting supervised by event-camera data, as a subclass of nerfstudio's Splatfacto.

Each training step renders the camera at the start (t0) and end (t) of an event window and compares their
log-intensity change with the window's accumulated events (see get_loss_dict).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy
from gsplat.strategy.ops import remove
from pytorch_msssim import SSIM
from typing_extensions import Literal

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.model_components import renderers
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig, get_viewmat, random_quat_tensor


@dataclass
class ESplatfactoModelConfig(SplatfactoModelConfig):
    """Splatfacto with event supervision. Densification follows the official Event3DGS (3DGS) recipe."""

    _target: Type = field(default_factory=lambda: ESplatfactoModel)
    background_color: Literal["dataset", "random", "black", "white"] = "dataset"  # type: ignore[assignment]
    """Training background. 'dataset' uses the event dataset's background (EventNeRF: 159/255 grey)."""
    num_downscales: int = 0
    """Must stay 0 for Bayer (colour) events: downscaling would mix the colour channels of the mosaic."""
    cull_alpha_thresh: float = 0.005
    cull_scale_thresh: float = 0.1
    """3DGS culls Gaussians larger than 0.1 x camera extent; the EventNeRF ring has radius 0.9."""
    densify_grad_thresh: float = 0.0002
    use_absgrad: bool = False
    cull_screen_size: float = 0.058
    """3DGS culls Gaussians with a screen radius above 20 px (0.058 of a 346 px wide image)."""
    stop_screen_size_at: int = 15000
    use_scale_regularization: bool = True
    ssim_lambda: float = 0.2
    """weight of the D-SSIM term on the Bayer-masked event maps (official Event3DGS: 0.2)"""
    event_threshold: Optional[float] = None
    """contrast threshold C; defaults to the dataset's value (EventNeRF: 0.25 synthetic, 0.1 real)"""
    tonemap_eps: Optional[float] = None
    """eps of the log tonemap; defaults to the dataset's value"""
    event_gamma: float = 2.2
    """tonemap log(rgb**gamma + eps): 2.2 as EventNeRF, 1.0 as the official Event3DGS default"""
    event_loss: Literal["mse", "l1"] = "l1"
    """per-pixel penalty between the rendered log-intensity change and C * accumulated events"""
    rand_ratio: float = 0.7
    """fraction of pixels kept in the event loss each step (official Event3DGS README: 0.7; 1.0 keeps all)"""
    clip_to_bound: bool = True
    """make Gaussians outside the dataset's bounding cylinder invisible and cull them (EventNeRF density clipping)"""
    opacity_reg: float = 0.0
    """weight of an L1 penalty on opacities"""


class ESplatfactoModel(SplatfactoModel):
    config: ESplatfactoModelConfig

    def populate_modules(self):
        metadata = self.kwargs.get("metadata") or {}
        self.event_meta = metadata.get("events") or {}
        self.event_threshold = self.config.event_threshold or self.event_meta.get("event_threshold", 0.25)
        self.tonemap_eps = self.config.tonemap_eps or self.event_meta.get("tonemap_eps", 1e-5)
        self.use_bayer = self.event_meta.get("bayer", "RGGB") is not None
        if self.use_bayer and self.config.num_downscales != 0:
            raise ValueError("num_downscales must be 0 for Bayer event data")
        bound = self.event_meta.get("bound")
        self.bound = None if bound is None else (float(bound["radius"]), float(bound["z_min"]), float(bound["z_max"]))
        self._bayer_mask = None
        self._infos: List[Dict] = []

        # Splatfacto resolves background_color with get_color, which has no "dataset" entry.
        bg_mode = self.config.background_color
        self.config.background_color = "black" if bg_mode == "dataset" else bg_mode
        super().populate_modules()
        self.config.background_color = bg_mode
        if bg_mode == "dataset":
            self.background_color = torch.tensor(self.event_meta.get("background", [159 / 255] * 3), dtype=torch.float32)

        if (self.seed_points is None or self.config.random_init) and self.bound is not None:
            self._init_in_bound()

        # Every frame is a training camera, so Splatfacto's pause of num_train_data steps after each opacity
        # reset would disable densification for most of every cycle; 3DGS densifies without a pause.
        self.strategy = DefaultStrategy(
            prune_opa=self.config.cull_alpha_thresh,
            grow_grad2d=self.config.densify_grad_thresh,
            grow_scale3d=self.config.densify_size_thresh,
            grow_scale2d=self.config.split_screen_size,
            prune_scale3d=self.config.cull_scale_thresh,
            prune_scale2d=self.config.cull_screen_size,
            refine_scale2d_stop_iter=self.config.stop_screen_size_at,
            refine_start_iter=self.config.warmup_length,
            refine_stop_iter=self.config.stop_split_at,
            reset_every=self.config.reset_alpha_every * self.config.refine_every,
            refine_every=self.config.refine_every,
            pause_refine_after_reset=0,
            absgrad=self.config.use_absgrad,
            revised_opacity=False,
            verbose=False,
        )
        self.strategy_state = self.strategy.initialize_state(scene_scale=1.0)

    def _init_in_bound(self):
        """Random Gaussians filling the scene's bounding cylinder."""
        r, z0, z1 = self.bound
        n = self.config.num_random
        rad = r * torch.sqrt(torch.rand(n))
        theta = 2 * math.pi * torch.rand(n)
        means = torch.stack([rad * torch.cos(theta), rad * torch.sin(theta), z0 + (z1 - z0) * torch.rand(n)], -1)
        distances, _ = self.k_nearest_sklearn(means, 3)
        avg_dist = torch.from_numpy(distances).mean(dim=-1, keepdim=True).float()
        dim_sh = self.gauss_params["features_rest"].shape[1] + 1
        self.gauss_params = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(means),
                "scales": torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3))),
                "quats": torch.nn.Parameter(random_quat_tensor(n)),
                "features_dc": torch.nn.Parameter(torch.rand(n, 3)),
                "features_rest": torch.nn.Parameter(torch.zeros(n, dim_sh - 1, 3)),
                "opacities": torch.nn.Parameter(torch.logit(0.1 * torch.ones(n, 1))),
            }
        )

    def outside_bound(self, means: torch.Tensor) -> Optional[torch.Tensor]:
        """Boolean mask of Gaussians whose centre lies outside the scene's bounding cylinder (None if no clipping)."""
        if not self.config.clip_to_bound or self.bound is None:
            return None
        r, z0, z1 = self.bound
        return (means[:, 0] ** 2 + means[:, 1] ** 2 > r * r) | (means[:, 2] < z0) | (means[:, 2] > z1)

    def step_cb(self, optimizers, step):
        super().step_cb(optimizers, step)
        self._infos = []

    def step_post_backward(self, step):
        assert step == self.step
        if self._infos and step < self.strategy.refine_stop_iter:
            # Both renders of the window receive gradients, so both feed the densification statistics;
            # gsplat has no public call for an extra update, and the last info is added by step_post_backward.
            for info in self._infos[:-1]:
                self.strategy._update_state(self.gauss_params, self.strategy_state, info, packed=False)
            self.strategy.step_post_backward(
                params=self.gauss_params,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=self._infos[-1],
                packed=False,
            )
        self._infos = []
        if step > self.config.warmup_length and step % self.config.refine_every == 0:
            outside = self.outside_bound(self.means)
            if outside is not None and outside.any():
                remove(params=self.gauss_params, optimizers=self.optimizers, state=self.strategy_state, mask=outside)

    def _get_background_color(self):
        if not self.training and renderers.BACKGROUND_COLOR_OVERRIDE is not None:
            return renderers.BACKGROUND_COLOR_OVERRIDE.to(self.device)
        if self.config.background_color == "dataset":
            return self.background_color.to(self.device)
        return super()._get_background_color()

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}
        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            camera_to_world = camera.camera_to_worlds

        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(int(camera.width.item()), int(camera.height.item()), self.background_color)
        else:
            crop_ids = slice(None)

        means = self.means[crop_ids]
        opacities = torch.sigmoid(self.opacities[crop_ids]).squeeze(-1)
        outside = self.outside_bound(means)
        if outside is not None:
            opacities = opacities * (~outside).to(opacities.dtype)
        colors = torch.cat((self.features_dc[crop_ids][:, None, :], self.features_rest[crop_ids]), dim=1)

        viewmat = get_viewmat(camera_to_world)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        render_mode = "RGB+ED" if self.config.output_depth_during_training or not self.training else "RGB"
        sh_degree = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)

        render, alpha, info = rasterization(
            means=means,
            quats=self.quats[crop_ids],
            scales=torch.exp(self.scales[crop_ids]),
            opacities=opacities,
            colors=colors,
            viewmats=viewmat,
            Ks=K,
            width=W,
            height=H,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree,
            sparse_grad=False,
            absgrad=self.strategy.absgrad,
            rasterize_mode=self.config.rasterize_mode,
        )
        if self.training:
            self.strategy.step_pre_backward(self.gauss_params, self.optimizers, self.strategy_state, self.step, info)
            self._infos.append(info)

        background = self._get_background_color()
        rgb = render[..., :3] + (1 - alpha) * background
        if not self.training:
            # Clamping during training would zero the event-loss gradient of every pixel above 1.
            rgb = torch.clamp(rgb, 0.0, 1.0)
        depth = None
        if render_mode == "RGB+ED":
            depth = torch.where(alpha > 0, render[..., 3:4], render[..., 3:4].detach().max()).squeeze(0)
        if not self.training:
            background = background.expand(H, W, 3)
        return {"rgb": rgb.squeeze(0), "depth": depth, "accumulation": alpha.squeeze(0), "background": background}

    def bayer_mask(self, H: int, W: int) -> torch.Tensor:
        """(H, W, 3) one-hot RGGB colour filter of the DAVIS 346C (EventNeRF convention)."""
        if self._bayer_mask is None or self._bayer_mask.shape[:2] != (H, W):
            m = torch.zeros((H, W, 3), device=self.device)
            m[0::2, 0::2, 0] = 1  # r
            m[0::2, 1::2, 1] = 1  # g
            m[1::2, 0::2, 1] = 1  # g
            m[1::2, 1::2, 2] = 1  # b
            self._bayer_mask = m
        return self._bayer_mask

    def event_prediction(self, rgb: torch.Tensor, rgb_pre: torch.Tensor) -> torch.Tensor:
        """Rendered log-intensity change from t0 to t as the sensor sees it: (H, W) behind a Bayer filter, else (H, W, 3).

        Events fire on changes of log linear intensity, so the difference is taken of log(rgb**gamma + eps).
        """
        g, eps = self.config.event_gamma, self.tonemap_eps
        diff = torch.log(rgb.clamp(min=0) ** g + eps) - torch.log(rgb_pre.clamp(min=0) ** g + eps)
        if self.use_bayer:
            return (diff * self.bayer_mask(*diff.shape[:2])).sum(-1)
        return diff

    def event_target(self, batch, like: torch.Tensor) -> torch.Tensor:
        """C * (accumulated signed events (+ polarity offset)) of the window; zero where nothing fired."""
        target = batch["events"].to(self.device) * self.event_threshold
        return target if like.dim() == 2 else target[..., None].expand_as(like)

    def loss_view(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(H, W, 3) maps as the official Event3DGS loss sees them: each pixel only in its Bayer channel,
        on a random rand_ratio subset of pixels."""
        if pred.dim() == 2:
            m = self.bayer_mask(*pred.shape)
            pred, target = pred[..., None] * m, target[..., None] * m
        keep = (torch.rand(pred.shape[:2], device=self.device) < self.config.rand_ratio).to(pred.dtype)[..., None]
        return pred * keep, target * keep

    def get_metrics_dict(self, outputs, outputs_pre, batch) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        metrics_dict = {}
        with torch.no_grad():
            pred = self.event_prediction(outputs["rgb"], outputs_pre["rgb"])
            target = self.event_target(batch, pred)
            mse = torch.mean((pred - target) ** 2)
            metrics_dict["event_mse"] = mse
            # error relative to predicting "no change" everywhere (1.0 = no better than a blank model)
            metrics_dict["event_mse_rel"] = mse / torch.mean(target**2).clamp(min=1e-12)
        metrics_dict["gaussian_count"] = self.num_points
        return metrics_dict

    def get_loss_dict(self, outputs, outputs_pre, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        pred = self.event_prediction(outputs["rgb"], outputs_pre["rgb"])
        target = self.event_target(batch, pred)
        pred, target = self.loss_view(pred, target)
        err = pred - target
        main = err.pow(2).mean() if self.config.event_loss == "mse" else err.abs().mean()
        if self.config.ssim_lambda > 0:
            p, t = pred.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None]
            simloss = 1 - SSIM(data_range=1.0, size_average=True, channel=3)(p, t)
            main = (1 - self.config.ssim_lambda) * main + self.config.ssim_lambda * simloss
        loss_dict = {"main_loss": main}
        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.scales)
            ratio = scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1)
            loss_dict["scale_reg"] = 0.1 * (torch.clamp(ratio, min=self.config.max_gauss_ratio) - self.config.max_gauss_ratio).mean()
        if self.config.opacity_reg > 0:
            loss_dict["opacity_reg"] = self.config.opacity_reg * torch.sigmoid(self.opacities).mean()
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        # Dataset images are event-window previews, not RGB ground truth (scripts/eval_gt.py scores against GT):
        # show the preview, the render and the opacity, where floaters show up.
        preview = self.get_gt_img(batch["image"])[..., :3]
        acc = outputs["accumulation"].repeat(1, 1, 3)
        images_dict = {"img": torch.cat([preview, outputs["rgb"], acc], dim=1)}
        return {"gaussian_count": float(self.num_points)}, images_dict
