"""Score a trained model against EventNeRF ground-truth RGB views and render off-orbit views.

    python scripts/eval_gt.py --load-config outputs/.../config.yml --gt-dir comp_data/lego/gt --out eval/lego

* GT views: EventNeRF's ``comp_data.zip`` (https://github.com/r00tman/EventNeRF) contains, per synthetic scene,
  ``<scene>/gt/r_XXXXX.png`` rendered at training pose XXXXX. We render the model at the same poses.
* Events only constrain log-intensity *changes*, so absolute colour is unknown up to a per-channel gain/offset in
  log space. As in EventNeRF / Event3DGS evaluation, we fit log(gt) = a * log(pred) + b per channel jointly over
  all views (pixels with pred > 20/255) before computing PSNR / SSIM / LPIPS. ``--no-align`` reports raw scores.
* Off-orbit views: all training cameras lie on one horizontal ring, so floaters that hide behind the object from
  every training view only show up from above/below. We render the orbit tilted by +-``--elevations`` degrees,
  together with the accumulated opacity (alpha) so floaters are visible regardless of their colour.
"""
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import yaml

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.configs.method_configs import all_methods


def load_pipeline(config_path: Path):
    """nerfstudio's eval_setup without its isinstance(pipeline, nerfstudio Pipeline) assert, so that checkpoints
    trained with older versions of this repo (whose pipeline did not subclass it) can be evaluated too."""
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    config.pipeline.datamanager._target = all_methods[config.method_name].pipeline.datamanager._target
    config.load_dir = config.get_checkpoint_dir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = config.pipeline.setup(device=device, test_mode="test")
    pipeline.eval()
    ckpts = sorted(Path(config.load_dir).glob("step-*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {config.load_dir}")
    # Your own training checkpoint: nerfstudio stores numpy scalars in it, which torch>=2.6's default
    # weights_only=True loader rejects.
    state = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
    pipeline.load_pipeline(state["pipeline"], state["step"])
    print(f"Loaded {ckpts[-1]}")
    return pipeline


def fit_log_alignment(preds, gts, margin=20.0 / 255, eps=1e-5):
    a, b = [], []
    for c in range(3):
        x = np.concatenate([p[..., c].ravel() for p in preds]).astype(np.float64)
        y = np.concatenate([g[..., c].ravel() for g in gts]).astype(np.float64)
        sel = x > margin
        lx, ly = np.log(x[sel] + eps), np.log(y[sel] + eps)
        var = np.mean((lx - lx.mean()) ** 2)
        slope = np.mean((lx - lx.mean()) * (ly - ly.mean())) / var if var > 1e-12 else 1.0
        a.append(slope)
        b.append(ly.mean() - slope * lx.mean())
    return np.array(a), np.array(b)


def apply_alignment(img, a, b, eps=1e-5):
    return np.clip(np.exp(np.log(img + eps) * a + b) - eps, 0, 1)


def all_cameras(pipeline):
    """Map frame number (from r_XXXXX.png) -> single camera, over train and eval datasets."""
    out = {}
    dm = pipeline.datamanager
    for ds in (dm.train_dataset, dm.eval_dataset):
        for i, fname in enumerate(ds.image_filenames):
            m = re.search(r"r_(\d+)", Path(fname).stem)
            if m:
                out[int(m.group(1))] = ds.cameras[i : i + 1]
    return out


def tilt(camera: Cameras, degrees: float) -> Cameras:
    """Rotate the camera about the horizontal axis through the orbit centre (origin, z up) that is
    perpendicular to its position vector, i.e. raise/lower it along its meridian, keeping it looking at the
    same point relative to the scene."""
    c2w = camera.camera_to_worlds[0].clone()
    pos = c2w[:3, 3]
    axis = torch.linalg.cross(torch.tensor([0.0, 0.0, 1.0]), pos)
    axis = axis / axis.norm()
    t = math.radians(degrees)
    K = torch.tensor([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = torch.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K
    new = torch.eye(4)[:3]
    new[:3, :3] = R @ c2w[:3, :3]
    new[:3, 3] = R @ pos
    cam = Cameras(
        camera_to_worlds=new[None],
        fx=camera.fx, fy=camera.fy, cx=camera.cx, cy=camera.cy,
        width=camera.width, height=camera.height, camera_type=camera.camera_type,
    )
    return cam


@torch.no_grad()
def render(pipeline, camera, background=None):
    model = pipeline.model
    if background is not None:
        from nerfstudio.model_components import renderers

        renderers.BACKGROUND_COLOR_OVERRIDE = torch.tensor(background, dtype=torch.float32)
    out = model.get_outputs_for_camera(camera.to(model.device))
    rgb = out["rgb"].clamp(0, 1).cpu().numpy()
    acc = out["accumulation"][..., 0].clamp(0, 1).cpu().numpy()
    return rgb, acc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--load-config", type=Path, required=True)
    ap.add_argument("--gt-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--background", type=float, nargs=3, default=[159 / 255] * 3,
                    help="background colour for all renders (GT uses 159/255 gray)")
    ap.add_argument("--elevations", type=float, nargs="*", default=[30.0, -30.0])
    ap.add_argument("--no-align", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pipeline = load_pipeline(args.load_config)
    cams = all_cameras(pipeline)
    summary = {"config": str(args.load_config), "num_gaussians": int(pipeline.model.num_points)}

    if args.gt_dir is not None:
        frames, preds, gts = [], [], []
        for f in sorted(args.gt_dir.glob("r_*.png")):
            k = int(re.search(r"r_(\d+)", f.stem).group(1))
            cam = cams.get(k)
            if cam is None and k == max(cams) + 1:  # r_01000 duplicates r_00000 on a closed orbit
                cam = cams.get(0)
            if cam is None:
                continue
            rgb, _ = render(pipeline, cam, args.background)
            gt = np.asarray(Image.open(f).convert("RGB"), dtype=np.float32) / 255.0
            if gt.shape != rgb.shape:
                raise ValueError(f"shape mismatch {gt.shape} vs {rgb.shape} for {f}")
            frames.append(k)
            preds.append(rgb)
            gts.append(gt)
        a, b = (np.ones(3), np.zeros(3)) if args.no_align else fit_log_alignment(preds, gts)
        from pytorch_msssim import ssim as ssim_fn
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)
        psnrs, ssims, lpipss = [], [], []
        (args.out / "gt_compare").mkdir(exist_ok=True)
        for k, p, g in zip(frames, preds, gts):
            pa = apply_alignment(p, a, b).astype(np.float32)
            mse = np.mean((pa - g) ** 2)
            psnrs.append(10 * np.log10(1.0 / mse))
            tp, tg = (torch.from_numpy(x).permute(2, 0, 1)[None] for x in (pa, g))
            ssims.append(float(ssim_fn(tp, tg, data_range=1.0)))
            lpipss.append(float(lpips(tp, tg)))
            if k % 100 == 0:
                Image.fromarray((np.concatenate([g, pa], 1) * 255).astype(np.uint8)).save(
                    args.out / "gt_compare" / f"r_{k:05d}.png")
        summary.update(psnr=float(np.mean(psnrs)), ssim=float(np.mean(ssims)), lpips=float(np.mean(lpipss)),
                       num_views=len(frames), align_a=a.tolist(), align_b=b.tolist())

    # off-orbit renders: rows = elevations, columns = 6 azimuths; rgb on top of alpha
    keys = sorted(cams)
    period = keys[-1]
    picks = [keys[int(i * period / 6)] for i in range(6)]
    rows = []
    alpha_stats = {}
    for elev in [0.0] + list(args.elevations):
        rgb_row, acc_row, accs = [], [], []
        for k in picks:
            cam = tilt(cams[k], elev) if elev else cams[k]
            rgb, acc = render(pipeline, cam, args.background)
            rgb_row.append(rgb)
            acc_row.append(np.repeat(acc[..., None], 3, -1))
            accs.append(acc)
        rows += [np.concatenate(rgb_row, 1), np.concatenate(acc_row, 1)]
        accs = np.stack(accs)
        # fraction of the image covered by (semi-)opaque splats: floaters inflate this off-orbit
        alpha_stats[f"elev{elev:+.0f}"] = {"mean_alpha": float(accs.mean()), "frac_alpha>0.5": float((accs > 0.5).mean())}
    Image.fromarray((np.concatenate(rows, 0) * 255).astype(np.uint8)).save(args.out / "orbit_grid.png")
    summary["alpha"] = alpha_stats
    with open(args.out / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
