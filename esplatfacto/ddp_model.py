import torch
import torch.nn as nn
from utils import TINY_NUMBER, HUGE_NUMBER
from collections import OrderedDict
from nerf_network import DummyEmbedder, Embedder, MLPNet
import os
import logging

logger = logging.getLogger(__package__)


class NerfNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        # foreground
        if args.use_pe:
            self.fg_embedder_position = Embedder(input_dim=3,
                                                 max_freq_log2=args.max_freq_log2 - 1,
                                                 N_freqs=args.max_freq_log2,
                                                 N_anneal=args.N_anneal,
                                                 N_anneal_min_freq=args.N_anneal_min_freq,
                                                 use_annealing=args.use_annealing)
            self.fg_embedder_viewdir = Embedder(input_dim=3,
                                                max_freq_log2=args.max_freq_log2_viewdirs - 1,
                                                N_freqs=args.max_freq_log2_viewdirs,
                                                N_anneal=args.N_anneal,
                                                N_anneal_min_freq=args.N_anneal_min_freq_viewdirs,
                                                use_annealing=args.use_annealing)
        else:
            self.fg_embedder_position = DummyEmbedder(input_dim=3)
            self.fg_embedder_viewdir = DummyEmbedder(input_dim=3)

        self.fg_net = MLPNet(D=args.netdepth, W=args.netwidth,
                             input_ch=self.fg_embedder_position.out_dim,
                             input_ch_viewdirs=self.fg_embedder_viewdir.out_dim,
                             use_viewdirs=args.use_viewdirs,
                             act=args.activation,
                             garf_sigma=args.garf_sigma,
                             crop_y=(args.crop_y_min, args.crop_y_max),
                             crop_r=args.crop_r,
                             init_gain=args.init_gain)

        self.with_bg = args.with_bg
        self.with_ldist = args.use_ldist_reg

        self.bg_color = args.bg_color

    def forward(self, ray_o, ray_d, fg_z_max, fg_z_vals, iteration):
        '''
        :param ray_o, ray_d: [..., 3]
        :param fg_z_max: [...,]
        :param fg_z_vals: [..., N_samples]
        :return
        '''
        ray_d_norm = torch.norm(ray_d, dim=-1, keepdim=True)  # [..., 1]
        viewdirs = ray_d / ray_d_norm  # [..., 3]
        dots_sh = list(ray_d.shape[:-1])

        ######### render foreground
        N_samples = fg_z_vals.shape[-1]
        fg_ray_o = ray_o.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        fg_ray_d = ray_d.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        fg_viewdirs = viewdirs.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        fg_pts = fg_ray_o + fg_z_vals.unsqueeze(-1) * fg_ray_d
        fg_raw = self.fg_net(fg_pts, fg_viewdirs, iteration=iteration,
                             embedder_position=self.fg_embedder_position,
                             embedder_viewdir=self.fg_embedder_viewdir)
        # alpha blending
        fg_dists = fg_z_vals[..., 1:] - fg_z_vals[..., :-1]
        # account for view directions
        fg_dists = ray_d_norm * torch.cat((fg_dists, fg_z_max.unsqueeze(-1) - fg_z_vals[..., -1:]),
                                          dim=-1)  # [..., N_samples]
        fg_alpha = 1. - torch.exp(-fg_raw['sigma'] * fg_dists)  # [..., N_samples]
        T = torch.cumprod(1. - fg_alpha + TINY_NUMBER, dim=-1)  # [..., N_samples]
        bg_lambda = T[..., -1]
        T = torch.cat((torch.ones_like(T[..., 0:1]), T[..., :-1]), dim=-1)  # [..., N_samples]
        fg_weights = fg_alpha * T  # [..., N_samples]
        fg_rgb_map = torch.sum(fg_weights.unsqueeze(-1) * fg_raw['rgb'], dim=-2)  # [..., 3]
        fg_depth_map = torch.sum(fg_weights * fg_z_vals, dim=-1)     # [...,]


        fg_midpoint = (fg_z_vals[..., 1:] + fg_z_vals[..., :-1])/2
        fg_midpoint = ray_d_norm * torch.cat((fg_midpoint, (fg_z_max.unsqueeze(-1) + fg_z_vals[..., -1:])/2),
                                          dim=-1)

        fg_midpointdist = abs(fg_midpoint.unsqueeze(-1) - fg_midpoint.unsqueeze(-2))

        if self.with_ldist:
            fg_ldist1 = torch.sum(fg_weights.unsqueeze(-1)*fg_weights.unsqueeze(-2)*fg_midpointdist, (-2, -1))
            fg_ldist2 = torch.sum(1/3*(fg_weights**2)*fg_dists, -1)
            fg_ldist = fg_ldist1+fg_ldist2

        if self.with_bg:
            rgb_map = fg_rgb_map + bg_rgb_map
        else:
            rgb_map = fg_rgb_map + bg_lambda.unsqueeze(-1)*(self.bg_color/255.)  # hard coded value of background in sRGB = 159/255

        ret = OrderedDict([('rgb', rgb_map),            # loss
                           ('fg_weights', fg_weights),  # importance sampling
                           ('fg_rgb', fg_rgb_map.detach()),      # below are for logging
                           ('fg_depth', fg_depth_map.detach()),
                           ('bg_lambda', bg_lambda)
                           ])
        if self.with_ldist:
            ret['fg_ldist'] = fg_ldist      # distortion regularizer
        return ret


def remap_name(name):
    name = name.replace('.', '-')  # dot is not allowed by pytorch
    if name[-1] == '/':
        name = name[:-1]
    idx = name.rfind('/')
    for i in range(2):
        if idx >= 0:
            idx = name[:idx].rfind('/')
    return name[idx + 1:]


class NerfNetWithAutoExpo(nn.Module):
    def __init__(self, args, optim_autoexpo=False, img_names=None):
        super().__init__()
        self.nerf_net = NerfNet(args)

        self.optim_autoexpo = optim_autoexpo
        if self.optim_autoexpo:
            assert(img_names is not None)
            logger.info('Optimizing autoexposure!')

            self.img_names = [remap_name(x) for x in img_names]
            logger.info('\n'.join(self.img_names))
            self.autoexpo_params = nn.ParameterDict(OrderedDict([(x, nn.Parameter(torch.Tensor([0.5, 0.]))) for x in self.img_names]))

    def forward(self, ray_o, ray_d, fg_z_max, fg_z_vals, iteration, img_name=None):
        '''
        :param ray_o, ray_d: [..., 3]
        :param fg_z_max: [...,]
        :param fg_z_vals: [..., N_samples]
        :return
        '''
        if img_name is not None:
            img_name = remap_name(img_name)

        ret = self.nerf_net(ray_o, ray_d, fg_z_max, fg_z_vals, iteration)
        if self.optim_autoexpo and (img_name in self.autoexpo_params):
            autoexpo = self.autoexpo_params[img_name]
            scale = torch.abs(autoexpo[0]) + 0.5  # make sure scale is always positive
            shift = autoexpo[1]
            ret['autoexpo'] = (scale, shift)

        return ret
