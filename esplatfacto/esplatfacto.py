from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import numpy as np
import open_clip
import torch
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.ray_samplers import PDFSampler
from nerfstudio.model_components.renderers import DepthRenderer
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.utils.colormaps import ColormapOptions, apply_colormap
from nerfstudio.viewer.viewer_elements import *
from torch.nn import Parameter

# from lerf.encoders.image_encoder import BaseImageEncoder
# from lerf.lerf_field import LERFField
# from lerf.lerf_fieldheadnames import LERFFieldHeadNames
# from lerf.lerf_renderers import CLIPRenderer, MeanRenderer




























from data_loader_split import load_event_data_split
from module_3dgs_sample_ray_split import CameraManager

camera_mgr = CameraManager(learnable=False)

ray_samplers = load_event_data_split('C:/Users/sjxu/3_Event_3DGS/Event3DGS/data',
                                    'chick',
                                    camera_mgr=camera_mgr,
                                    split='train',
                                    skip=1,
                                    max_winsize=1,
                                    use_ray_jitter=True,
                                    is_colored=True,
                                    polarity_offset=0.0,
                                    cycle=False,
                                    is_rgb_only=False,
                                    randomize_winlen=True,
                                    win_constant_count=0)
