from data.utils.data_loader_split import load_event_data_split
from data.utils.module_3dgs_sample_ray_split import CameraManager

camera_mgr = CameraManager(learnable=False)

ray_samplers = load_event_data_split('C:/Users/sjxu/3_Event_3DGS/Data/EventNeRF',
                                    'chick',
                                    camera_mgr=camera_mgr,
                                    split='train',
                                    skip=1,
                                    max_winsize=100,
                                    use_ray_jitter=True,
                                    is_colored=True,
                                    polarity_offset=0.0,
                                    cycle=True,
                                    is_rgb_only=False,
                                    randomize_winlen=True,
                                    win_constant_count=0)