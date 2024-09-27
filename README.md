# ESplat: Event Camera 3D Gaussian Splatting
This is the unofficial implementation for [Event3DGS: Event-based 3D Gaussian Splatting for High-Speed Robot Egomotion
(CoRL 2024)](https://arxiv.org/abs/2406.02972).


<div align='center'> 
<img src="https://arxiv.org/html/2406.02972v3/x2.png" height="230px">
</div>

# Installation
LERF follows the integration guidelines described [here](https://docs.nerf.studio/en/latest/developer_guides/new_methods.html) for custom methods within Nerfstudio. Update to `nerfstudio==1.0.3`.
### 0. Install Nerfstudio dependencies
[Follow these instructions](https://docs.nerf.studio/en/latest/quickstart/installation.html) up to and including "tinycudann" to install dependencies and create an environment
### 1. Clone this repo
`git clone https://github.com/jayhsu0627/Event3DGS`
### 2. Install this repo as a python package
Navigate to this folder and run `python -m pip install -e .`

### 3. Reinstall gsplat to avoid this [issue](https://github.com/nerfstudio-project/nerfstudio/issues/2727)
`pip install git+https://github.com/nerfstudio-project/gsplat.git@v0.1.10`
<!-- ### 4. Run `ns-install-cli` -->

### Checking the install
Run `ns-train -h`: you should see a list of "subcommands" with lerf, lerf-big, and lerf-lite included among them.

# Using ESplat
Now that ESplat is installed you can play with it! 

- Launch training with `ns-train esplatfacto --data <data_folder>`. This specifies a data folder to use. For more details, see [Nerfstudio documentation](https://docs.nerf.studio/en/latest/quickstart/first_nerf.html). 

```
ns-train esplatfacto --data C:\Users\sjxu\3_Event_3DGS\Data\nerfstudio\sewing
```

```
ns-train esplatfacto-big --data C:\Users\sjxu\3_Event_3DGS\Data\nerfstudio\sewing --pipeline.model.use_scale_regularization True --pipeline.model.cull_alpha_thresh=0.005 --pipeline.model.continue_cull_post_densification=False
```


- Connect to the viewer by forwarding the viewer port (we use VSCode to do this), and click the link to `viewer.nerf.studio` provided in the output of the train script. Use the viewer running locally at: `http://localhost:7007`

TODO: edit `class ExportGaussianSplat(Exporter)` in `exporter.py`
- Output `*.ply`
`ns-export gaussian-splat --load-config outputs\plane\esplatfacto\2024-04-22_201709\config.yml --output-dir exports/ply`
```
  File "C:\Users\sjxu\AppData\Local\miniconda3\envs\event3dgs\lib\site-packages\nerfstudio\scripts\exporter.py", line 614, in entrypoint
    tyro.cli(Commands).main()
  File "C:\Users\sjxu\AppData\Local\miniconda3\envs\event3dgs\lib\site-packages\nerfstudio\scripts\exporter.py", line 536, in main
    assert isinstance(pipeline.model, SplatfactoModel)
AssertionError
```

1. Split RGB channels independent 3dgs
2. I'm now a pure 3dgs with no 159 assumption
3. Add t0 and t estimation?
4. 

- 

```test

ray_samplers = load_event_data_split(args.datadir, args.scene, camera_mgr=camera_mgr, split=args.train_split,
                                         skip=args.trainskip, max_winsize=args.winsize,
                                         use_ray_jitter=args.use_ray_jitter, is_colored=args.is_colored,
                                         polarity_offset=args.polarity_offset, cycle=args.is_cycled,
                                         is_rgb_only=args.is_rgb_only, randomize_winlen=args.use_random_window_len,
                                         win_constant_count=args.use_window_constant_count)

To see how "load_event_data_split" determine

prev_file = img_files[(i-winsize+len(img_files))%len(img_files)]
curr_file = img_files[i]
```

<!-- ## Relevancy Map NormalizVation
By default, the viewer shows **raw** relevancy scaled with the turbo colormap. As values lower than 0.5 correspond to irrelevant regions, **we recommend setting the `range` parameter to (-1.0, 1.0)**. To match the visualization from the paper, check the `Normalize` tick-box, which stretches the values to use the full colormap.

The images below show the rgb, raw, centered, and normalized output views for the query "Lily".


<div align='center'>
<img src="readme_images/lily_rgb.jpg" width="150px">
<img src="readme_images/lily_raw.jpg" width="150px">
<img src="readme_images/lily_centered.jpg" width="150px">
<img src="readme_images/lily_normalized.jpg" width="150px">
</div> -->


<!-- ## Resolution
The Nerfstudio viewer dynamically changes resolution to achieve a desired training throughput.

**To increase resolution, pause training**. Rendering at high resolution (512 or above) can take a second or two, so we recommend rendering at 256px
## `lerf-big` and `lerf-lite`
If your GPU is struggling on memory, we provide a `lerf-lite` implementation that reduces the LERF network capacity and number of samples along rays. If you find you still need to reduce memory footprint, the most impactful parameters for memory are `num_lerf_samples`, hashgrid levels, and hashgrid size.

`lerf-big` provides a larger model that uses ViT-L/14 instead of ViT-B/16 for those with large memory GPUs.

# Extending LERF
Be mindful that code for visualization will change as more features are integrated into Nerfstudio, so if you fork this repo and build off of it, check back regularly for extra changes.
### Issues
Please open Github issues for any installation/usage problems you run into. We've tried to support as broad a range of GPUs as possible with `lerf-lite`, but it might be necessary to provide even more low-footprint versions. Thank you!
#### Known TODOs
- [ ] Integrate into `ns-render` commands to render videos from the command line with custom prompts
### Using custom image encoders
We've designed the code to modularly accept any image encoder that implements the interface in `BaseImageEncoder` (`image_encoder.py`). An example of different encoder implementations can be seen in `clip_encoder.py` vs `openclip_encoder.py`, which implement OpenAI's CLIP and OpenCLIP respectively.
### Code structure
(TODO expand this section)
The main file to look at for editing and building off LERF is `lerf.py`, which extends the Nerfacto model from Nerfstudio, adds an additional language field, losses, and visualization. The CLIP and DINO pre-processing are carried out by `pyramid_interpolator.py` and `dino_dataloader.py`. -->

<!--
## Bibtex
If you find this useful, please cite the paper!
<pre id="codecell0">@inproceedings{lerf2023,
&nbsp;author = {Kerr, Justin and Kim, Chung Min and Goldberg, Ken and Kanazawa, Angjoo and Tancik, Matthew},
&nbsp;title = {LERF: Language Embedded Radiance Fields},
&nbsp;booktitle = {International Conference on Computer Vision (ICCV)},
&nbsp;year = {2023},
} </pre> -->
