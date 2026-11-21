# Training on your own event data

ESplat needs three things from a recording: the event stream, a camera pose for many instants along the trajectory, and the camera intrinsics. This page describes the scene folder `dataloader.py` reads, then the steps from a raw recording to a trained model.

## Scene folder

```
data/myscene/
├── events.npz          # required: x, y, t, p
├── pose/               # required: one 4x4 camera-to-world matrix per pose, sorted by file name
│   ├── 00000.txt
│   └── ...
├── intrinsics.txt      # optional: 3x3 or 4x4 K matrix
└── timestamps.txt      # optional: one time per pose
```

| File | Content |
|---|---|
| `events.npz` | Equal-length arrays `x`, `y` (integer pixel coordinates, 0-based), `t` (any unit, e.g. microseconds), `p` (polarity, `{0, 1}` or `{-1, +1}`). Events are sorted by `t` if they are not already. Coordinates must lie inside the sensor (`--width`, `--height`, default 346 x 260). |
| `pose/*.txt` | 4x4 **OpenCV** camera-to-world matrix per pose: camera x right, y down, z forward (looking direction). The world frame is yours; tell the loader its up axis with `--world-up {y-down,z-up,y-up}` (EventNeRF data is `y-down`). |
| `intrinsics.txt` | Pinhole K of the **undistorted** events: `fx 0 cx / 0 fy cy / 0 0 1`. Without it, `--focal` (or the preset focal) and a centred principal point are used; `--fx --fy --cx --cy` override single values. |
| `timestamps.txt` | Time of each pose, in the same unit as `t`, strictly increasing. Without it, poses are taken as uniformly spaced over `[0, t_max]` (the EventNeRF convention). |

Training uses every pose: an event window always runs between two poses, so poses should be dense along the trajectory (EventNeRF uses 1000 per orbit). If the first and last pose coincide, the trajectory is treated as a closed loop and windows may wrap around.

## From a recording to a model

**1. Export events.** Convert your camera's format (AEDAT, HDF5, rosbag) to `events.npz`, for example from arrays you have loaded:

```python
import numpy as np
np.savez("data/myscene/events.npz", x=x.astype(np.int16), y=y.astype(np.int16),
         t=t.astype(np.int64), p=p.astype(np.int8))
```

**2. Undistort.** The renderer is a pinhole camera, so lens distortion must be removed from the event coordinates first (for example `cv2.undistortPoints` with your calibration, then rounding and dropping events that leave the sensor). Write the resulting K to `intrinsics.txt`.

**3. Poses.** Get camera poses from a robot arm, motion capture, a turntable, or structure-from-motion on frames reconstructed from events (e.g. E2VID). They must be the pose of the event camera's optical centre (apply the hand-eye or rig calibration), in OpenCV axes. Sample them densely and write their times to `timestamps.txt`, or resample them to uniform spacing.

**4. Colour.** Colour DAVIS sensors put an RGGB Bayer filter over the pixels; ESplat masks the loss with that pattern. For a monochrome sensor pass `--monochrome`.

**5. Preprocess.**

```
python dataloader.py -p data -s myscene --type real --world-up z-up --normalize
```

- `--type real` selects EventNeRF's real-camera event settings (C = 0.1, window 100, tonemap eps 1e-5). Set `--event-threshold` to your sensor's contrast threshold if you know it.
- `--normalize` recentres the world on the centroid of the camera positions and scales it so the farthest camera is 0.9 from the origin. The model's culling thresholds and the automatic scene bound assume this EventNeRF scale and an object near the origin, so use it unless your poses already satisfy that.
- `--bound auto` (default) estimates a bounding cylinder from event activity, assuming a static object in front of a uniform background, seen from all around. For other captures (forward-facing, cluttered background) use `--bound none` or give `--bound radius,z_min,z_max`.
- `--bg-color` is the background intensity in [0, 255] used for both renders of a window.
- `--polarity-offset` compensates a sensor whose ON and OFF thresholds differ (EventNeRF: -0.375 for sewing).

The loader prints the intrinsics, camera distances and bound it used, and warns when the cameras are far from the expected scale. Check `data/myscene/output/images/`: each preview shows the events of one window and should show the object's edges.

**6. Train and view.**

```
ns-train esplatfacto --data data/myscene/output --experiment-name myscene --vis viewer+tensorboard
```

## Checklist when results look wrong

| Symptom | Likely cause |
|---|---|
| Nothing converges, `event_mse_rel` stays near 1 | Poses in the wrong convention (OpenGL instead of OpenCV, or world-to-camera instead of camera-to-world), or pose times not matching event times (`timestamps.txt` in a different unit). |
| Object doubled or smeared | Pose/event time offset, or wrong focal length. |
| Everything culled, or a few huge Gaussians | Scene far from the 0.9 camera scale: use `--normalize`. |
| Object clipped | Automatic bound too tight for your capture: `--bound none`. |
| Grey haze in empty space | Background intensity (`--bg-color`) far from the real background. |
