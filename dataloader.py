"""Pre-process event data (EventNeRF layout) into a nerfstudio dataset folder.

    python dataloader.py -p data -s lego                 # EventNeRF synthetic (auto-detected)
    python dataloader.py -p data -s sewing --polarity-offset -0.375
    python dataloader.py -p data -s myscene --type real --world-up z-up --normalize   # own recording

A scene folder holds one event ``.npz`` (x, y, t, p), ``pose/*.txt`` (one 4x4 OpenCV camera-to-world
matrix per frame), and optionally ``intrinsics.txt`` (3x3 or 4x4 K) and ``timestamps.txt`` (one time per
pose, in event time units). See docs/CUSTOM_DATA.md.

Writes ``<scene>/output/transforms.json`` (intrinsics, poses in an OpenGL-camera z-up world, and an
``events`` block) plus ``images/`` previews that serve only as viewer thumbnails. Training accumulates
event windows from the raw stream on the fly (esplatfacto/data/esplatfacto_datamanager.py).
"""
import argparse
import glob
import json
import os

import numpy as np
from PIL import Image

DAVIS346 = (346, 260)  # (width, height) of the EventNeRF sensor

# Event and camera settings of the EventNeRF configs (configs/nerf/*.txt and configs/real/*.txt).
PRESETS = {
    "synthetic": dict(focal=480.55, event_threshold=0.25, max_winsize=50, tonemap_eps=0.1),
    "real": dict(focal=389.25, event_threshold=0.1, max_winsize=100, tonemap_eps=1e-5),
}
# EventNeRF's density-clipping cylinder for its real recordings (crop_r, crop_y_min, crop_y_max), z-up frame.
EVENTNERF_REAL_BOUND = dict(radius=0.25, z_min=-0.15, z_max=0.35)
# Camera ring radius of the EventNeRF scenes; the model's culling thresholds assume this scale.
EVENTNERF_RING_RADIUS = 0.9

CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])
WORLD_TO_ZUP = {
    "y-down": np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64),
    "y-up": np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64),
    "z-up": np.eye(4),
}


def load_events(path):
    """(x, y, t, p) sorted by time, polarity in {-1, +1}."""
    data = np.load(path)
    missing = [k for k in ("x", "y", "t", "p") if k not in data]
    if missing:
        raise KeyError(f"{path} lacks arrays {missing}; expected x, y, t, p")
    x, y, t, p = (np.asarray(data[k]) for k in ("x", "y", "t", "p"))
    if np.any(np.diff(t) < 0):
        order = np.argsort(t, kind="stable")
        x, y, t, p = x[order], y[order], t[order], p[order]
    p = p.astype(np.float32)
    values = set(np.unique(p).tolist())
    if values <= {0.0, 1.0}:
        p = p * 2 - 1
    elif not values <= {-1.0, 1.0}:
        raise ValueError(f"Polarities must be {{0, 1}} or {{-1, 1}}, found {sorted(values)[:5]}")
    return x.astype(np.int64), y.astype(np.int64), t.astype(np.float64), p


def load_poses(pose_dir):
    files = sorted(glob.glob(os.path.join(pose_dir, "*.txt")))
    if not files:
        raise FileNotFoundError(f"No pose files in {pose_dir}")
    poses = np.stack([np.loadtxt(f).reshape(4, 4) for f in files])
    if not np.allclose(poses[:, 3], [0, 0, 0, 1]):
        raise ValueError("Poses must be homogeneous 4x4 camera-to-world matrices")
    return poses, files


def load_intrinsics(path):
    """K from a 3x3 or 4x4 matrix file -> (fx, fy, cx, cy)."""
    k = np.loadtxt(path)
    if k.shape not in ((3, 3), (4, 4)):
        raise ValueError(f"{path}: expected a 3x3 or 4x4 matrix, got {k.shape}")
    return float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])


def frame_times(t, n_frames, timestamps_path=None):
    """Pose times: from timestamps.txt if given, else EventNeRF's uniform spacing over [0, t_max]."""
    if timestamps_path is None:
        return np.linspace(0.0, float(t.max()), n_frames)
    times = np.loadtxt(timestamps_path).reshape(-1).astype(np.float64)
    if len(times) != n_frames:
        raise ValueError(f"{timestamps_path}: {len(times)} timestamps for {n_frames} poses")
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"{timestamps_path}: timestamps must be strictly increasing")
    if times[-1] < t[0] or times[0] > t[-1]:
        raise ValueError(f"{timestamps_path}: pose times [{times[0]}, {times[-1]}] do not overlap the events "
                         f"[{t[0]}, {t[-1]}]; check the units")
    return times


def window_counts(events, times, start, end, size):
    """Per-pixel (signed sum, event count) of the events in (times[start], times[end]]; size = (W, H)."""
    x, y, t, p = events
    w, h = size
    lo, hi = np.searchsorted(t, [times[start], times[end]], side="right")
    idx = y[lo:hi] * w + x[lo:hi]
    signed = np.bincount(idx, weights=p[lo:hi], minlength=w * h).reshape(h, w)
    count = np.bincount(idx, minlength=w * h).reshape(h, w)
    return signed, count


def normalize_poses(poses_cv, ring_radius=EVENTNERF_RING_RADIUS):
    """Recentre on the camera centroid and scale so the farthest camera sits at ring_radius."""
    centers = poses_cv[:, :3, 3]
    centroid = centers.mean(0)
    scale = ring_radius / np.linalg.norm(centers - centroid, axis=1).max()
    out = poses_cv.copy()
    out[:, :3, 3] = (centers - centroid) * scale
    return out, centroid, scale


def estimate_bound(events, times, poses_cv, intr, size, world_to_zup, n_views=72, half_window=10, margin=0.1,
                   grid=96):
    """Conservative visual hull from event activity -> bounding cylinder (z-up frame).

    For a static object on a uniform background, events only fire on the object. Keeps the grid points
    that project inside the convex hull of event-active pixels in >= 95% of the views that see them.
    """
    import cv2

    fx, fy, cx, cy = intr
    w, h = size
    n = len(poses_cv)
    period = n - 1 if np.allclose(poses_cv[0], poses_cv[-1], atol=1e-6) else n
    extent = 0.85 * np.linalg.norm(poses_cv[:, :3, 3], axis=1).min()  # stay inside the camera ring
    lin = np.linspace(-extent, extent, grid)
    pts = np.stack(np.meshgrid(lin, lin, lin, indexing="ij"), -1).reshape(-1, 3)
    inside = np.zeros(len(pts), dtype=np.int32)
    seen = np.zeros(len(pts), dtype=np.int32)
    kernel = np.ones((3, 3), np.uint8)
    for k in np.linspace(half_window, period - 1 - half_window, n_views).astype(int):
        _, count = window_counts(events, times, k - half_window, k + half_window, size)
        active = cv2.morphologyEx((count > 0).astype(np.uint8), cv2.MORPH_OPEN, kernel)
        mask = np.zeros_like(active)
        ys, xs = np.nonzero(active)
        if len(xs) < 10:
            continue
        hull = cv2.convexHull(np.stack([xs, ys], 1).astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 1)
        mask = cv2.dilate(mask, kernel, iterations=3)
        c2w = poses_cv[k]
        cam = (pts - c2w[:3, 3]) @ c2w[:3, :3]  # world -> OpenCV camera
        z = cam[:, 2]
        u = fx * cam[:, 0] / np.maximum(z, 1e-6) + cx
        v = fy * cam[:, 1] / np.maximum(z, 1e-6) + cy
        vis = (z > 1e-3) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        seen += vis
        ui, vi = u[vis].astype(int), v[vis].astype(int)
        inside[np.nonzero(vis)[0][mask[vi, ui] > 0]] += 1
    keep = (seen >= n_views // 2) & (inside >= 0.95 * seen)
    if keep.sum() < 10:
        raise RuntimeError("Bound estimation failed; pass --bound none or explicit values.")
    kept = (world_to_zup[:3, :3] @ pts[keep].T).T
    radius = np.linalg.norm(kept[:, :2], axis=1).max()
    z_min, z_max = kept[:, 2].min(), kept[:, 2].max()
    pad = margin * max(radius, z_max - z_min)
    return dict(radius=float(radius + pad), z_min=float(z_min - pad), z_max=float(z_max + pad))


def save_previews(events, times, out_dir, window, threshold, polarity_offset, size):
    os.makedirs(out_dir, exist_ok=True)
    for k in range(len(times)):
        signed, count = window_counts(events, times, max(k - window, 0), k, size)
        e = (signed + polarity_offset * count) * threshold
        img = np.clip(0.5 + 0.5 * e, 0, 1)
        Image.fromarray((np.repeat(img[..., None], 3, -1) * 255).astype(np.uint8)).save(
            os.path.join(out_dir, f"r_{k:05d}.png")
        )


def main(args):
    scene_dir = os.path.join(args.root_path, args.scene)
    out_dir = args.output or os.path.join(scene_dir, "output")
    npz = sorted(glob.glob(os.path.join(scene_dir, "*.npz")) + glob.glob(os.path.join(scene_dir, "events", "*.npz")))
    if len(npz) != 1:
        raise FileNotFoundError(f"Expected exactly one event .npz in {scene_dir}, found {npz}")
    events = load_events(npz[0])
    poses_cv, _ = load_poses(os.path.join(scene_dir, "pose"))
    n = len(poses_cv)
    ts_path = os.path.join(scene_dir, "timestamps.txt")
    times = frame_times(events[2], n, ts_path if os.path.isfile(ts_path) else None)

    size = (args.width, args.height)
    x, y = events[0], events[1]
    if x.min() < 0 or y.min() < 0 or x.max() >= size[0] or y.max() >= size[1]:
        raise ValueError(f"Event coordinates span x [{x.min()}, {x.max()}], y [{y.min()}, {y.max()}], outside the "
                         f"{size[0]}x{size[1]} sensor; set --width/--height")

    data_type = args.type
    if data_type == "auto":
        # EventNeRF synthetic streams use integer millisecond stamps; its real recordings are float.
        data_type = "synthetic" if np.all(events[2] == np.round(events[2])) else "real"
        print(f"[{args.scene}] auto-detected '{data_type}' EventNeRF settings; pass --type for other data")
    preset = dict(PRESETS[data_type])
    for key in ("event_threshold", "max_winsize", "tonemap_eps"):
        if getattr(args, key) is not None:
            preset[key] = getattr(args, key)

    intr_path = os.path.join(scene_dir, "intrinsics.txt")
    focal = args.focal or preset["focal"]
    fx, fy, cx, cy = load_intrinsics(intr_path) if os.path.isfile(intr_path) else (focal, focal, size[0] / 2,
                                                                                    size[1] / 2)
    fx, fy, cx, cy = (v if o is None else o for v, o in zip((fx, fy, cx, cy), (args.fx, args.fy, args.cx, args.cy)))

    if args.normalize:
        poses_cv, centroid, scale = normalize_poses(poses_cv)
        print(f"[{args.scene}] normalized: centroid {np.round(centroid, 4).tolist()}, scale {scale:.4g}")
    world_to_zup = WORLD_TO_ZUP[args.world_up]
    cycle = bool(np.allclose(poses_cv[0], poses_cv[-1], atol=1e-6))

    if args.bound == "auto":
        bound = estimate_bound(events, times, poses_cv, (fx, fy, cx, cy), size, world_to_zup)
    elif args.bound == "eventnerf":
        bound = dict(EVENTNERF_REAL_BOUND)
    elif args.bound == "none":
        bound = None
    else:
        r, z0, z1 = (float(v) for v in args.bound.split(","))
        bound = dict(radius=r, z_min=z0, z_max=z1)

    frames = []
    for k in range(n):
        c2w = world_to_zup @ poses_cv[k] @ CV_TO_GL
        frames.append({"file_path": f"images/r_{k:05d}.png", "transform_matrix": c2w.tolist(), "time": float(times[k])})

    out = {
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "w": size[0],
        "h": size[1],
        "camera_model": "OPENCV",
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "orientation_override": "none",
        "events": {
            "path": os.path.relpath(os.path.abspath(npz[0]), os.path.abspath(out_dir)),
            "data_type": data_type,
            "event_threshold": preset["event_threshold"],
            "polarity_offset": args.polarity_offset,
            "max_winsize": int(preset["max_winsize"]),
            "tonemap_eps": preset["tonemap_eps"],
            "cycle": cycle,
            "bayer": None if args.monochrome else "RGGB",
            "background": [args.bg_color / 255.0] * 3,
            "bound": bound,
        },
        "frames": frames,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "transforms.json"), "w") as f:
        json.dump(out, f, indent=2)
    ring = np.linalg.norm(poses_cv[:, :3, 3], axis=1)
    print(f"[{args.scene}] {data_type}: {n} poses, {len(events[2])} events, cycle={cycle}, "
          f"fx={fx:g} fy={fy:g} cx={cx:g} cy={cy:g}, camera distance {ring.min():.3g}-{ring.max():.3g}")
    if not 0.5 < ring.max() < 2.0:
        print(f"[{args.scene}] warning: cameras are {ring.max():.3g} from the origin; the model's culling thresholds "
              "assume ~0.9 (EventNeRF). Consider --normalize.")
    print(f"[{args.scene}] bound: {bound}")
    save_previews(events, times, os.path.join(out_dir, "images"), preset["max_winsize"] // 2,
                  preset["event_threshold"], args.polarity_offset, size)
    print(f"[{args.scene}] wrote {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-p", "--root_path", type=str, default="data", help="Root folder holding the scenes.")
    parser.add_argument("-s", "--scene", type=str, default="lego", help="Scene folder name.")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output folder (default: <root>/<scene>/output).")
    parser.add_argument("--type", choices=["auto", "synthetic", "real"], default="auto",
                        help="Preset for C, window and tonemap eps (and focal, if no intrinsics are given).")
    parser.add_argument("--width", type=int, default=DAVIS346[0], help="Sensor width in pixels.")
    parser.add_argument("--height", type=int, default=DAVIS346[1], help="Sensor height in pixels.")
    parser.add_argument("--focal", type=float, default=None, help="Focal length (pixels) when there is no intrinsics.txt.")
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--world-up", dest="world_up", choices=list(WORLD_TO_ZUP), default="y-down",
                        help="Up axis of the pose world frame (EventNeRF: y-down).")
    parser.add_argument("--normalize", action="store_true",
                        help="Recentre on the camera centroid and scale the farthest camera to radius 0.9.")
    parser.add_argument("--event-threshold", dest="event_threshold", type=float, default=None,
                        help="Contrast threshold C (EventNeRF: 0.25 synthetic, 0.1 real).")
    parser.add_argument("--polarity-offset", dest="polarity_offset", type=float, default=0.0,
                        help="Added to every event polarity (EventNeRF real scenes: sewing -0.375, chick -0.438).")
    parser.add_argument("--max-winsize", dest="max_winsize", type=int, default=None)
    parser.add_argument("--tonemap-eps", dest="tonemap_eps", type=float, default=None)
    parser.add_argument("--bg-color", dest="bg_color", type=float, default=159.0,
                        help="Background intensity in [0, 255] (EventNeRF uses 159).")
    parser.add_argument("--bound", type=str, default="auto",
                        help="'auto' (event visual hull), 'eventnerf' (EventNeRF real-scene cylinder), 'none', "
                        "or 'radius,z_min,z_max'.")
    parser.add_argument("--monochrome", action="store_true", help="Events are not behind an RGGB Bayer filter.")
    main(parser.parse_args())
