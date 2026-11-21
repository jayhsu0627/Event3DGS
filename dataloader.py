"""Pre-process EventNeRF-style event data into a nerfstudio dataset folder.

    python dataloader.py -p data/ -s lego                 # synthetic (auto-detected)
    python dataloader.py -p data/ -s sewing --polarity-offset -0.375

Writes ``data/<scene>/output/``:

* ``transforms.json``: intrinsics, one camera pose per frame, and an ``events`` block
  (path to the raw event file, event threshold, polarity offset, window size, scene bound...).
* ``images/r_xxxxx.png``: previews of event windows. They are only used as viewer thumbnails;
  they are *not* training targets.

Training targets are accumulated from the raw event stream on the fly
(see ``esplatfacto/data/esplatfacto_datamanager.py``), so every iteration sees a freshly sampled
window, and the signed event counts are never rescaled.

Coordinate frame: EventNeRF poses are OpenCV camera-to-world matrices in a world frame whose
+y axis points *down*. We convert the camera axes to the OpenGL convention used by nerfstudio and
rotate the world so that +z points up. No re-centering or re-scaling is applied, so the scene bound
(``events.bound``) is expressed in the same units as EventNeRF's ``crop_r`` / ``crop_y_*``.
"""
import argparse
import glob
import json
import os

import numpy as np
from PIL import Image

SENSOR_W, SENSOR_H = 346, 260  # DAVIS 346C

# Values follow the EventNeRF configs (configs/nerf/*.txt and configs/real/*.txt).
PRESETS = {
    "synthetic": dict(focal=480.55, event_threshold=0.25, max_winsize=50, tonemap_eps=0.1),
    "real": dict(focal=389.25, event_threshold=0.1, max_winsize=100, tonemap_eps=1e-5),
}
# EventNeRF's density-clipping cylinder for its real recordings (crop_r, crop_y_min, crop_y_max),
# converted to our z-up frame (z = -y_eventnerf).
EVENTNERF_REAL_BOUND = dict(radius=0.25, z_min=-0.15, z_max=0.35)

# OpenCV camera axes -> OpenGL camera axes, and EventNeRF world (y down) -> z-up world.
CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])
WORLD_YDOWN_TO_ZUP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)


def load_events(path):
    data = np.load(path)
    x, y, t, p = (np.asarray(data[k]) for k in ("x", "y", "t", "p"))
    if np.any(np.diff(t) < 0):
        order = np.argsort(t, kind="stable")
        x, y, t, p = x[order], y[order], t[order], p[order]
    p = p.astype(np.float32)
    if set(np.unique(p).tolist()) <= {0.0, 1.0}:
        p = p * 2 - 1
    return x.astype(np.int64), y.astype(np.int64), t.astype(np.float64), p


def load_poses(pose_dir):
    files = sorted(glob.glob(os.path.join(pose_dir, "*.txt")))
    if not files:
        raise FileNotFoundError(f"No pose files in {pose_dir}")
    return np.stack([np.loadtxt(f).reshape(4, 4) for f in files]), files


def frame_times(t, n_frames):
    """EventNeRF convention: pose k is taken at time k / (n_frames - 1) * t_max."""
    return np.linspace(0.0, float(t.max()), n_frames)


def window_counts(events, times, start, end):
    """Per-pixel (signed sum, event count) of the events in (times[start], times[end]]."""
    x, y, t, p = events
    lo, hi = np.searchsorted(t, [times[start], times[end]], side="right")
    idx = y[lo:hi] * SENSOR_W + x[lo:hi]
    signed = np.bincount(idx, weights=p[lo:hi], minlength=SENSOR_W * SENSOR_H).reshape(SENSOR_H, SENSOR_W)
    count = np.bincount(idx, minlength=SENSOR_W * SENSOR_H).reshape(SENSOR_H, SENSOR_W)
    return signed, count


def estimate_bound(events, times, poses_cv, focal, n_views=72, half_window=10, margin=0.1, grid=96):
    """Conservative visual hull from event activity -> bounding cylinder (z-up frame).

    For a static object on a uniform background, events only fire on the object. For each of
    ``n_views`` views we take the convex hull of event-active pixels (after removing isolated
    noise pixels), and keep the 3D grid points that project inside that hull in >= 95% of the
    views that see them. The convex hull can only over-estimate the silhouette, so this never
    carves away real geometry; it just removes the empty space where floaters live.
    """
    import cv2

    n = len(poses_cv)
    period = n - 1 if np.allclose(poses_cv[0], poses_cv[-1], atol=1e-6) else n
    centers = poses_cv[:, :3, 3]
    extent = 0.85 * np.linalg.norm(centers, axis=1).min()  # stay inside the camera ring
    lin = np.linspace(-extent, extent, grid)
    pts = np.stack(np.meshgrid(lin, lin, lin, indexing="ij"), -1).reshape(-1, 3)
    inside = np.zeros(len(pts), dtype=np.int32)
    seen = np.zeros(len(pts), dtype=np.int32)
    kernel = np.ones((3, 3), np.uint8)
    for k in np.linspace(half_window, period - 1 - half_window, n_views).astype(int):
        _, count = window_counts(events, times, k - half_window, k + half_window)
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
        u = focal * cam[:, 0] / np.maximum(z, 1e-6) + SENSOR_W / 2
        v = focal * cam[:, 1] / np.maximum(z, 1e-6) + SENSOR_H / 2
        vis = (z > 1e-3) & (u >= 0) & (u < SENSOR_W) & (v >= 0) & (v < SENSOR_H)
        seen += vis
        ui, vi = u[vis].astype(int), v[vis].astype(int)
        inside[np.nonzero(vis)[0][mask[vi, ui] > 0]] += 1
    keep = (seen >= n_views // 2) & (inside >= 0.95 * seen)
    if keep.sum() < 10:
        raise RuntimeError("Bound estimation failed; pass --bound none or explicit values.")
    kept = (WORLD_YDOWN_TO_ZUP[:3, :3] @ pts[keep].T).T
    radius = np.linalg.norm(kept[:, :2], axis=1).max()
    z_min, z_max = kept[:, 2].min(), kept[:, 2].max()
    pad = margin * max(radius, z_max - z_min)
    return dict(radius=float(radius + pad), z_min=float(z_min - pad), z_max=float(z_max + pad))


def save_previews(events, times, out_dir, window, threshold, polarity_offset):
    os.makedirs(out_dir, exist_ok=True)
    for k in range(len(times)):
        signed, count = window_counts(events, times, max(k - window, 0), k)
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
    times = frame_times(events[2], n)

    data_type = args.type
    if data_type == "auto":
        # EventNeRF synthetic streams use integer millisecond stamps; real recordings are float.
        data_type = "synthetic" if np.all(events[2] == np.round(events[2])) else "real"
    preset = dict(PRESETS[data_type])
    for key in ("focal", "event_threshold", "max_winsize", "tonemap_eps"):
        if getattr(args, key) is not None:
            preset[key] = getattr(args, key)
    cycle = bool(np.allclose(poses_cv[0], poses_cv[-1], atol=1e-6))

    if args.bound == "auto":
        bound = estimate_bound(events, times, poses_cv, preset["focal"])
    elif args.bound == "eventnerf":
        bound = dict(EVENTNERF_REAL_BOUND)
    elif args.bound == "none":
        bound = None
    else:
        r, z0, z1 = (float(v) for v in args.bound.split(","))
        bound = dict(radius=r, z_min=z0, z_max=z1)

    frames = []
    for k in range(n):
        c2w = WORLD_YDOWN_TO_ZUP @ poses_cv[k] @ CV_TO_GL
        frames.append({"file_path": f"images/r_{k:05d}.png", "transform_matrix": c2w.tolist(), "time": float(times[k])})

    out = {
        "fl_x": preset["focal"],
        "fl_y": preset["focal"],
        "cx": SENSOR_W / 2,
        "cy": SENSOR_H / 2,
        "w": SENSOR_W,
        "h": SENSOR_H,
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
    print(f"[{args.scene}] {data_type}: {n} poses, {len(events[2])} events, cycle={cycle}, focal={preset['focal']}")
    print(f"[{args.scene}] bound: {bound}")
    save_previews(events, times, os.path.join(out_dir, "images"), preset["max_winsize"] // 2,
                  preset["event_threshold"], args.polarity_offset)
    print(f"[{args.scene}] wrote {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-p", "--root_path", type=str, default="data", help="Root folder holding the scenes.")
    parser.add_argument("-s", "--scene", type=str, default="lego", help="Scene folder name.")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output folder (default: <root>/<scene>/output).")
    parser.add_argument("--type", choices=["auto", "synthetic", "real"], default="auto")
    parser.add_argument("--focal", type=float, default=None, help="Override the preset focal length (pixels).")
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
    parser.add_argument("--monochrome", action="store_true", help="Events are not behind a Bayer filter.")
    main(parser.parse_args())
