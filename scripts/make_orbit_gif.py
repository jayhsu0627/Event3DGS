"""Orbit GIF of a trained model along its training ring: events | render | opacity.

python scripts/make_orbit_gif.py --load-config outputs/drums/esplatfacto/<timestamp>/config.yml --out img/drums_orbit.gif
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataloader import frame_times, load_events, window_counts  # noqa: E402
from eval_gt import all_cameras, load_pipeline, render  # noqa: E402

ON_COLOR = np.array([0.76, 0.25, 0.05])
OFF_COLOR = np.array([0.11, 0.30, 0.85])


def event_panel(signed: np.ndarray, saturate: float = 3.0) -> np.ndarray:
    """White background; ON events orange, OFF events blue, intensity by net count."""
    a = np.clip(np.abs(signed) / saturate, 0, 1)[..., None]
    color = np.where(signed[..., None] > 0, ON_COLOR, OFF_COLOR)
    return (1 - a) + a * color


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--load-config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--step", type=int, default=10, help="Use every n-th training pose.")
    ap.add_argument("--window", type=int, default=10, help="Event window length in frames.")
    ap.add_argument("--ms-per-frame", type=int, default=60)
    args = ap.parse_args()

    pipeline = load_pipeline(args.load_config)
    cams = all_cameras(pipeline)
    data_dir = Path(pipeline.datamanager.config.dataparser.data)
    meta = json.loads((data_dir / "transforms.json").read_text())["events"]
    events = load_events(os.path.normpath(data_dir / meta["path"]))
    times = frame_times(events[2], len(glob.glob(str(data_dir / "images" / "r_*.png"))))

    frames = []
    for k in range(args.window, max(cams) + 1, args.step):
        rgb, acc = render(pipeline, cams[k])
        signed, _ = window_counts(events, times, k - args.window, k)
        tile = np.concatenate([event_panel(signed), rgb, np.repeat(acc[..., None], 3, -1)], axis=1)
        frames.append(Image.fromarray((tile * 255).astype(np.uint8)).quantize(colors=128, method=Image.MEDIANCUT))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=args.ms_per_frame, loop=0,
                   optimize=True)
    print(f"Wrote {args.out} ({len(frames)} frames, {args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
