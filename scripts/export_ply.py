"""Export a trained model as a standard 3DGS .ply (readable by common Gaussian splat viewers).

python scripts/export_ply.py --load-config outputs/lego/esplatfacto/<timestamp>/config.yml --out exports/lego.ply
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_gt import load_pipeline  # noqa: E402


def write_ply(path: Path, fields: dict) -> None:
    """Binary little-endian PLY with one float32 vertex property per entry of fields."""
    names = list(fields)
    data = np.stack([fields[k].astype(np.float32) for k in names], axis=1)
    header = "\n".join(
        ["ply", "format binary_little_endian 1.0", f"element vertex {len(data)}"]
        + [f"property float {k}" for k in names]
        + ["end_header"]
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--load-config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    model = load_pipeline(args.load_config).model
    keep = torch.ones(model.num_points, dtype=torch.bool, device=model.means.device)
    outside = model.outside_bound(model.means)
    if outside is not None:
        keep &= ~outside  # rendered with zero opacity, so not part of the scene
    n = int(keep.sum())

    means = model.means[keep].cpu().numpy()
    fields = {"x": means[:, 0], "y": means[:, 1], "z": means[:, 2]}
    for k in ("nx", "ny", "nz"):
        fields[k] = np.zeros(n)
    dc = model.features_dc[keep].cpu().numpy()
    for i in range(3):
        fields[f"f_dc_{i}"] = dc[:, i]
    rest = model.features_rest[keep].transpose(1, 2).reshape(n, -1).cpu().numpy()  # 3DGS: channel-major
    for i in range(rest.shape[1]):
        fields[f"f_rest_{i}"] = rest[:, i]
    fields["opacity"] = model.opacities[keep].cpu().numpy()[:, 0]  # logit, as 3DGS stores it
    scales = model.scales[keep].cpu().numpy()  # log scale
    for i in range(3):
        fields[f"scale_{i}"] = scales[:, i]
    quats = model.quats[keep].cpu().numpy()  # (w, x, y, z)
    for i in range(4):
        fields[f"rot_{i}"] = quats[:, i]

    finite = np.all(np.isfinite(np.stack(list(fields.values()), 1)), axis=1)
    write_ply(args.out, {k: v[finite] for k, v in fields.items()})
    print(f"Wrote {args.out}: {int(finite.sum())} Gaussians ({model.num_points - n} outside the bound dropped)")


if __name__ == "__main__":
    main()
