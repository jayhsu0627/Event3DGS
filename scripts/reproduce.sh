#!/usr/bin/env bash
# Preprocess, train and evaluate lego, drums and sewing, then print the results table.
# usage: COMP_DATA=/path/to/comp_data bash scripts/reproduce.sh   (PREFIX=name_ to rename the runs)
set -euo pipefail
cd "$(dirname "$0")/.."
: "${COMP_DATA:?set COMP_DATA to the unzipped EventNeRF comp_data folder}"
PREFIX="${PREFIX:-}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

python dataloader.py -p data -s lego
python dataloader.py -p data -s drums
python dataloader.py -p data -s sewing --polarity-offset -0.375

for scene in lego drums sewing; do
    name="${PREFIX}${scene}"
    ns-train esplatfacto --data "data/${scene}/output" --experiment-name "$name" --vis tensorboard
    config="$(ls -d "outputs/${name}/esplatfacto/"*/ | tail -1)config.yml"
    gt=()
    [ -d "${COMP_DATA}/${scene}/gt" ] && gt=(--gt-dir "${COMP_DATA}/${scene}/gt")
    python scripts/eval_gt.py --load-config "$config" "${gt[@]}" --out "eval/${name}"
done

python - "$PREFIX" <<'EOF'
import json, sys
print(f"{'scene':8s} {'gaussians':>9s} {'PSNR':>6s} {'SSIM':>6s} {'LPIPS':>6s}  off-orbit mean alpha (0, +30, -30)")
for scene in ("lego", "drums", "sewing"):
    m = json.load(open(f"eval/{sys.argv[1]}{scene}/metrics.json"))
    q = "".join(f" {m[k]:6.3f}" if k in m else "      -" for k in ("psnr", "ssim", "lpips"))
    a = ", ".join(f"{v['mean_alpha']:.2f}" for v in m["alpha"].values())
    print(f"{scene:8s} {m['num_gaussians']:9d}{q}  {a}")
EOF
