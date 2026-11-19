#!/usr/bin/env bash
# End-to-end smoke test on SYNTHETIC features (no MELD needed). Results are not scientific results.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/make_synthetic.py data/features_synthetic
python - <<'PY'
import yaml
c = yaml.safe_load(open("configs/experiment.yaml"))
c["paths"] = {"features": "data/features_synthetic", "outputs": "outputs_synthetic"}
c["train"]["seeds"] = [0, 1]; c["train"]["epochs"] = 10
c["analysis_plan"]["bootstrap_B"] = 100
c["analysis_plan"]["corruption"] = [{"name": "audio_noise", "modality": "audio", "calibration": "matched"},
                                    {"name": "audio_noise", "modality": "audio", "calibration": "mismatched"}]
yaml.safe_dump(c, open("configs/smoke.yaml", "w"))
PY
python -m src.run run --config configs/smoke.yaml --mode dev
echo "Outputs in outputs_synthetic/ (summary_conformal_val.csv, worst_pattern_val.csv, summary_classification_val.csv)"
