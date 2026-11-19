#!/usr/bin/env bash
# Real run order on MELD. Read README.md first; check configs/experiment.yaml paths against your download.
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=configs/experiment.yaml
python -m src.run inspect --config $CFG                       # 1. counts, class counts, missing media; compare with official counts
for S in train dev test; do python -m src.features.extract --config $CFG --split $S --modality all; python -m src.features.extract assemble --config $CFG --split $S; done   # 2. features (GPU recommended)
python -m src.run run --config $CFG --mode dev                # 3. development runs (dev subset only)
python -m src.run resplits --config $CFG --k 20               # 4. variability across calibration draws (development)
# 5. FREEZE the analysis plan, then evaluate the test set once
python -m src.run freeze --config $CFG
python -m src.run run --config $CFG --mode final
