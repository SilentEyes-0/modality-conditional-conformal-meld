"""Create SYNTHETIC cached features to smoke-test the pipeline. Meaningless scientifically."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.synthetic import write_synthetic_dataset, corrupt_modality

out = sys.argv[1] if len(sys.argv) > 1 else "data/features_synthetic"
print(write_synthetic_dataset(out, seed=0, sizes=(160, 24, 48)))
for sp in ["train", "dev", "test"]:
    corrupt_modality(out, sp, "audio", "audio_noise", 3.0)
print("wrote", out)
