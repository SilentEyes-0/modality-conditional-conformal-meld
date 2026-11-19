import hashlib, json, os, random
import numpy as np
import yaml


def load_config(path):
    with open(path, "r", encoding="utf8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def dump_json(obj, path):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf8") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def load_json(path):
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def plan_hash(plan: dict) -> str:
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode("utf8")).hexdigest()
