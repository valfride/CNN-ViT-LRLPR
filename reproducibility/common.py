"""Standard-library-only identity checks. Never deserialize a model or dataset."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "reproducibility"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_hash(path, expected, label):
    path = Path(path).resolve(strict=True)
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label}: SHA256 mismatch (expected {expected}, got {actual})")
    return path


def paper_manifest():
    return load_json(HERE / "paper_manifest.json")


def assets(table, models=None):
    if table == 5:
        if models:
            raise ValueError("Table5 uses the complete original 18-asset seed manifest")
        return load_json(HERE / "ablation_manifest.json")["assets"]
    manifest = paper_manifest()
    selected = list(manifest["models"]) if not models else models
    if len(selected) != len(set(selected)) or not selected:
        raise ValueError("Select distinct, nonempty model identifiers")
    if any(key not in manifest["models"] for key in selected):
        raise ValueError("Unknown model identifier")
    return [dict(id=key, **manifest["models"][key]) for key in selected]


def check_weights(mapping_path, selected):
    mapping_path = Path(mapping_path).resolve(strict=True)
    mapping = load_json(mapping_path)
    checked = {}
    for asset in selected:
        key = asset["id"]
        if key not in mapping:
            raise ValueError(f"Missing approved checkpoint mapping: {key}")
        path = Path(mapping[key]).expanduser()
        if not path.is_absolute():
            path = mapping_path.parent / path
        checked[key] = require_hash(path, asset["checkpoint_sha256"], key)
        require_hash(ROOT / asset["config"], asset["config_sha256"], asset["config"])
    return checked


def check_dataset(dataset, cohort):
    spec = paper_manifest()["datasets"][cohort]
    path = Path(dataset).expanduser().resolve(strict=True)
    for name, expected in spec["files"].items():
        require_hash(path / name, expected, f"{cohort}/{name}")
    return {"cohort": cohort, "phase": spec["phase"], "samples": spec["samples"],
            "fingerprints": dict(spec["files"])}


def check_sources():
    for name, expected in load_json(HERE / "scientific_source_hashes.json").items():
        require_hash(ROOT / name, expected, name)


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def new_output(path):
    path = Path(path).expanduser().resolve()
    if path == ROOT or ROOT in path.parents:
        raise ValueError("Private outputs must be outside the source checkout")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)
    return path
