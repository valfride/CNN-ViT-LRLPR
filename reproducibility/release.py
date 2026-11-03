"""Prepare and verify a private, history-free source staging tree. Never publish."""
import argparse
import json
from pathlib import Path
import subprocess

from .common import ROOT, check_sources, new_output, sha256, write_new

EXCLUDED = {
    "datasets/augtest.py": "Historical visualization utility; outside paper execution",
    "datasets/backup/wrappers.py": "Unused backup and unavailable optional synthetic engine",
    "datasets/wrappers (copy).py": "Unused historical duplicate",
    "models/custom/VSR_curvature_att copy.py": "Unused historical model duplicate",
    "misc4torch/translate_weights.py": "Historical import-time checkpoint rewriting utility; never a paper recipe",
    "RELEASE_READINESS_REPORT.md": "Private release-gate report",
    "CLEAN_EXPORT_AUDIT.md": "Private staging audit",
    "READER_VALIDATION_REPORT.md": "Private reader-validation report",
    "FINAL_PAPER_CODE_ALIGNMENT.md": "Private provisional author comparison",
    "PUBLICATION_DECISIONS.md": "Private author/permission decision queue",
}
FORBIDDEN_SUFFIXES = {".pth", ".pt", ".ckpt", ".pkl", ".mdb", ".jpg", ".jpeg",
                      ".png", ".zip", ".tar", ".gz", ".pdf", ".log", ".pyc"}


def validate_name(name):
    p = Path(name)
    if p.is_absolute() or ".." in p.parts or ".git" in p.parts:
        raise ValueError("Unsafe export path")
    if p.suffix.lower() in FORBIDDEN_SUFFIXES or name.endswith("_predictions.csv"):
        raise ValueError("Model/data/output artifacts are forbidden in source export")


def export(output, redact=()):
    check_sources()
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    stage = new_output(output)
    stage.chmod(0o700)
    destination = stage / "source"
    destination.mkdir(mode=0o700)
    omitted, transformed = {}, []
    for name in sorted(n for n in tracked if n):
        validate_name(name)
        if name in EXCLUDED:
            omitted[name] = EXCLUDED[name]
            continue
        source = ROOT / name
        if source.is_symlink() or not source.is_file():
            raise ValueError("Export accepts regular tracked files only")
        content = source.read_bytes()
        if b"\0" in content:
            raise ValueError("Unexpected binary content")
        text = content.decode("utf-8")
        # Only non-executable documentation/provenance wording may be redacted.
        # Configuration snapshots and Python execution paths remain byte-identical.
        if source.suffix in {".md", ".json", ".csv"}:
            for value in redact:
                if value and value in text:
                    text = text.replace(value, "controlled host")
                    if name not in transformed:
                        transformed.append(name)
            content = text.encode("utf-8")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(content)
        target.chmod(0o600)
    ledger_path = destination / "reproducibility/scientific_source_hashes.json"
    original = json.loads(ledger_path.read_text())
    ledger = {name: sha256(destination / name) for name in original if (destination / name).is_file()}
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    transformed.append("reproducibility/scientific_source_hashes.json")
    files = []
    for p in sorted(destination.rglob("*")):
        if p.is_symlink():
            raise ValueError("Unexpected staged symlink")
        if p.is_file():
            name = p.relative_to(destination).as_posix()
            validate_name(name)
            files.append(dict(path=name, bytes=p.stat().st_size, sha256=sha256(p)))
    write_new(stage / "EXPORT_FILES.json", files)
    write_new(stage / "EXPORT_AUDIT.json", dict(status="PASS_source_staging_only",
              files=len(files), omitted=omitted, transformed=transformed,
              source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              source_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
              publication="BLOCKED pending rightsholder, identity and author approval; no license applied",
              limitations="No history, binary artifacts, or private decision reports copied. Full file review remains required."))
    verify(stage)
    return stage


def verify(stage):
    stage = Path(stage).resolve()
    manifest = json.loads((stage / "EXPORT_FILES.json").read_text())
    expected = {r["path"]: r for r in manifest}
    if len(expected) != len(manifest):
        raise ValueError("Duplicate export manifest entry")
    actual = set()
    for p in (stage / "source").rglob("*"):
        validate_name(p.relative_to(stage / "source").as_posix())
        if p.is_symlink():
            raise ValueError("Staged symlink forbidden")
        if p.is_file():
            name = p.relative_to(stage / "source").as_posix()
            validate_name(name)
            if name not in expected or sha256(p) != expected[name]["sha256"] or p.stat().st_size != expected[name]["bytes"]:
                raise ValueError("Staged file missing, changed, or unlisted")
            actual.add(name)
    if actual != set(expected):
        raise ValueError("Export manifest completeness mismatch")
    return len(actual)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--verify")
    parser.add_argument("--redact-text", action="append", default=[])
    args = parser.parse_args()
    if bool(args.output) == bool(args.verify):
        parser.error("Select one of --output NEW-private-directory or --verify staging-directory")
    if args.verify:
        print("Verified staged files:", verify(args.verify))
    else:
        print("Private source staging prepared:", export(args.output, args.redact_text))


if __name__ == "__main__":
    main()
