"""Read-only comparison of active manuscript Tables 1–5 with frozen references.

This deliberately supports the reviewed manuscript's table layout, not arbitrary
LaTeX. Missing, duplicate or structurally changed rows fail rather than guessing.
No manuscript files are copied, edited, compiled or executed.
"""
import argparse
import json
from pathlib import Path
import re

from .common import sha256
from .verify import reference

LABELS = ("baseline_comparison", "fusion_controls", "lplcv2_cross_dataset",
          "efficiency", "ablation")
MODELS = ("MDiff4STR", "CPPD", "LISTER", "IGTR", "SVTRv2-AR", "OTE", "SVTRv2", "Ours")


def uncomment(text):
    return "\n".join(re.split(r"(?<!\\)%", line, maxsplit=1)[0]
                     for line in text.splitlines())


def tables(text):
    found = {}
    for block in re.findall(r"\\begin\{table\*?\}.*?\\end\{table\*?\}",
                            uncomment(text), re.S):
        for label in LABELS:
            if "\\label{tab:" + label + "}" in block:
                if label in found:
                    raise ValueError("Duplicate active table: " + label)
                found[label] = block
    if set(found) != set(LABELS):
        raise ValueError("Missing active Tables 1–5")
    return found


def numbers(text):
    return re.findall(r"\d+(?:\.\d+)?", text)


def model_rows(block, cells, temporal=False):
    rows, model = {}, None
    for line in block.splitlines():
        first = line.split("&")[0]
        for candidate in MODELS:
            if re.search(re.escape(candidate) + r"(?![\w-])", first):
                model = candidate
                break
        if "&" not in line or model is None:
            continue
        values = numbers(" & ".join(line.split("&")[1:]))
        if len(values) != cells:
            continue
        key = (model, int(values[0])) if temporal else model
        if key in rows:
            raise ValueError("Duplicate manuscript row: " + str(key))
        rows[key] = values[1:] if temporal else values
    return rows


def compare(actual, expected, label):
    if actual != expected:
        missing = sorted(str(k) for k in set(expected) - set(actual))
        extra = sorted(str(k) for k in set(actual) - set(expected))
        changed = {str(k): {"paper": actual[k], "reference": expected[k]}
                   for k in actual.keys() & expected.keys() if actual[k] != expected[k]}
        raise ValueError(label + ": " + json.dumps(dict(missing=missing, extra=extra, changed=changed)))
    return sum(len(v) for v in expected.values())


def audit(path):
    blocks = tables(Path(path).read_text(encoding="utf-8"))
    checked = {}
    expected = {(r["model"], int(r["F"])):
                [f"{float(r[k]):.1f}" for k in ("test_RR", "partial6_percent", "partial5_percent")]
                for r in reference("TABLE1_EXPECTED.csv")}
    checked["Table1"] = compare(model_rows(blocks[LABELS[0]], 4, True), expected, "Table1")
    rows = reference("TABLE2_EXPECTED.csv")
    expected = {}
    for r in rows:
        expected[(r["model"], int(r["F"]))] = [
            next(x["rounded_RR"] for x in rows if x["model"] == r["model"]
                 and x["F"] == r["F"] and x["fusion"] == rule)
            for rule in ("bayes", "average", "majority", "char_majority")]
    checked["Table2"] = compare(model_rows(blocks[LABELS[1]], 5, True), expected, "Table2")
    expected = {r["model"]: [r[k] for k in ("exact_percent", "partial6_percent", "partial5_percent",
                                             "poor_exact_percent", "illegible_exact_percent")]
                for r in reference("TABLE3_EXPECTED.csv")}
    checked["Table3"] = compare(model_rows(blocks[LABELS[2]], 5), expected, "Table3")
    columns = (("params_m", 3), ("f1_gflops_mean", 3), ("f1_latency_ms_mean", 3),
               ("f1_latency_ms_std", 3), ("f1_peak_memory_mib", 1),
               ("f5_batched_gflops_mean", 3), ("f5_batched_latency_ms_mean", 3),
               ("f5_batched_latency_ms_std", 3), ("f5_batched_peak_memory_mib", 1))
    expected = {r["model"]: [f"{float(r[k]):.{places}f}" for k, places in columns]
                for r in reference("TABLE4_ARCHIVED.csv")}
    checked["Table4"] = compare(model_rows(blocks[LABELS[3]], 9), expected, "Table4")
    names = {"Full model": "full", "without Restormer": "no_restormer",
             "without PixelShuffle": "no_pixelshuffle", "without SFB": "no_sfb",
             "Linear classifier": "linear_head", "Primary instead of EMA": "primary_instead_of_ema"}
    actual = {}
    for line in blocks[LABELS[4]].splitlines():
        for name, variant in names.items():
            if name in line.split("&")[0] and "&" in line:
                if variant in actual:
                    raise ValueError("Duplicate ablation row")
                actual[variant] = numbers(" & ".join(line.split("&")[1:]))
    expected = {}
    for r in reference("TABLE5_EXPECTED.csv"):
        expected.setdefault(r["variant"], []).extend(f"{float(r[k]):.1f}" for k in ("mean_RR", "sample_std_RR"))
    checked["Table5"] = compare(actual, expected, "Table5")
    active = uncomment(Path(path).read_text())
    if re.search(r"\\input\{[^}]*ablation_combined", active):
        raise ValueError("Optional combined-ablation draft is active; author review required")
    return dict(status="PASS_active_table_displays", cells_checked=checked,
                manuscript_sha256=sha256(path), optional_combined_table="inactive",
                scope="Displayed Tables 1–5 only; not a claim of experimental reproduction or prose equivalence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(audit(args.paper_root / "manuscript/4-experiments/experiments.tex"), indent=2))
    except (ValueError, OSError, KeyError, StopIteration) as exc:
        parser.exit(1, f"Paper comparison stopped: {exc}\n")


if __name__ == "__main__":
    main()
