"""Verify paper aggregates without model imports, private predictions or GPU use."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from .common import HERE, assets, load_json, paper_manifest


def reference(name):
    with (HERE / "reference" / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def count(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Invalid nonnegative integer {label}")
    return value


def validate_metrics(metrics):
    n = count(metrics["sequence_total"], "sequence_total")
    values = [count(metrics[k], k) for k in
              ["sequence_correct", "partial_6_correct", "partial_5_correct"]]
    if not n or not 0 <= values[0] <= values[1] <= values[2] <= n:
        raise ValueError("Impossible exact/partial metric accounting")
    for key, correct in zip(["sequence_accuracy_percent", "partial_6_percent", "partial_5_percent"], values):
        if key in metrics and (not math.isfinite(metrics[key]) or
                               abs(metrics[key] - 100 * correct / n) > 1e-8):
            raise ValueError(f"Incorrect percentage: {key}")
    return n, values


def same_counts(metrics, expected):
    validate_metrics(metrics)
    for key, value in expected.items():
        if key.endswith(("correct", "total", "errors", "characters")):
            count(metrics.get(key), key)
            if metrics.get(key) != value:
                raise ValueError(f"Frozen count differs: {key} ({metrics[key]} vs {value})")


def table3_display_verification():
    """Separate reported displays, integer arithmetic and approved exceptions.

    Never substitute a manuscript display into an evaluator's metric payload.
    Approval applies only to the three exact ledger entries, not to tolerances.
    """
    counts = reference("TABLE3_COUNTS.csv")
    reported = reference("TABLE3_EXPECTED.csv")
    expected_models = {a["name"] for a in assets(3)}
    if (len(counts) != 8 or len(reported) != 8 or
            {r["model"] for r in counts} != expected_models or
            {r["model"] for r in reported} != expected_models):
        raise ValueError("Incomplete/duplicate Table3 model population")
    by_model = {r["model"]: r for r in reported}
    ledger = load_json(HERE / "reference/PAPER_DISPLAY_DISCLOSURES.json")
    if ledger.get("approval", {}).get("status") != "approved":
        raise ValueError("Historical display differences lack explicit approval")
    differences = ledger["differences"]
    allowed = {(r["model"], r["metric"]): r for r in differences}
    if len(differences) != 3 or len(allowed) != 3:
        raise ValueError("Approval covers exactly three distinct historical displays")
    found = []
    for r in counts:
        total = int(r["total"])
        if (total != 7023 or int(r["poor_total"]) != 6240 or int(r["illegible_total"]) != 783 or
                int(r["poor_correct"]) + int(r["illegible_correct"]) != int(r["correct"])):
            raise ValueError("External subgroup total mismatch")
        if not 0 <= int(r["correct"]) <= int(r["partial6_correct"]) <= int(r["partial5_correct"]) <= total:
            raise ValueError("Invalid external exact/partial counts")
        for metric, numerator, denominator in [
                ("exact_percent", "correct", "total"),
                ("partial6_percent", "partial6_correct", "total"),
                ("partial5_percent", "partial5_correct", "total"),
                ("poor_exact_percent", "poor_correct", "poor_total"),
                ("illegible_exact_percent", "illegible_correct", "illegible_total")]:
            n, d = int(r[numerator]), int(r[denominator])
            if not 0 <= n <= d:
                raise ValueError("Invalid external group count")
            arithmetic = 100 * n / d
            direct = f"{arithmetic:.1f}"
            display = f"{float(by_model[r['model']][metric]):.1f}"
            if direct == display:
                continue
            key = (r["model"], metric)
            approved = allowed.get(key)
            if not approved or (n, d, display, direct, f"{arithmetic:.2f}") != (
                    approved["numerator"], approved["denominator"], approved["paper_display"],
                    approved["direct_one_decimal"], approved["archived_two_decimal"]):
                raise ValueError("Unapproved or altered Table3 display/arithmetic difference")
            found.append(dict(approved, direct_percentage_from_counts=arithmetic,
                              approval_status="approved_reported_display_only"))
    if {(r["model"], r["metric"]) for r in found} != set(allowed):
        raise ValueError("Stale paper-display approval ledger")
    return {
        "reported_manuscript_values": {"table": 3, "status": "pass", "cells_checked": 40,
            "approved_reported_values": [{"model": r["model"], "metric": r["metric"],
                                           "reported_percent": r["paper_display"]} for r in found],
            "basis": "Authoritative reported reference values, including three approved historical displays"},
        "direct_count_arithmetic": {"status": "calculated_from_original_counts",
            "direct_rounding_matches_reported_cells": 37,
            "direct_rounding_differs_from_reported_cells": 3,
            "differences": [{"model": r["model"], "metric": r["metric"],
                             "numerator": r["numerator"], "denominator": r["denominator"],
                             "percentage": r["direct_percentage_from_counts"],
                             "direct_one_decimal": r["direct_one_decimal"],
                             "reported_percent": r["paper_display"]} for r in found],
            "note": "Direct rounding does not produce the three manuscript displays"},
        "approved_historical_display_differences": {"status": "approved_and_disclosed",
            "approval": ledger["approval"], "differences": found},
        "paper_display_differences": found,
    }


def paper_reference_status():
    return paper_manifest()["paper_reference"]


def check_reference_integrity():
    t1 = reference("TABLE1_EXPECTED.csv")
    t2 = reference("TABLE2_EXPECTED.csv")
    if len(t1) != 24 or len(t2) != 64:
        raise ValueError("Incomplete Table1/Table2 references")
    lookup = {(r["model"], r["F"]): r for r in t1}
    for r in t1:
        n = int(r["total"])
        values = [int(r[k]) for k in ["correct", "partial6_correct", "partial5_correct"]]
        if n != 3000 or not values[0] <= values[1] <= values[2] <= n:
            raise ValueError("Invalid Table1 reference counts")
        for k, c in zip(["test_RR", "partial6_percent", "partial5_percent"], values):
            if abs(float(r[k]) - 100 * c / n) > 1e-8:
                raise ValueError("Reference count/percentage mismatch")
    keys = set()
    for r in t2:
        key = (r["model"], r["F"], r["fusion"])
        if key in keys:
            raise ValueError("Duplicate fusion reference")
        keys.add(key)
        if int(r["total"]) != 3000 or abs(float(r["test_RR_percent"]) - 100 * int(r["correct"]) / 3000) > 1e-8:
            raise ValueError("Invalid Table2 arithmetic")
        if r["validation_selected"] != "bayes":
            raise ValueError("Frozen validation selection changed")
        if r["fusion"] == "bayes" and r["correct"] != lookup[(r["model"], r["F"])]["correct"]:
            raise ValueError("Table1/Table2 PRF inconsistency")
    display_verification = table3_display_verification()
    ablation = load_json(HERE / "ablation_manifest.json")["assets"]
    summaries = {}
    for variant in ["full", "no_restormer", "no_pixelshuffle", "no_sfb", "linear_head", "primary_instead_of_ema"]:
        selected = [a for a in ablation if
                    (a["weight_source"] == "student" if variant == "primary_instead_of_ema" else
                     a["weight_source"] == "ghost" and a["variant"] == variant)]
        if len(selected) != 3 or {a["seed"] for a in selected} != {42, 123, 2026}:
            raise ValueError("Missing original ablation seed")
        for frame in [1, 3, 5]:
            rr = [100 * a["expected"][str(frame)]["sequence_correct"] /
                  a["expected"][str(frame)]["sequence_total"] for a in selected]
            summaries[(variant, str(frame))] = (statistics.mean(rr), statistics.stdev(rr))
    for r in reference("TABLE5_EXPECTED.csv"):
        actual = summaries[(r["variant"], r["F"])]
        expected = [float(r["mean_RR"]), float(r["sample_std_RR"])]
        if any(abs(a - b) > 1e-10 for a, b in zip(actual, expected)):
            raise ValueError("Table5 mean/sample std differ from per-seed counts")
    return {"Table1_cells": 72, "Table2_cells": 64, "Table3_cells": 40,
            "Table5_mean_std_pairs": 18, "raw_predictions_required": False,
            "paper_reference": paper_reference_status(), **display_verification}


def verify_aggregate(document):
    table = document["table"]
    validation = document.get("validation", False)
    if (isinstance(table, bool) or not isinstance(table, int) or
            not isinstance(validation, bool) or table not in [1, 2, 3, 5] or (validation and table != 2)):
        raise ValueError("Unexpected table/cohort")
    records = document["records"]
    expected_assets = {a["id"]: a for a in assets(table)}
    frames = [1] if table == 3 else [3, 5] if table == 2 else [1, 3, 5]
    rules = ["bayes", "average", "majority", "char_majority"] if table == 2 else ["bayes"]
    expected_keys = {(key, frame, rule) for key in expected_assets for frame in frames for rule in rules}
    seen = set()
    t1 = {(r["model"], int(r["F"])): r for r in reference("TABLE1_EXPECTED.csv")}
    t2 = {(r["model"], int(r["F"]), r["fusion"]): r for r in
          reference("VALIDATION_SELECTION.csv" if validation else "TABLE2_EXPECTED.csv")}
    t3 = {r["model"]: r for r in reference("TABLE3_COUNTS.csv")}
    cohort_name = "lplcv2" if table == 3 else "validation" if validation else "test"
    cohort = paper_manifest()["datasets"][cohort_name]
    for r in records:
        count(r["frames"], "frames")
        key = (r["id"], r["frames"], r["fusion"])
        if key not in expected_keys or key in seen:
            raise ValueError("Unknown/duplicate output record")
        seen.add(key)
        a = expected_assets[r["id"]]
        if r.get("table") != table:
            raise ValueError("Record table mismatch")
        if r["checkpoint_sha256"] != a["checkpoint_sha256"] or r["config_sha256"] != a["config_sha256"]:
            raise ValueError("Wrong frozen model/config identity")
        if r["dataset"] != {"cohort": cohort_name, "phase": cohort["phase"],
                            "samples": cohort["samples"], "fingerprints": cohort["files"]}:
            raise ValueError("Wrong frozen dataset identity")
        m = r["metrics"]
        n, _ = validate_metrics(m)
        if n != cohort["samples"]:
            raise ValueError("Wrong metric denominator")
        if table == 5:
            if (r.get("seed"), r.get("variant"), r.get("weight_source")) != (a["seed"], a["variant"], a["weight_source"]):
                raise ValueError("Wrong ablation run identity")
            same_counts(m, a["expected"][str(r["frames"])])
            continue
        if r["model"] != a["name"]:
            raise ValueError("Model label mismatch")
        ref = t1[(a["name"], r["frames"])] if table == 1 else t2[(a["name"], r["frames"], r["fusion"])] if table == 2 else t3[a["name"]]
        if m["sequence_correct"] != int(ref["correct"]):
            raise ValueError(f"Exact count differs for {key}")
        if table in [1, 3]:
            if [m["partial_6_correct"], m["partial_5_correct"]] != [int(ref["partial6_correct"]), int(ref["partial5_correct"])]:
                raise ValueError(f"Partial recognition counts differ for {key}")
        if table == 3:
            for group, prefix in [("Poor", "poor"), ("Illegible", "illegible")]:
                count(r["groups"][group]["correct"], "group correct")
                count(r["groups"][group]["total"], "group total")
                if r["groups"][group] != {"correct": int(ref[prefix + "_correct"]), "total": int(ref[prefix + "_total"])}:
                    raise ValueError("External subgroup differs")
    if seen != expected_keys:
        raise ValueError(f"Incomplete table: {len(expected_keys - seen)} missing records")
    display_verification = table3_display_verification() if table == 3 else {"paper_display_differences": []}
    return {"table": table, "records": len(seen), "frozen_counts": "match",
            "paper_reference": paper_reference_status(), **display_verification,
            "runtime": document.get("runtime"), "note": "Aggregate consistency; not independent proof of execution"}


def verify_efficiency(path, strict_timing=False):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected = {r["model"]: r for r in reference("TABLE4_ARCHIVED.csv")}
    observed = {r["model"]: r for r in rows}
    if len(rows) != 8 or len(observed) != 8 or set(observed) != set(expected):
        raise ValueError("Efficiency requires all eight models exactly once")
    arithmetic = ["params_m", "f1_gflops_mean", "f5_batched_gflops_mean"]
    hardware = ["f1_latency_ms_mean", "f1_latency_ms_std", "f5_batched_latency_ms_mean",
                "f5_batched_latency_ms_std", "f1_peak_memory_mib", "f5_batched_peak_memory_mib"]
    differences = []
    for model, r in observed.items():
        if r["precision"] != "fp16":
            raise ValueError("Table4 requires FP16")
        for key in arithmetic + hardware:
            digits = 1 if "memory" in key else 3
            value = float(r[key])
            if not math.isfinite(value) or value < 0:
                raise ValueError("Invalid efficiency metric")
            if f"{value:.{digits}f}" != f"{float(expected[model][key]):.{digits}f}":
                differences.append({"model": model, "metric": key, "observed": value,
                                    "reference": float(expected[model][key])})
                if key in arithmetic or strict_timing:
                    raise ValueError(f"Table4 mismatch: {model}/{key}")
    return {"table": 4, "models": 8, "arithmetic": "match", "hardware_differences": differences,
            "status": "exact rounded reference match" if not differences else
                      "hardware measurements differ; no tolerance/equivalence claim made"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--references", action="store_true")
    g.add_argument("--aggregate", type=Path)
    g.add_argument("--efficiency", type=Path)
    p.add_argument("--strict-timing", action="store_true")
    p.add_argument("--strict-paper-display", action="store_true",
                   help="Require approved manuscript display consistency; the three disclosed approvals are permitted")
    p.add_argument("--strict-direct-rounding", action="store_true",
                   help="Additionally require direct count rounding to equal every paper display (fails for the three approved differences)")
    args = p.parse_args(argv)
    try:
        result = check_reference_integrity() if args.references else verify_aggregate(load_json(args.aggregate)) if args.aggregate else verify_efficiency(args.efficiency, args.strict_timing)
        if args.strict_paper_display and result.get("reported_manuscript_values", {}).get("status", "pass") != "pass":
            raise ValueError("Reported manuscript values do not match approved references")
        if args.strict_direct_rounding and result.get("paper_display_differences"):
            raise ValueError("Direct rounding differs from three approved manuscript displays; original counts and reported values are preserved")
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        p.exit(1, f"Verification failed: {exc}\n")


if __name__ == "__main__":
    main()
