"""Pinned Tables1–5 plans and explicitly requested fresh private executions.

No torch, dataset, model, GPU or pickle imports occur in this module.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from .common import (HERE, ROOT, assets, check_dataset, check_sources,
                     check_weights, load_json, new_output, paper_manifest, write_new)

EVALUATOR = ROOT / "test.py"
BENCHMARK = ROOT / "experiments/revision_eval/reviewer1/01_efficiency/scripts/benchmark_efficiency.py"


def plan(table, selected, validation=False, requested_frames=None):
    if validation and table != 2:
        raise ValueError("Validation evidence replay belongs only to Table2")
    frames = [1] if table == 3 else [3, 5] if table == 2 else [1, 3, 5]
    if requested_frames is not None:
        if table == 4 or not requested_frames or len(requested_frames) != len(set(requested_frames)):
            raise ValueError("Frame subsets require distinct values and are unavailable for Table4")
        if any(f not in frames for f in requested_frames):
            raise ValueError("Frame subset is outside this table's frozen protocol")
        frames = requested_frames
    rules = ["bayes", "average", "majority", "char_majority"] if table == 2 else ["bayes"]
    if table == 4:
        return [{"id": a["id"], "table": 4} for a in selected]
    return [{"id": a["id"], "table": table, "frames": f, "fusion": rule}
            for a in selected for f in frames for rule in rules]


def environment():
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               PYTHONDONTWRITEBYTECODE="1")
    return env


def runtime_check():
    """CPU-only library identity probe; never query or initialize a GPU."""
    probe = (
        "import json,platform,torch,torchvision,numpy,yaml; "
        "print(json.dumps(dict(python=platform.python_version(),torch=torch.__version__,"
        "torchvision=torchvision.__version__,numpy=numpy.__version__,"
        "PyYAML=yaml.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version())))"
    )
    env = environment()
    env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True,
                            text=True, check=True)
    observed = json.loads(result.stdout)
    expected = paper_manifest()["runtime"]
    mismatches = {k: {"expected": expected[k], "observed": v}
                  for k, v in observed.items() if v != expected[k]}
    return {"observed": observed, "mismatches": mismatches}


def evaluator_command(asset, dataset, unit, frame, fusion):
    # The original evaluator chooses from a directory. One verified file prevents
    # all directory score/tie policies from changing the frozen model selection.
    return [sys.executable, "-c",
            "import runpy,sys,torch; "
            "torch.backends.cuda.matmul.allow_tf32=False; "
            "torch.backends.cudnn.allow_tf32=True; "
            "torch.use_deterministic_algorithms(False); "
            "torch.set_float32_matmul_precision('highest'); "
            "sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')",
            str(EVALUATOR), "--config", str(ROOT / asset["config"]),
            "--checkpoints", str(unit / "checkpoint"), "--split", str(dataset),
            "--dataset-phase", "test", "--mode", "val", "--in_images", str(frame),
            "--fusion", fusion, "--metrics-json", str(unit / "metrics.json"),
            "--predictions-csv", str(unit / "predictions.csv")]


def execution_device(gpu, allow_mismatch=False):
    """Queried only for an explicitly requested execution, never for preflight."""
    probe = ("import json,torch; "
             "assert torch.cuda.is_available(), 'Paper execution requires CUDA; no silent CPU/precision fallback'; "
             "print(json.dumps(dict(name=torch.cuda.get_device_name(0))))")
    env = environment()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    result = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True,
                            text=True, check=True)
    observed = json.loads(result.stdout)
    if "RTX 8000" not in observed["name"] and not allow_mismatch:
        raise ValueError("GPU differs from original RTX8000; explicit --allow-runtime-mismatch required")
    return observed


def run_logged(command, cwd, env, path):
    with path.open("x", encoding="utf-8") as log:
        subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                       text=True, check=True)


def metric_record(table, asset, spec, payload, cohort):
    if payload["frames"] != spec["frames"] or payload["fusion"] != spec["fusion"]:
        raise ValueError("Evaluator result protocol mismatch")
    if payload["tta"] or payload["swa"] or payload["dataset_phase"] != cohort["phase"]:
        raise ValueError("Unexpected TTA/SWA/dataset phase")
    if payload["mode"] != "val" or payload["cls_loss"] != asset.get("loss", "CE"):
        raise ValueError("Wrong structured-metrics mode or native loss identity")
    expected_runtime = {"seed": 42, "cudnn_benchmark": False, "cudnn_deterministic": True}
    if payload["reproducibility"] != expected_runtime:
        raise ValueError("Unexpected evaluation seed/backend settings")
    if len(payload["checkpoint_files"]) != 1 or Path(payload["checkpoint_files"][0]).name != asset["checkpoint_filename"]:
        raise ValueError("Evaluator did not use the sole pinned checkpoint filename")
    if payload["metrics"]["sequence_total"] != cohort["samples"]:
        raise ValueError("Unexpected evaluated sample count")
    return {"table": table, "id": asset["id"], "model": asset.get("name"),
            "variant": asset.get("variant"), "seed": asset.get("seed"),
            "weight_source": asset.get("weight_source"), "frames": spec["frames"],
            "fusion": spec["fusion"], "checkpoint_sha256": asset["checkpoint_sha256"],
            "config_sha256": asset["config_sha256"], "dataset": cohort,
            "metrics": payload["metrics"]}


def table3_groups(predictions, manifest_path, metrics):
    """Verify the private annotation/GT join; return nonidentifying group counts."""
    samples = load_json(manifest_path)["samples"]
    by_id = {r["track_id"]: r for r in samples}
    if len(samples) != 7023 or len(by_id) != 7023:
        raise ValueError("Wrong or duplicate annotation population")
    seen = set()
    groups = {"Poor": {"correct": 0, "total": 0}, "Illegible": {"correct": 0, "total": 0}}
    exact = partial6 = partial5 = 0
    with Path(predictions).open(newline="", encoding="utf-8") as stream:
        for r in csv.DictReader(stream):
            key = r["track"]
            if key not in by_id or key in seen:
                raise ValueError("Unknown/duplicate annotation in prediction output")
            seen.add(key)
            source = by_id[key]
            gt = r["ground_truth"]
            pred = r["prediction"]
            if gt != source["ocr"] or len(gt) != 7:
                raise ValueError("Ground-truth/manifest mismatch")
            correct = pred == gt
            positions = sum(i < len(pred) and pred[i] == gt[i] for i in range(7))
            if r["correct"].lower() not in {"true", "false"}:
                raise ValueError("Invalid exact-match flag")
            if correct != (r["correct"].lower() == "true") or positions != int(r["positional_matches"]):
                raise ValueError("Prediction metric mismatch")
            exact += int(correct)
            partial6 += int(positions >= 6)
            partial5 += int(positions >= 5)
            group = groups[source["source_legibility"]]
            group["correct"] += int(correct)
            group["total"] += 1
    if seen != set(by_id) or [groups[k]["total"] for k in ["Poor", "Illegible"]] != [6240, 783]:
        raise ValueError("Incomplete source-group accounting")
    if [exact, partial6, partial5] != [metrics[k] for k in
                                      ["sequence_correct", "partial_6_correct", "partial_5_correct"]]:
        raise ValueError("CSV/JSON counts disagree")
    return groups


def execute(args, selected, checked, cohort, specs, runtime):
    runtime["device"] = execution_device(args.gpu, args.allow_runtime_mismatch)
    output = new_output(args.output)
    write_new(output / "run.json", {"table": args.table, "validation": args.validation,
              "plan": specs, "dataset": cohort, "runtime": runtime,
              "paper_commit": paper_manifest()["paper_commit"],
              "paper_reference": paper_manifest()["paper_reference"],
              "status": "started; aggregate.json exists only on successful completion"})
    dataset = Path(args.dataset).expanduser().resolve(strict=True)
    env = environment()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.table == 4:
        keys = {"mdiff4str": "mdiff"}
        command = [sys.executable, str(BENCHMARK), "--models"] + [keys.get(a["id"], a["id"]) for a in selected]
        command += ["--dataset", str(dataset), "--output-dir", str(output / "benchmark"),
                    "--precision", "fp16", "--gpu", args.gpu, "--sample-count", "100",
                    "--warmup", "30", "--iterations", "100", "--flop-samples", "10",
                    "--hash-checkpoints", "--rebuild-input-cache", "--strict"]
        for asset in selected:
            command += ["--checkpoint-override", f"{keys.get(asset['id'], asset['id'])}={checked[asset['id']]}"]
        run_logged(command, ROOT, env, output / "execution.log")
        print(f"Benchmark finished. Verify {output / 'benchmark/efficiency_summary.csv'} separately.")
        return
    records = []
    by_id = {a["id"]: a for a in selected}
    for spec in specs:
        asset = by_id[spec["id"]]
        unit = output / f"{asset['id']}_F{spec['frames']}_{spec['fusion']}"
        (unit / "checkpoint").mkdir(parents=True, exist_ok=False)
        (unit / "checkpoint" / asset["checkpoint_filename"]).symlink_to(checked[asset["id"]])
        command = evaluator_command(asset, dataset, unit, spec["frames"], spec["fusion"])
        phase_index = command.index("--dataset-phase") + 1
        command[phase_index] = cohort["phase"]
        write_new(unit / "command.json", command)
        run_logged(command, ROOT, env, unit / "execution.log")
        payload = load_json(unit / "metrics.json")
        record = metric_record(args.table, asset, spec, payload, cohort)
        if args.table == 3:
            record["groups"] = table3_groups(unit / "predictions.csv", dataset / "selection_manifest.json", payload["metrics"])
        records.append(record)
    write_new(output / "aggregate.json", {"schema_version": 1, "table": args.table,
              "validation": args.validation, "runtime": runtime, "records": records,
              "paper_reference": paper_manifest()["paper_reference"]})
    print(f"Finished {len(records)} evaluations; now verify {output / 'aggregate.json'}.")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["plan", "preflight", "evaluate"])
    p.add_argument("--table", type=int, choices=range(1, 6), required=True)
    p.add_argument("--models", nargs="+")
    p.add_argument("--frames", nargs="+", type=int,
                   help="Diagnostic subset of this table's frozen frame counts; complete-table verification still requires all counts")
    p.add_argument("--weights", type=Path, help="Private JSON: model/asset id -> original checkpoint file")
    p.add_argument("--dataset", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--validation", action="store_true", help="Table2 validation evidence only; never reselect")
    p.add_argument("--runtime-check", action="store_true", help="CPU-only dependency probe, hides GPUs")
    p.add_argument("--allow-runtime-mismatch", action="store_true", help="Record noncanonical runtime explicitly")
    p.add_argument("--gpu", default="0", help="GPU visible only inside an explicitly executed child")
    p.add_argument("--execute", action="store_true", help="Explicitly request expensive execution")
    args = p.parse_args(argv)
    try:
        selected = assets(args.table, args.models)
        specs = plan(args.table, selected, args.validation, args.frames)
        if args.action == "plan":
            print(json.dumps({"table": args.table, "runs": specs, "inference_executed": False}, indent=2))
            return
        if not args.weights or not args.dataset:
            raise ValueError("--weights and --dataset are required for preflight/evaluate")
        if args.action == "evaluate" and (not args.execute or not args.output):
            raise ValueError("Execution requires --execute and a NEW private --output outside the checkout")
        check_sources()
        checked = check_weights(args.weights, selected)
        cohort_name = "lplcv2" if args.table == 3 else "validation" if args.validation else "test"
        cohort = check_dataset(args.dataset, cohort_name)
        runtime = runtime_check() if args.runtime_check or args.action == "evaluate" else {"status": "not probed"}
        if runtime.get("mismatches") and not args.allow_runtime_mismatch:
            raise ValueError("Runtime differs from frozen environment: " + json.dumps(runtime["mismatches"]))
        if args.action == "preflight":
            print(json.dumps({"identities": "pass", "assets": list(checked), "dataset": cohort,
                              "runtime": runtime, "inference_executed": False}, indent=2))
            return
        execute(args, selected, checked, cohort, specs, runtime)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        p.exit(1, f"Reproduction stopped: {exc}\n")


if __name__ == "__main__":
    main()
