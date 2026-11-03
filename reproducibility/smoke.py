"""Bounded synthetic GPU0 compatibility checks, never numerical reproduction.

Controller uses only the standard library; each checkpoint runs in a fresh worker.
No dataset is opened and no decoded sample text is exported.
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from .common import (ROOT, HERE, assets, check_sources, check_weights, load_json,
                     new_output, require_hash, write_new)
from .run import environment, runtime_check

RULES = ("bayes", "average", "majority", "char_majority")
BENCHMARK = ROOT / "experiments/revision_eval/reviewer1/01_efficiency/scripts/benchmark_one.py"


def selected_assets():
    return assets(1) + [a for a in assets(5) if a["seed"] == 42]


def units(asset):
    return [(f, r) for f in (1, 3, 5) for r in RULES] if "name" in asset else [(1, "bayes")]


def gpu_snapshot():
    def query(args):
        return subprocess.check_output(["nvidia-smi", *args, "--format=csv,noheader,nounits"],
                                       text=True, timeout=10).strip().splitlines()
    row = query(["--id=0", "--query-gpu=uuid,name,memory.total,memory.free,utilization.gpu"])[0]
    uuid, name, total, free, util = [s.strip() for s in row.split(",")]
    processes = []
    for row in query(["--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory"]):
        parts = [s.strip() for s in row.split(",")]
        if parts[0] == uuid:
            processes.append(dict(pid=int(parts[1]), executable=parts[2], memory_mib=parts[3]))
    return dict(uuid=uuid, name=name, total_mib=int(total), free_mib=int(free),
                utilization=int(util), processes=processes)


def guard(snapshot, expected_uuid, baseline=None, active=False):
    if snapshot["uuid"] != expected_uuid or "RTX 8000" not in snapshot["name"]:
        raise RuntimeError("Unexpected physical GPU0 identity")
    if snapshot["free_mib"] < 40000 or (not active and snapshot["utilization"] > 10):
        raise RuntimeError("GPU0 resource guard: require 40000 MiB free and idle before execution")
    if baseline is not None:
        observed = {p["pid"] for p in snapshot["processes"]}
        if not observed <= set(baseline):
            raise RuntimeError("Unexpected competing GPU0 process")


def ready(expected_uuid):
    for attempt in range(3):
        snapshot = gpu_snapshot()
        try:
            guard(snapshot, expected_uuid)
            return snapshot
        except RuntimeError:
            if attempt == 2:
                raise
            time.sleep(2)


def validate_decoded(prediction, raw, normalized, evaluator):
    if not isinstance(prediction, str) or any(c not in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" for c in prediction):
        raise ValueError("Native decoder must yield normalized alphanumeric text")
    if not math.isfinite(raw) or not math.isfinite(normalized):
        raise ValueError("Nonfinite decoded confidence")
    # Variable-length/empty native predictions are legitimate on synthetic input.
    if not 0 <= evaluator.positional_match_count(prediction, "0" * 7) <= 7:
        raise ValueError("Native output incompatible with fixed seven-position metric")
    return dict(length=len(prediction), fixed_seven_position_interface=True,
                raw_confidence_finite=True, normalized_confidence_finite=True)


def normalize_gpu_uuid(value):
    """PyTorch omits nvidia-smi's GPU- prefix for the same physical UUID."""
    return str(value).removeprefix("GPU-").lower()


def worker(args):
    # Identity enforcement deliberately precedes torch import and deserialization.
    selected = {a["id"]: a for a in selected_assets()}
    asset = selected[args.worker]
    weight = check_weights(args.weights, [asset])[asset["id"]]
    check_sources()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    before = ready(args.expected_gpu_uuid)
    import numpy as np
    import torch
    import yaml
    import datasets
    import models
    import test as evaluator
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA device required; no CPU fallback")
    props = torch.cuda.get_device_properties(0)
    actual_uuid = str(props.uuid)
    if normalize_gpu_uuid(actual_uuid) != normalize_gpu_uuid(args.expected_gpu_uuid):
        raise RuntimeError(f"PyTorch physical-device UUID mismatch: {actual_uuid}")
    torch.set_num_threads(2)
    evaluator.utils.setup_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats(0)
    config = yaml.safe_load((ROOT / asset["config"]).read_text())
    model = models.make(config["model_g"])
    checkpoint = torch.load(weight, map_location="cpu", weights_only=False)
    preferred = "model_g_sd" if asset.get("weight_source") == "student" else None
    state, source = evaluator.state_dict_from_checkpoint(checkpoint, preferred)
    evaluator.load_model_state(model, state, weight.name)
    del checkpoint, state
    model = model.to("cuda:0").eval()
    converter = evaluator.strLabelConverter(config.get("alphabet", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    # A deterministic RGB gradient, no real plate image, identifier, label or dataset.
    y, x = np.mgrid[:13, :37]
    image = np.stack(((x * 7 + y * 3) % 256, (x * 3 + y * 11) % 256,
                      (x * 13 + y * 5) % 256), axis=-1).astype(np.uint8)
    class Synthetic:
        def __len__(self):
            return 1
        def __getitem__(self, index):
            return dict(img_path="synthetic/single/input", name="input", img_raw=image, gt="")
    spec = config["val_dataset"]["wrapper"]
    wrapper_args = dict(spec["args"], dataset=Synthetic(), test=True)
    result = dict(id=asset["id"], config=asset["config"], config_sha256=asset["config_sha256"],
                  checkpoint_filename=weight.name, checkpoint_sha256=asset["checkpoint_sha256"],
                  checkpoint_size_bytes=weight.stat().st_size, strict_load="PASS", state_source=source,
                  gpu=before, torch_gpu_uuid=actual_uuid, torch_gpu_name=props.name,
                  synthetic=True, batch_size=1, tta=False, paths=[], benchmark=None)
    tensors = []
    def observe(label, tensor):
        if tensor.ndim != 3 or tensor.shape[-1] < 2 or tensor.shape[-2] < 1:
            raise ValueError(f"Unexpected native score shape: {tuple(tensor.shape)}")
        if tensor.device.type != "cuda" or tensor.device.index != 0:
            raise ValueError("Scores must remain on assigned GPU0")
        if not torch.isfinite(tensor).all():
            raise ValueError("Nonfinite native scores")
        tensors.append(dict(stage=label, shape=list(tensor.shape), dtype=str(tensor.dtype),
                            device=str(tensor.device), finite=True))
    with torch.no_grad():
        for frames, fusion in units(asset):
            item = dict(frames=frames, fusion=fusion)
            try:
                args_for_wrapper = dict(wrapper_args, in_images=frames)
                wrapped = datasets.make(dict(name=spec["name"], args=args_for_wrapper))
                batch = wrapped.collate_fn([wrapped[0]])
                inputs = batch["lr_seq"].to("cuda:0")
                tensors.clear()
                use_fp16 = bool(config.get("use_fp16", True))
                decoded = evaluator.infer_sequence(model, inputs, str(config["cls_loss"]).upper(), converter,
                                                   fusion, use_fp16, False, observe)
                torch.cuda.synchronize(0)
                item.update(status="PASS", input_shape=list(inputs.shape), input_dtype=str(inputs.dtype),
                            amp_fp16=use_fp16, scores=list(tensors), decoder=validate_decoded(*decoded, evaluator))
            except Exception as exc:
                item.update(status="FAIL", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
            result["paths"].append(item)
        if "name" in asset:
            try:
                spec_bench = importlib.util.spec_from_file_location("paper_benchmark", BENCHMARK)
                bench = importlib.util.module_from_spec(spec_bench)
                spec_bench.loader.exec_module(bench)
                # Exactly one F1 forward; no benchmark warmups, timing or profiler.
                inputs = inputs[:, 0].contiguous(memory_format=torch.channels_last)
                torch.backends.cudnn.allow_tf32 = False  # original efficiency worker setting
                scores = bench.forward_logits_once(model, inputs, config["cls_loss"] == "SVTRV2_AR",
                                                   torch.device("cuda:0"), True)
                torch.cuda.synchronize(0)
                tensors.clear()
                observe("benchmark_fp16", scores)
                result["benchmark"] = dict(status="PASS", forwards=1, amp_fp16=True, scores=list(tensors),
                                           input_shape=list(inputs.shape), input_dtype=str(inputs.dtype))
            except Exception as exc:
                result["benchmark"] = dict(status="FAIL", error=f"{type(exc).__name__}: {exc}",
                                           traceback=traceback.format_exc())
    result.update(peak_allocated_mib=torch.cuda.max_memory_allocated(0) / 2**20,
                  peak_reserved_mib=torch.cuda.max_memory_reserved(0) / 2**20)
    result["checkpoint_unchanged_after"] = str(require_hash(weight, asset["checkpoint_sha256"], asset["id"])) == str(weight)
    result["status"] = "PASS" if all(p["status"] == "PASS" for p in result["paths"]) and (
        result["benchmark"] is None or result["benchmark"]["status"] == "PASS") else "FAIL"
    write_new(args.result, result)
    del model, inputs
    torch.cuda.empty_cache()


def stop_owned(process):
    if process.poll() is None and os.getpgid(process.pid) == process.pid:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def campaign(args):
    check_sources()
    output = new_output(args.output)
    selected = selected_assets()
    runtime = runtime_check()
    before = ready(args.expected_gpu_uuid)
    baseline = {p["pid"] for p in before["processes"]}
    write_new(output / "preflight.json", dict(runtime=runtime, gpu=before,
              source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              source_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
              limitation="Synthetic compatibility only; no accuracy, cohort or timing reproduction"))
    records = []
    def interrupted(signum, frame):
        raise KeyboardInterrupt("Campaign interrupted; owned worker cleanup follows")
    signal.signal(signal.SIGTERM, interrupted)
    passed_by_identity = {}
    for asset in selected:
        key = asset["id"]
        identity = (asset["config_sha256"], asset["checkpoint_sha256"])
        try:
            # Check every mapping, including reused ablation evidence, independently.
            check_weights(args.weights, [asset])
            if identity in passed_by_identity:
                record = dict(id=key, status="PASS", reused_from=passed_by_identity[identity])
            else:
                snapshot = ready(args.expected_gpu_uuid)
                guard(snapshot, args.expected_gpu_uuid, baseline)
                path = output / (key + ".json")
                env = environment()
                env["CUDA_VISIBLE_DEVICES"] = "0"
                command = [sys.executable, "-m", "reproducibility.smoke", "--worker", key,
                           "--weights", str(Path(args.weights).resolve()), "--result", str(path),
                           "--expected-gpu-uuid", args.expected_gpu_uuid]
                with (output / (key + ".log")).open("x") as log:
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        deadline = time.monotonic() + args.timeout
                        while process.poll() is None:
                            if time.monotonic() > deadline:
                                raise TimeoutError("Bounded worker timeout")
                            time.sleep(2)
                            guard(gpu_snapshot(), args.expected_gpu_uuid, baseline | {process.pid}, active=True)
                        if process.returncode or not path.exists():
                            raise RuntimeError(f"Worker exited {process.returncode}; see private log")
                    finally:
                        stop_owned(process)
                record = load_json(path)
                if record["status"] == "PASS":
                    passed_by_identity[identity] = key
        except Exception as exc:
            record = dict(id=key, status="BLOCKED", error=f"{type(exc).__name__}: {exc}")
        records.append(record)
        print(key, record["status"], flush=True)
        write_new(output / (key + ".summary.json"), record)
    after = gpu_snapshot()
    write_new(output / "campaign.json", dict(records=records, gpu_before=before, gpu_after=after,
              baseline_processes_still_present=baseline <= {p["pid"] for p in after["processes"]},
              limitation="No experimental table metrics or original benchmark measurements recomputed"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output")
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--worker", choices=[a["id"] for a in selected_assets()], help=argparse.SUPPRESS)
    parser.add_argument("--result", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 10 <= args.timeout <= 300:
        parser.error("--timeout must be between 10 and 300 seconds")
    if args.worker:
        if not args.result:
            parser.error("Worker requires private result path")
        worker(args)
    else:
        if not args.output:
            parser.error("A NEW private --output outside source is required")
        campaign(args)


if __name__ == "__main__":
    main()
