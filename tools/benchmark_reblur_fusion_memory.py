"""Run the eight archived REBlur configurations in separate three-rank processes."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs/benchmark_reblur_fusion_memory_3gpu.yml"
IDS = {"gate4ca", "mean_v", "temporal", "local_st", "dual_expert", "bi_gru", "dynamic_filter", "direct2ca"}
CSV_FIELDS = ["id", "status", "original_training_gpus", "benchmark_gpus", "devices", "per_gpu_batch", "precision",
              "parameters", "train_allocated_gib", "train_reserved_gib", "validation_allocated_gib",
              "validation_reserved_gib", "step_seconds_max", "source_config", "config_sha256", "log", "error"]


def validate_config(entry, config):
    model, train = config["model"], config["train"]
    data, options = config["datasets"], model.get("fusion_options", {})
    if model.get("fusion_mode") != entry["fusion_mode"]:
        raise ValueError("Archived fusion mode does not match manifest")
    if (model.get("event_in", 6), model.get("base_dim", 32), model.get("tdc_kernel_size", 5),
            model.get("event3d_conv_type", "tdc")) != (6, 32, 5, "tdc"):
        raise ValueError("Expected 6 bins, base_dim=32 and TDC5")
    if model.get("two_stage", False) or model.get("cross_attn_type", "channel") != "channel":
        raise ValueError("Expected single-stage channel CA")
    base_sa = model.get("encoder_self_attn", "restormer_channel")
    if any((model.get("encoder_self_attn_" + branch) or base_sa) != "restormer_channel"
           for branch in ("rgb", "event2d", "event3d")):
        raise ValueError("Expected channel SA in all three branches")
    for split in ("train", "val"):
        if data[split].get("dataset_type", "h5") != "h5" or not data[split].get("norm_event"):
            raise ValueError("Expected original normalized REBlur H5 inputs")
    if data["train"].get("batch_size") != 1 or data["train"].get("patch_size") != 256 or data["val"].get("patch_size") is not None:
        raise ValueError("Expected batch 1, train crop 256 and full-frame validation")
    if (train.get("num_epochs"), train.get("learning_rate"), train.get("seed"), train.get("deterministic"),
            train.get("val_interval")) != (200, 0.0002, 42, True, 5):
        raise ValueError("Archived training protocol differs from the agreed REBlur study")
    if "readout" in entry and options.get("e3_readout", "temporal") != entry["readout"]:
        raise ValueError("Wrong temporal readout")
    if entry["id"] in ("mean_v", "local_st", "dual_expert"):
        if options.get("local_st_scales") != [0, 1, 2] or options.get("local_st_heads") != 1:
            raise ValueError("Refusing a coarse-only or wrong-head local configuration")
    if entry["id"] == "bi_gru" and options.get("gru_scales") != [0, 1, 2]:
        raise ValueError("Refusing a coarse-only GRU configuration")


def build_plan(manifest_path, experiments_root):
    manifest = yaml.safe_load(Path(manifest_path).read_text(encoding="utf-8"))
    entries = manifest["experiments"]
    if manifest.get("schema") != "reblur_fusion_memory.v1" or len(entries) != 8 or {e["id"] for e in entries} != IDS:
        raise ValueError("Manifest must contain exactly the eight distinct agreed experiments")
    plan, protocol = [], None
    for entry in entries:
        directory = Path(entry["directory"])
        if directory.is_absolute() or len(directory.parts) != 1 or ".." in directory.parts:
            raise ValueError("Experiment directory must be one folder name")
        folder = Path(experiments_root) / directory
        source = folder / "train_config.yml"
        raw = source.read_bytes()
        config = yaml.safe_load(raw.decode("utf-8"))
        validate_config(entry, config)
        log = (folder / "train_log.txt").read_text(encoding="utf-8", errors="replace")
        worlds = {(int(w), int(b)) for w, b in re.findall(r"DDP world_size: (\d+) \| per_gpu_batch: (\d+)", log)}
        if worlds != {(entry["original_world_size"], 1)}:
            raise ValueError(f"Unexpected original GPU/batch history for {entry['id']}: {worlds}")
        current = {key: config[key] for key in ("datasets", "loss", "train")}
        if protocol is not None and current != protocol:
            raise ValueError(f"Archived data/loss/training settings differ for {entry['id']}; no automatic overrides applied")
        protocol = current
        plan.append({**entry, "source_config": str(source.resolve()), "config_sha256": hashlib.sha256(raw).hexdigest(),
                     "raw_config": raw, "model_config": config["model"]})
    return plan


def worker_command(config, result, warmup, measured, samples):
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=3",
            str(ROOT / "tools/check_attention_memory.py"), "--config", str(config),
            "--warmup-steps", str(warmup), "--train-steps", str(measured), "--val-samples", str(samples),
            "--result-json", str(result)]


def stop_worker(process):
    if process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()


def run_worker(command, environment, log_path, timeout):
    with Path(log_path).open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=os.name == "posix")
        try:
            return process.wait(timeout=timeout)
        finally:
            stop_worker(process)


def checked_metrics(result_path, returncode, entry, warmup, measured, samples):
    if returncode != 0:
        raise RuntimeError(f"Worker exited with code {returncode}; inspect its log")
    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
    if (result.get("schema"), result.get("status"), result.get("world_size"), result.get("per_gpu_batch"),
            result.get("precision")) != ("attention_memory.v1", "pass", 3, 1, "float32"):
        raise ValueError("Worker did not report a successful three-rank FP32 measurement")
    if result.get("config_sha256") != entry["config_sha256"] or result.get("model_config") != entry["model_config"]:
        raise ValueError("Worker configuration does not match the unmodified archive")
    if (result.get("warmup_steps"), result.get("measured_steps")) != (warmup, measured):
        raise ValueError("Wrong measurement budget")
    if result.get("train_samples") != 486 or result["validation"].get("dataset_samples") != 903:
        raise ValueError("Expected the original 486/903 REBlur frame counts")
    if result["train"]["batch_shapes"]["event"] != [1, 6, 256, 256]:
        raise ValueError("Unexpected training tensor shape")
    val = result["validation"]
    if val.get("patch_size") is not None or val.get("mode") != "Complete-image validation" or val.get("samples_checked") != samples:
        raise ValueError("Full-frame validation preflight was not completed")
    row = {"parameters": result["parameters"], "step_seconds_max": result["train"]["step_seconds_max"]}
    for phase, prefix in (("train", "train"), ("validation", "validation")):
        for field in ("allocated", "reserved"):
            value = result[phase][field + "_gib"]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid memory measurement")
            row[prefix + "_" + field + "_gib"] = value
    if not math.isfinite(row["step_seconds_max"]) or row["step_seconds_max"] <= 0:
        raise ValueError("Invalid measured step duration")
    return row


def save_summary(directory, rows):
    payload = {"schema": "reblur_fusion_memory_summary.v1", "benchmark_gpus": 3,
               "scope": "steady-state preflight peaks, not a 200-epoch maximum or a new training score", "results": rows}
    target = Path(directory) / "summary.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    with (Path(directory) / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--experiments-root", type=Path, default=Path("/data1/jzy/experiments"))
    parser.add_argument("--devices", default="0,2,3")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--train-steps", type=int, default=3)
    parser.add_argument("--val-samples", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true", help="Validate the eight archives without CUDA or writing output files.")
    args = parser.parse_args()
    devices = args.devices.split(",")
    if len(devices) != 3 or len(set(devices)) != 3 or any(not d.isdecimal() for d in devices):
        parser.error("Supply exactly three distinct physical GPU indices")
    if args.warmup_steps < 3 or args.train_steps < 2 or not 1 <= args.val_samples <= 903 or args.timeout_seconds < 1:
        parser.error("Invalid preflight budget")
    if any(int(os.environ.get(key, "0")) > 0 for key in ("WORLD_SIZE", "LOCAL_RANK", "RANK")):
        parser.error("Run this launcher with python, not torchrun or inside another distributed job")
    plan = build_plan(args.manifest, args.experiments_root)
    for entry in plan:
        print(f"{entry['id']}: original training={entry['original_world_size']} GPUs, benchmark=3 GPUs; {entry['source_config']}", flush=True)
    if args.dry_run:
        print("DRY RUN PASS: eight archived configurations verified; no CUDA process or output file created.")
        return 0
    output = args.output_dir or ROOT / "outputs" / ("reblur_memory_3gpu_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "configs").mkdir()
    (output / "logs").mkdir()
    (output / "metrics").mkdir()
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.devices
    environment.setdefault("OMP_NUM_THREADS", "1")
    rows = []
    for entry in plan:
        key = entry["id"]
        snapshot, log, result = output / "configs" / f"{key}.yml", output / "logs" / f"{key}.log", output / "metrics" / f"{key}.json"
        snapshot.write_bytes(entry["raw_config"])
        row = {"id": key, "status": "failed", "source_experiment": entry["directory"],
               "original_training_gpus": entry["original_world_size"], "benchmark_gpus": 3, "per_gpu_batch": 1,
               "devices": args.devices,
               "precision": "float32", "source_config": entry["source_config"], "config_sha256": entry["config_sha256"], "log": str(log)}
        print(f"Measuring {key}; log={log}", flush=True)
        try:
            code = run_worker(worker_command(snapshot, result, args.warmup_steps, args.train_steps, args.val_samples),
                              environment, log, args.timeout_seconds)
            row.update(checked_metrics(result, code, entry, args.warmup_steps, args.train_steps, args.val_samples), status="pass")
        except (Exception, KeyboardInterrupt) as error:
            row["error"] = f"{type(error).__name__}: {error}"
            rows.append(row)
            save_summary(output, rows)
            print(f"STOP: {row['error']}\nPartial results: {output}", file=sys.stderr, flush=True)
            return 1
        rows.append(row)
        save_summary(output, rows)
        print(f"PASS {key}: train allocated={row['train_allocated_gib']:.3f} GiB, reserved={row['train_reserved_gib']:.3f} GiB", flush=True)
    print(f"All eight complete. CSV: {output / 'summary.csv'}\nJSON: {output / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
