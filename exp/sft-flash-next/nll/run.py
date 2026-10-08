"""Resumable 30-request NLL worker for an interactive Kaggle Jupyter session."""
from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

from common import COUNTS, code_hash, digest, file_hash, read_json, verify_manifest, write_json
from results import bind_run, load_result, save_result
from protocol import POLICY, budget, execute_stages


def validate_bundle(bundle):
    bundle = Path(bundle)
    m = read_json(bundle / "manifest.json")
    verify_manifest(m)
    if m.get("real_data") is not True or len(m["samples"]) != 30 or m["counts"] != list(COUNTS):
        raise ValueError("GPU run requires the verified real-data 30-request/five-model panel")
    if len(set(r["sample_id"] for r in m["samples"])) != 30:
        raise ValueError("Duplicate selected samples")
    for r in m["samples"]:
        p = (bundle / r["path"]).resolve()
        if not p.is_relative_to(bundle.resolve()) or file_hash(p) != r["sample_sha256"]:
            raise ValueError("Sample file/checksum mismatch")
    for name, checksum in m["processor_files"].items():
        if Path(name).name != name or file_hash(bundle / "processor" / name) != checksum:
            raise ValueError("Processor checksum mismatch")
    for count, checksum in m["maps"].items():
        value = read_json(bundle / f"maps/keep-{count}.json")
        if digest(value) != checksum or value.get("validation_exposed") is not False:
            raise ValueError("Map checksum or calibration provenance mismatch")
    return m


def model_identity(directory, model_id):
    """Bind to an immutable Kaggle model version plus config/index and sizes.

    Does not claim a byte hash of 100+ GB of weights. The read-only, versioned
    Kaggle input is the weight identity. Mutable local model roots are rejected.
    """
    directory = Path(directory).resolve()
    if not directory.is_relative_to(Path("/kaggle/input")) or not model_id.rstrip("/").endswith("/1"):
        raise ValueError("Use the pinned Kaggle model version 1 under /kaggle/input")
    index = read_json(directory / "model.safetensors.index.json")
    sizes = {}
    for name in sorted(set(index["weight_map"].values())):
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError("Missing/invalid model shard")
        sizes[name] = path.stat().st_size
    return dict(model_id=model_id, identity_kind="immutable-kaggle-version+index-config-shard-sizes",
                index_sha256=file_hash(directory / "model.safetensors.index.json"),
                config_sha256=file_hash(directory / "config.json"), shard_sizes=sizes)


def fits_deadline(deadline, now, tokens, rate, margin=120):
    return now + tokens / max(rate, 1) * 1.3 + margin < deadline


def mirror_fresh(out, identity, max_lag, now=None):
    path = Path(out) / "mirror-ack.json"
    ack = read_json(path) if path.exists() else {}
    age = (time.time() if now is None else now) - ack.get("last_verified_unix", 0)
    return ack.get("identity") == identity and 0 <= age < max_lag


def worker(args):
    bundle, out = Path(args.bundle), Path(args.out)
    manifest = validate_bundle(bundle)
    if args.preflight_only:
        from data import encode, load_processor

        processor = load_processor(bundle / "processor")
        if type(processor.image_processor).__name__ != manifest["image_processor_class"]:
            raise ValueError("Image processor backend differs from CPU preparation")
        for row in manifest["samples"]:
            _, annotation = encode(processor, read_json(bundle / row["path"]))
            if digest(annotation) != digest({key: row[key] for key in annotation}):
                raise ValueError(f"Reprocessed annotation mismatch: {row['sample_id']}")
            if digest(read_json(bundle / "annotations" / f"{row['sample_id']}.json")) != digest(annotation):
                raise ValueError(f"Stored annotation mismatch: {row['sample_id']}")
            print(f"validated {row['sample_id']}: {annotation['total_tokens']} tokens", flush=True)
        print(json.dumps({"ready_cpu_bundle": True, "requests": 30, "jobs": manifest["jobs"],
                          "processed_tokens": manifest["processed_tokens"],
                          "staged_budget": budget(manifest),
                          "estimated_minutes": manifest["estimated_minutes"]}, indent=2))
        return
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "worker.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("An evaluation worker already owns this output") from None
        return execute(args, manifest, bundle, out)


def execute(args, manifest, bundle, out):
    import torch
    from data import encode, load_processor

    sys.path.insert(0, str(Path(args.reap_dir).resolve()))
    import reap_model
    import replay

    if not torch.cuda.is_available():
        raise RuntimeError("GPU not available; no model was loaded")
    device = torch.cuda.get_device_properties(0)
    if device.total_memory < 90 * 2**30:
        raise RuntimeError("This full-resident sweep requires the qualified 96 GB GPU profile")
    if importlib.metadata.version("transformers") != "5.18.0":
        raise RuntimeError("Expected pinned transformers==5.18.0")
    start = time.monotonic()
    deadline = start + args.session_minutes * 60
    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    signal.signal(signal.SIGINT, lambda *_: stopping.set())
    state = {"phase": "loading", "completed": 0, "session_minutes": args.session_minutes,
             "required_jobs": 60, "maximum_jobs": manifest["jobs"]}
    heartbeat_stop = threading.Event()

    def heartbeat():
        while not heartbeat_stop.is_set():
            write_json(out / "heartbeat.json", dict(state, unix_time=time.time(),
                                                     elapsed_seconds=time.monotonic()-start))
            heartbeat_stop.wait(15)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        model_info = model_identity(args.model_dir, args.model_id)
        versions = {name: importlib.metadata.version(name) for name in
                    ("torch", "transformers", "tokenizers", "numpy", "safetensors")}
        for name in ("flash-linear-attention", "causal-conv1d"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        reap_model.OPTIONS.update(compile_dequant=False, attention="einsum")
        fast_linear = reap_model.check_fast_linear_attention(log=print)
        config = dict(manifest_sha256=manifest["manifest_sha256"], model=model_info,
                      code_sha256=code_hash(Path(__file__).parent, Path(args.reap_dir)),
                      versions=versions, gpu=device.name, capability=torch.cuda.get_device_capability(),
                      chunk=args.chunk, lm_block=args.lm_block, dtype="bfloat16", kernel_options=reap_model.OPTIONS,
                      fast_linear_attention=fast_linear, cuda=torch.version.cuda,
                      numerical_tolerance=args.noise_tolerance)
        config["protocol"] = POLICY
        # JSON-normalize tuples so reconnecting does not cause false mismatch.
        config = json.loads(json.dumps(config))
        identity = bind_run(out, config)
        write_json(out / "manifest.json", manifest)
        (out / "results").mkdir(exist_ok=True)
        completed = sum(load_result(out, identity, c, r) is not None for r in manifest["samples"]
                        for c in COUNTS)
        state["completed"] = completed
        from report import generate
        if generate(out)["complete"]:
            state["phase"] = "complete"
            print("Staged protocol already complete; no model load needed.", flush=True)
            return
        state["phase"] = "awaiting_external_collector"
        collector_deadline = min(deadline-120, time.monotonic()+90)
        while not mirror_fresh(out, identity, args.mirror_max_lag):
            if stopping.is_set() or time.monotonic() >= collector_deadline:
                raise InterruptedError("External collector not connected; stopped before loading weights")
            time.sleep(1)
        state["phase"] = "loading"
        model, _ = reap_model.load_model(args.model_dir, record=False)
        model.eval()
        if manifest["image_backend"] != "pil":
            raise ValueError("Expected the fixed PIL image-processing backend")
        processor = load_processor(bundle / "processor")
        if type(processor.image_processor).__name__ != manifest["image_processor_class"]:
            raise ValueError("Image processor backend differs from CPU preparation")
        text_config = model.config.text_config
        masks = {}
        for count in COUNTS:
            value = read_json(bundle / f"maps/keep-{count}.json")
            if len(value["kept"]) != text_config.num_hidden_layers or text_config.num_experts != 512:
                raise ValueError("Map/model architecture mismatch")
            mask = torch.zeros(text_config.num_hidden_layers, 512, dtype=torch.bool)
            for i in range(text_config.num_hidden_layers):
                keep = value["kept"][str(i)]
                if len(keep) != count or len(set(keep)) != count or not all(0 <= x < 512 for x in keep):
                    raise ValueError("Invalid retained expert list")
                mask[i, keep] = True
            masks[count] = None if count == 512 else mask
        rates = [500.0]  # conservative until a few complete jobs provide evidence
        encoded_id, encoded = None, None

        def evaluate(row, count, repeat=False, chunk=None):
            nonlocal encoded_id, encoded
            prior = load_result(out, identity, count, row)
            if prior is not None and not repeat:
                return prior[1]
            if not mirror_fresh(out, identity, args.mirror_max_lag):
                raise InterruptedError("External collector missing/stale; results are preserved, restart after reconnecting")
            if (stopping.is_set() or (out / "STOP").exists() or not fits_deadline(
                    deadline, time.monotonic(), row["total_tokens"], min(rates[-5:]))):
                raise InterruptedError("Graceful stop at request boundary")
            state.update(phase="scoring", sample_id=row["sample_id"], experts=count)
            if encoded_id != row["sample_id"]:
                encoded, metadata = encode(processor, read_json(bundle / row["path"]))
                if digest(metadata) != digest({key: row[key] for key in metadata}):
                    raise ValueError("Runtime encoding differs from CPU-frozen annotations")
                encoded_id = row["sample_id"]
            ids = encoded["input_ids"]
            target_mask = torch.zeros_like(ids, dtype=torch.bool)
            target_mask[:, row["prompt_tokens"]:] = True
            reap_model.set_pruning(model, masks[count])
            torch.cuda.reset_peak_memory_stats()
            t0 = time.monotonic()
            output = replay.replay(model, None, encoded, torch.zeros_like(ids),
                                   target_mask=target_mask, token_losses=True,
                                   chunk_tokens=chunk or args.chunk, lm_block=args.lm_block)
            torch.cuda.synchronize()
            elapsed = time.monotonic()-t0
            losses = output["token_nll"].numpy()
            positions = output["token_positions"].numpy()
            if not repeat:
                save_result(out, identity, count, row, losses, positions,
                            seconds=elapsed, processed_tokens=row["total_tokens"],
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            peak_reserved_bytes=torch.cuda.max_memory_reserved())
                state["completed"] += 1
                rates.append(row["total_tokens"]/elapsed)
                print(f"{state['completed']}/{state['required_jobs']} {row['sample_id']} {count}: "
                      f"NLL={losses.mean():.6f}, {elapsed:.1f}s", flush=True)
            return losses

        short = min(manifest["samples"], key=lambda r: r["total_tokens"])
        smoke_path = out / "smoke.json"
        if smoke_path.exists():
            if read_json(smoke_path)["identity"] != identity:
                raise ValueError("Smoke-check identity mismatch")
        else:
            baseline = evaluate(short, 512)
            repeated = evaluate(short, 512, repeat=True)
            reblocked = evaluate(short, 512, repeat=True, chunk=max(1, args.chunk//2))
            deltas = [float(np.abs(a.astype(np.float64)-baseline).mean()) for a in (repeated, reblocked)]
            if max(deltas) > args.noise_tolerance:
                raise RuntimeError(f"Numerical/chunk-parity check failed: mean absolute token deltas {deltas}")
            write_json(smoke_path, dict(identity=identity, sample_id=short["sample_id"],
                                       mean_abs_token_deltas=deltas, tolerance=args.noise_tolerance))
        evaluate(max(manifest["samples"], key=lambda r: r["total_tokens"]), 512)
        def checkpoint():
            selection = generate(out)
            state.update(protocol_gate=selection["gate"], required_jobs=selection["required_jobs"],
                         maximum_jobs=selection["maximum_jobs"])
            return selection

        execute_stages(manifest["samples"], evaluate, checkpoint)
        state["phase"] = "complete"
    except InterruptedError as exc:
        state["phase"] = "paused"
        print(str(exc), flush=True)
    except Exception as exc:
        state.update(phase="error", error_type=type(exc).__name__)
        write_json(out / "failure.json", dict(state))
        raise
    finally:
        heartbeat_stop.set()
        thread.join()
        write_json(out / "heartbeat.json", dict(state, unix_time=time.time(), elapsed_seconds=time.monotonic()-start))
        if (out / "run.json").exists():
            from report import generate
            generate(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-dir")
    parser.add_argument("--model-id", default="dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1")
    parser.add_argument("--reap-dir", default=str(Path(__file__).resolve().parents[2] / "reap-flash-next"))
    parser.add_argument("--chunk", type=int, default=8192)
    parser.add_argument("--lm-block", type=int, default=256)
    parser.add_argument("--session-minutes", type=float, default=120)
    parser.add_argument("--noise-tolerance", type=float, default=0.01)
    parser.add_argument("--mirror-max-lag", type=float, default=300,
                        help="Pause if external Jupyter collector has not verified a mirror within this many seconds")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not args.preflight_only and not args.model_dir:
        parser.error("--model-dir is required to score")
    if min(args.chunk, args.lm_block, args.session_minutes, args.mirror_max_lag) <= 0 or args.noise_tolerance < 0:
        parser.error("Block sizes and time budgets must be positive; noise tolerance cannot be negative")
    worker(args)


if __name__ == "__main__":
    main()
