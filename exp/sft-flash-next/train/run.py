"""Single-GPU full-context W4A16 + LoRA qualification, training and evaluation."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import importlib.metadata
import json
import math
from pathlib import Path
import random
import resource
import signal
import time

import torch

import backend
import checkpoints
import kernel_checks
from offload import ActivationOffload
from monitor import MemoryMonitor, host_memory
from dataset import Requests, digest, file_hash, fold_games, read_json, write_json

PROFILES = {
    "a100-80gb": dict(min_gib=79, capability=8, expert_block=512, query_block=8, index_block=64, loss_block=64),
    "rtx-pro-6000-96gb": dict(min_gib=94, capability=12, expert_block=8192, query_block=16, index_block=256, loss_block=128),
}


def check_model(root, bundle, keep_path=None):
    root = Path(root)
    config = read_json(root / "config.json")
    map_path = Path(keep_path) if keep_path else root / "keep.json"
    keep = read_json(map_path)
    if config.get("model_type") != "qwen4_exp" or config["text_config"]["num_experts"] not in (256, 512):
        raise ValueError("Expected Qwen3.8-Flash-Next/qwen4_exp with 256 or 512 source experts")
    if config["text_config"]["num_experts"] == 256 and keep_path and read_json(root / 'keep.json') != keep:
        raise ValueError('A pruned checkpoint must use its own expert map')
    quant = config.get("quantization_config", {})
    if quant.get("bits") != 4 or quant.get("sym") is not True or quant.get("packing_format") != "auto_round:auto_gptq":
        raise ValueError("Expected symmetric Intel AutoRound GPTQ W4A16 packing; this is not NF4")
    if keep.get("validation_exposed") is not False or not keep.get("calibration_provenance"):
        raise ValueError("Use the train-only keep-256.json from the completed NLL bundle, not the all-game model recipe")
    known, validation = fold_games(read_json(bundle.root / "folds.json"), bundle.manifest["validation_fold"])
    known_short = {g.split("-")[0] for g in known}
    held_short = {g.split("-")[0] for g in validation}
    calib = {r["game"] for r in keep["calibration_provenance"].values()}
    if not calib or not calib <= known_short - held_short:
        raise ValueError("Expert map includes held-out or unknown games")
    expected_layers = {str(i) for i in range(config["text_config"]["num_hidden_layers"])}
    if set(keep["kept"]) != expected_layers or any(len(v) != 256 or v != sorted(set(v)) or min(v) < 0 or max(v) >= 512 for v in keep["kept"].values()):
        raise ValueError("Invalid per-layer original expert IDs")
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                 "preprocessor_config.json", "processor_config.json", "special_tokens_map.json",
                 "vocab.json", "merges.txt", "tokenizer.model"):
        p = bundle.root / "processor" / name
        if p.exists() and (not (root / name).exists() or file_hash(root / name) != file_hash(p)):
            raise ValueError(f"Model/dataset processor asset differs: {name}")
    index = read_json(root / "model.safetensors.index.json")
    # Full content hashes bind resumed adapters to weights, not merely a path/header.
    hashes = {"expert_map": file_hash(map_path)}
    names = sorted({"config.json", "model.safetensors.index.json", *index["weight_map"].values()})
    for name in names:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Unsafe model shard path")
    def hash_asset(name):
        print(f"hashing model asset {name}", flush=True)
        return name, file_hash(root / name)
    # Four bounded streaming readers avoid serial NFS startup without retaining
    # whole files in Python memory; every byte still enters the SHA256 identity.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix='model-hash') as pool:
        hashes.update(pool.map(hash_asset, names))
    return hashes


def hardware(profile):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one CUDA GPU using CUDA_VISIBLE_DEVICES")
    p = torch.cuda.get_device_properties(0)
    rule = PROFILES[profile]
    if p.total_memory / 2**30 < rule["min_gib"] or p.major != rule["capability"]:
        raise ValueError(f"{p.name} ({p.total_memory/2**30:.1f} GiB, sm_{p.major}{p.minor}) does not match {profile}")
    if not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 training is required")
    return dict(name=p.name, total_memory=p.total_memory, capability=[p.major, p.minor])


def runtime():
    names = ("torch", "transformers", "numpy", "safetensors", "flash-linear-attention", "causal-conv1d", "triton", "numpy", "pillow", "torchvision")
    result = {name: importlib.metadata.version(name) for name in names}
    import causal_conv1d_cuda
    result.update(cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                  cxx11_abi=torch.compiled_with_cxx11_abi(),
                  causal_conv1d_binary_sha256=file_hash(causal_conv1d_cuda.__file__))
    return result


def epoch_order(rows, epochs, seed):
    out = []
    for epoch in range(epochs):
        current = list(range(len(rows)))
        random.Random(seed + epoch).shuffle(current)
        out.extend(current)
    return out


def learning_rate(step, total, peak):
    warmup = max(1, math.ceil(total * .05))
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup - 1)
    return peak * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))


def finish_step(model, optimizer, pending, lr):
    parameters = [p for p in model.parameters() if p.requires_grad]
    for p in parameters:
        if p.grad is None or not torch.isfinite(p.grad).all():
            raise ValueError("Missing or nonfinite adapter gradient")
        p.grad.div_(pending)
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(norm)


def backward_request(model, bundle, row, profile, offload):
    enc, ann = bundle.get(row)
    context = ActivationOffload(model, **profile.get("offload_options", {})) if offload else nullcontext()
    with backend.request_ple_cache(model), context:
        loss = backend.loss_sum(model, enc, ann["prompt_tokens"], "cuda", profile["loss_block"]) / ann["target_tokens"]
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        loss.backward()  # unnormalized sum of request means; divide once at optimizer boundary
    return float(loss.detach())


def qualify(model, optimizer, bundle, rows, profile, identity, out, accumulation, offload):
    # Cover three different activation extremes, plus a complete accumulation window.
    probes = {max(rows, key=lambda r: r[field])["sample_id"] for field in ("total_tokens", "images", "target_tokens")}
    selected = [r for r in rows if r["sample_id"] in probes]
    selected += [max(rows, key=lambda r: r["total_tokens"])] * max(0, accumulation - len(selected))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    losses = []
    for row in selected:
        loss = backward_request(model, bundle, row, profile, offload)
        losses.append(dict(sample_id=row["sample_id"], loss=loss))
        print(json.dumps(losses[-1]), flush=True)
    norm = finish_step(model, optimizer, len(selected), 1e-4)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_reserved()
    capacity = identity["hardware"]["total_memory"]
    headroom = identity['recipe'].get('gpu_headroom_gib', 4.)
    report = dict(identity=identity, passed=peak + headroom*2**30 < capacity, losses=losses, grad_norm=norm,
        required_headroom_gib=headroom, observed_headroom_gib=(capacity-peak)/2**30,
        peak_reserved_bytes=peak, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        seconds=time.monotonic()-started, adapter_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    write_json(out / "qualification.json", report)
    if not report["passed"]:
        raise RuntimeError("Qualification did not retain the requested GPU headroom; see qualification.json")
    print("Qualification passed; diagnostic updates must be discarded before training.", flush=True)


def train(model, optimizer, bundle, rows, args, identity, profile):
    out = Path(args.out)
    report = read_json(args.qualification)
    if report.get("identity") != identity or report.get("passed") is not True:
        raise ValueError("Qualification does not match this dataset/model/runtime/GPU/recipe")
    cursor, step, pending = 0, 0, 0
    if args.resume:
        cursor, step, pending = checkpoints.load(out, model, optimizer, identity)
    elif (out / "latest.json").exists():
        raise ValueError("Existing checkpoint: use --resume or a new output directory")
    order = epoch_order(rows, args.epochs, args.seed)
    total_steps = args.epochs * math.ceil(len(rows) / args.accumulation)
    stop = [False]
    def request_stop(*_):
        stop[0] = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    started, initial_step = time.monotonic(), step
    while cursor < len(order):
        row = rows[order[cursor]]
        torch.cuda.reset_peak_memory_stats()
        t = time.monotonic()
        loss = backward_request(model, bundle, row, profile, not args.no_offload)
        pending += 1
        cursor += 1
        norm = None
        if pending == args.accumulation or cursor % len(rows) == 0:
            norm = finish_step(model, optimizer, pending, learning_rate(step, total_steps, args.lr))
            step += 1
            pending = 0
        # Every full request persists partial gradients too. SIGKILL loses at most one request.
        checkpoints.save(out, model, optimizer, identity, cursor, step, pending)
        metrics = dict(cursor=cursor, step=step, pending=pending, sample_id=row["sample_id"], loss=loss,
            grad_norm=norm, seconds=time.monotonic()-t, total_tokens=row["total_tokens"],
            target_tokens=row["target_tokens"], images=row["images"],
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), **host_memory())
        with open(out / "metrics.jsonl", "a") as stream:
            stream.write(json.dumps(metrics) + "\n")
        print(json.dumps(metrics), flush=True)
        if stop[0] or (args.session_minutes and time.monotonic()-started >= args.session_minutes*60) or (
                args.max_updates and step-initial_step >= args.max_updates):
            break
    write_json(out / "status.json", dict(complete=cursor == len(order), cursor=cursor, step=step, pending=pending))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("preflight", "qualify", "train", "fit"), required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--keep", help="Train-only keep-256 map; required with a 512-expert source")
    p.add_argument("--data", required=True)
    p.add_argument("--profile", choices=PROFILES, default="a100-80gb")
    p.add_argument("--out", required=True, help="Durable filesystem/mounted volume for checkpoints")
    p.add_argument("--qualification", help="Matching qualification.json required for training")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=32.)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--accumulation", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261008)
    p.add_argument("--no-offload", action="store_true")
    p.add_argument("--gpu-headroom-gib", type=float, default=4., help="Required free space above peak reserved VRAM during actual-data qualification")
    p.add_argument("--attention-backend", choices=("sdpa", "triton"), default="triton")
    p.add_argument("--gdn-checkpoint", action="store_true")
    p.add_argument("--gated-norm-block", type=int, default=0)
    p.add_argument("--gdn-chunk-tokens", type=int, default=0)
    p.add_argument("--gdn-block-tokens", type=int, default=0)
    p.add_argument("--attention-projection-block", type=int, default=0)
    p.add_argument("--rms-block-mib", type=float, default=0)
    p.add_argument("--ple-resident", action="store_true", help="Keep the entire frozen PLE table in host RAM; requires separate capacity qualification")
    p.add_argument("--no-ple-checkpoint", action="store_true")
    p.add_argument("--ple-block", type=int, default=0, help="Frozen PLE token windows with exact convolution halo; 0 disables")
    p.add_argument("--checkpoint-group", type=int, default=1)
    p.add_argument("--expert-block", type=int, help="Override profile expert microbatch")
    p.add_argument("--hyper-block", type=int, default=1024)
    p.add_argument("--ple-cache-gib", type=float, default=1.)
    p.add_argument("--ple-workers", type=int, default=8)
    p.add_argument("--disk-dir", help="Bounded activation scratch; not a checkpoint location")
    p.add_argument("--disk-budget-gib", type=float, default=0.)
    p.add_argument("--prefetch", type=int, default=2)
    p.add_argument("--host-memory-limit-gib", "--host-anon-limit-gib", dest="host_anon_limit_gib", type=float, default=160., help="Guard anonymous + shared/pinned + unreclaimable kernel RAM")
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--session-minutes", type=float, default=110.)
    p.add_argument("--max-updates", type=int, default=0, help="Stop budget; does not change the full-run LR schedule")
    args = p.parse_args()
    if args.attention_projection_block < 0 or args.gdn_block_tokens < 0 or args.rms_block_mib < 0 or args.gdn_chunk_tokens < 0 or args.gpu_headroom_gib <= 0 or args.gated_norm_block < 0 or args.ple_block < 0 or args.disk_budget_gib < 0 or args.prefetch < 0 or args.hyper_block < 0 or args.cpu_threads < 1 or args.ple_cache_gib < 0 or args.ple_workers < 1 or args.checkpoint_group < 1 or args.host_anon_limit_gib <= 0 or (args.expert_block is not None and args.expert_block < 1):
        p.error("Invalid memory/thread option")
    if args.disk_budget_gib and (not args.disk_dir or args.no_offload):
        p.error("Disk offload requires --disk-dir and enabled offload")
    torch.set_num_threads(args.cpu_threads)
    if min(args.rank, args.alpha, args.lr, args.epochs, args.accumulation) <= 0 or args.max_updates < 0 or args.session_minutes < 0:
        p.error("Invalid nonpositive training parameter or negative stop budget")
    if args.mode == "train" and not args.qualification:
        p.error("train requires --qualification")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.mode == "qualify" and (out / "qualification.json").exists():
        raise ValueError("Qualification already exists; use a fresh output directory")
    bundle = Requests(args.data)
    rows = bundle.split("train")
    if not rows:
        raise ValueError("No training games: panel30 is validation-only. Prepare generated-thinking data from training games.")
    gpu = hardware(args.profile) if args.mode != "preflight" else None
    versions = runtime() if args.mode != "preflight" else None
    model_hashes = check_model(args.model, bundle, args.keep)
    write_json(out / "preflight.json", dict(model_hashes=model_hashes, data=bundle.manifest["sha256"],
        training_requests=len(rows), validation_requests=len(bundle.split("validation")),
        max_tokens=max(r["total_tokens"] for r in rows), max_images=max(r["images"] for r in rows)))
    if args.mode == "preflight":
        return
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    write_json(out / "kernel-checks.json", kernel_checks.check(attention_backend=args.attention_backend))
    torch.manual_seed(args.seed)  # kernel probe uses its own fixed seed
    if args.gdn_chunk_tokens:
        import gdn_segments
        write_json(out / 'gdn-segment-checks.json', gdn_segments.check())
    if args.gdn_block_tokens:
        import gdn_blocks
        write_json(out / 'gdn-block-checks.json', gdn_blocks.check())
    if args.attention_projection_block:
        import attention_projection_checks
        write_json(out / 'attention-projection-checks.json', attention_projection_checks.check())
    model, _ = backend.load_pruned(args.model, args.keep)
    profile = dict(PROFILES[args.profile])
    if args.expert_block:
        profile["expert_block"] = args.expert_block
    profile["offload_options"] = dict(disk_dir=args.disk_dir, disk_budget_gib=args.disk_budget_gib, prefetch=args.prefetch)
    targets = backend.configure(model, rank=args.rank, alpha=args.alpha,
        expert_block=profile["expert_block"], query_block=profile["query_block"], index_block=profile["index_block"], attention_backend=args.attention_backend, hyper_block=args.hyper_block,
        ple_cache_gib=args.ple_cache_gib, ple_workers=args.ple_workers, checkpoint_group=args.checkpoint_group, ple_checkpoint=not args.no_ple_checkpoint, gdn_checkpoint=args.gdn_checkpoint, ple_block=args.ple_block, gated_norm_block=args.gated_norm_block, ple_resident=args.ple_resident, gdn_chunk_tokens=args.gdn_chunk_tokens, rms_block_mib=args.rms_block_mib, gdn_block_tokens=args.gdn_block_tokens, attention_projection_block=args.attention_projection_block)
    write_json(out / "model-gradient-check.json", kernel_checks.check_model_backward(model, profile["loss_block"]))
    code_root = Path(__file__).resolve().parents[1]
    code = {str(f.relative_to(code_root)): file_hash(f) for folder in ("train", "nll")
            for f in sorted((code_root / folder).glob("*.py"))}
    code["reap_model.py"] = file_hash(Path(backend.rm.__file__))
    identity = dict(version=1, model=digest(model_hashes), data=bundle.manifest["sha256"],
        runtime=versions, hardware=gpu, code=code, targets=targets,
        recipe=dict(profile=args.profile, rank=args.rank, alpha=args.alpha, lr=args.lr, epochs=args.epochs,
                    accumulation=args.accumulation, seed=args.seed, offload=not args.no_offload, gpu_headroom_gib=args.gpu_headroom_gib,
                    attention_backend=args.attention_backend, hyper_block=args.hyper_block,
                    disk_budget_gib=args.disk_budget_gib, prefetch=args.prefetch,
                    ple_cache_gib=args.ple_cache_gib, ple_workers=args.ple_workers, checkpoint_group=args.checkpoint_group, expert_block=profile["expert_block"], ple_checkpoint=not args.no_ple_checkpoint, gdn_checkpoint=args.gdn_checkpoint, ple_block=args.ple_block, gated_norm_block=args.gated_norm_block, ple_resident=args.ple_resident, gdn_chunk_tokens=args.gdn_chunk_tokens, rms_block_mib=args.rms_block_mib, gdn_block_tokens=args.gdn_block_tokens, attention_projection_block=args.attention_projection_block))
    if (out / "run-identity.json").exists() and read_json(out / "run-identity.json") != identity:
        raise ValueError("Output belongs to a different run identity")
    write_json(out / "run-identity.json", identity)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0., foreach=False)
    with MemoryMonitor(out, args.host_anon_limit_gib):
        if args.mode == "qualify":
            qualify(model, optimizer, bundle, rows, profile, identity, out, args.accumulation, not args.no_offload)
        elif args.mode == 'fit':
            from evaluate import evaluate_model
            validation = bundle.split('validation')
            if not validation:
                raise ValueError('fit requires the fold-0 validation rows in the prepared bundle')
            qualification_path = out / 'qualification.json'
            if not qualification_path.exists():
                initial = backend.adapter_state(model)
                cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
                qualify(model, optimizer, bundle, rows, profile, identity, out, args.accumulation, not args.no_offload)
                backend.load_adapter(model, initial)
                optimizer.state.clear()
                optimizer.zero_grad(set_to_none=True)
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state_all(cuda_rng)
                del initial
            args.qualification = str(qualification_path)
            if read_json(qualification_path).get('identity') != identity or read_json(qualification_path).get('passed') is not True:
                raise ValueError('Existing qualification belongs to a different run')
            metadata = dict(model_sha256=identity['model'], hardware=gpu, run_identity_sha256=digest(identity))
            evaluate_model(model, bundle, validation, profile, out / 'validation-base', metadata)
            model.train()
            model.model.visual.eval()
            train(model, optimizer, bundle, rows, args, identity, profile)
            status = read_json(out / 'status.json')
            evaluate_model(model, bundle, validation, profile, out / f"validation-step-{status['step']:08d}-cursor-{status['cursor']:08d}",
                           dict(**metadata, checkpoint_sha256=read_json(out / 'latest.json')['sha256']))
        else:
            train(model, optimizer, bundle, rows, args, identity, profile)



if __name__ == "__main__":
    main()
