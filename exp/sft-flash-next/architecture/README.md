# Flash-Next architecture and training

This is the active architecture implementation. `../train/` preserves earlier
experiments, their immutable sources, DVC evidence and the existing trajectory
preparation/resume pipeline. New architecture work belongs here.

## Layout

```text
architecture/
  train.py                 whole-sample supervised training
  test.py                  one loss/backward capture, zero optimizer updates
  config.py                validated architecture/optimization settings
  model.py                 native loading, PEFT initialization, component assembly
  components/
    attention.py           native QSA selection with direct dense bias
    head.py                native, target-only, CCE exact, Liger FLCE objectives
    normalization.py       optional Liger plain/grouped/gated RMSNorm
    mlp.py                 optional shared/routed Liger SwiGLU
    ple.py                 frozen disk lookup and bounded CPU preparation
    precision.py           BF16 activation ports; native statistics preserved
    expert_offload.py      CPU expert masters and bounded layer prefetch
    adapters.py            per-config PEFT support for packed expert projections
    common.py              component adoption with unchanged parameter names
  runtime/
    loop.py                shared data/forward/backward path
    resources.py           per-phase GPU/RAM/time measurements
    evidence.py            source snapshots, hashes, raw-gradient comparison
    worker.py              detached Jupyter job supervision
  configs/                 explicit run configurations
  tests/                   configuration, state/dispatch and arithmetic checks
  reports/                 small reviewed results in Git
  results/                 raw attempts and DVC pointers
  jupyter.mjs              Kaggle HTTP/WebSocket dispatch and collection
```

## Architecture selection

`reference` is the user-defined FP32 all-expert LoRA reference with BF16
activation ports, CPU expert prefetch and DataLoader-prepared disk PLE. See
[REFERENCE.md](REFERENCE.md) for its exact contract and test initialization.
`native` preserves the earlier untouched HF/AutoRound/PEFT path and rejects
optimization flags. `optimized` starts from the same loader and selects
independent components. With all flags disabled, no component is
replaced. Each enabled component retains existing parameter objects and state
keys; the builder refuses unexpected instance forwards/offload hooks.

There are no mutations of PEFT factories, Torch loading/saved-tensor functions,
native classes, or global expert registries. There is no runtime AST rewriting
or generated model code. Native selection/expert arithmetic copied into explicit
components is source-pinned and reviewed separately.

| Flag | Choices/effect |
| --- | --- |
| `head` | `native`, `target`, `cce_exact`, `liger_flce` |
| `direct_attention_bias` | Opt3 mask storage; still dense native SDPA |
| `disk_ple` | Opt4 full frozen disk table and prepared CPU samples |
| `bf16_lora` | BF16 adapter parameter storage |
| `bf16_activations` | Opt5 declared model activation ports; statistics stay native |
| `liger_rmsnorm` | Opt6 normalization kernels, independently selectable |
| `liger_swiglu` | Opt6 shared/routed activation kernels, independently selectable |
| `offload_routed_experts` | CPU canonical expert state; at most two GPU layers |
| `lora_routed_experts` | LoRA on every routed expert gate/up/down projection |

Flags are independent unless a component has an explicit restriction. Opt6
remains unqualified: its earlier full-model gradient drift is unresolved.
Migration of a flag does not qualify its numerical behavior. The reference preset fixes its four required settings; optional Liger/CCE/direct
bias changes remain disabled there. Optimized configurations declare their flags.

## Commands

On the pinned GPU runtime:

```bash
python test.py --config configs/reference.json
python test.py --config configs/optimized.json
python test.py --config /path/to/run.json --architecture optimized --head cce_exact --bf16-lora --bf16-activations
python train.py --config /path/to/complete-labeled-samples.json
```

Both entry points accept the same architecture/optimization overrides, including
`--no-<flag>`. `--validate-only` checks configuration without importing CUDA.
`test.py` performs exactly one anchor forward/backward and zero optimizer
updates. Only `architecture="reference"` retains raw gradients. Optimized/native
candidates compare all raw named gradients in memory against the configured
reference, save `comparison.json` (bitwise matches, relative L2, cosine and
per-tensor differences), then discard the candidate tensors. Training does not
archive per-update gradients. Use `configs/optimized.json` as the next experiment
base: it pins the current all-expert reference loss and gradient manifest.
Changes to its optimization flags do not change this retention policy. `comparison_mode="exact"` enforces bitwise
loss and every gradient for refactor isolation; `comparison_mode="report"`
records numerical differences without inventing an acceptance tolerance.

Training requires explicit labels in each complete encoded PT sample. Ignored
user/tool/image tokens use -100; every supervised span remains interleaved with
its original context. No truncation or label reconstruction occurs in training.
The diagnostic `prompt_tokens` fallback exists only in test mode. Diagnostic
adapters cannot initialize `train.py`.

Training accumulates whole samples weighted by their target counts, normalizes
once by actual targets before clipping, and writes adapter/optimizer/RNG/cursor
checkpoints. The current entry point starts fresh; durable resume and production
data/fold qualification remain in the historical trajectory pipeline until a
separate migration is verified. No production training is claimed here.

## Kaggle Jupyter execution and evidence

From this folder on the desktop:

```bash
node jupyter.mjs --config configs/optimized.json --mode test --timeout-seconds 5400
node jupyter.mjs --action status --attempt <attempt-id>
node jupyter.mjs --action collect --attempt <attempt-id>
```

The bearer URL stays in `/tmp/kaggle_probe_url`. Each attempt uploads an isolated
source snapshot with SHA256s, dispatch Git commit and verified pinned dependency
archives from `configs/dependencies.json`, then executes in a detached
process through Jupyter. Mutable status reads bypass HTTP caching. It refuses a
busy GPU, defaults to a1200second timeout (`--timeout-seconds` overrides it), and preserves failures. Collection excludes
only derived model-view symlinks, never copying the full model accidentally.

Each capture records loss/gradient statistics and comparison, initialization, input/source/package
identities, phase timings, exact CUDA allocated/reserved peaks and sampled parent
RSS/treePSS/childRSS/host-used RAM. Raw attempts are automatically cached locally with DVC after collection;
small comparison reports are kept in Git. No Git or DVC pushes are authorized.

## Refactor qualification

The full16K legacy isolation replay passed with a clean exit: loss
0.6256952285766602 and all744 raw gradients are bitwise identical to native v0.
See `reports/refactor-iso.json`. The new all-expert reference completed separately with loss0.6244627833366394;
see `reports/reference.md`. This legacy proof does not qualify its numerical changes.
Small component checks do not replace that full-model isolation gate. Earlier
v8/Opt6 results and their identities are preserved under `../train/`.

Historical full-sample and large hidden-state gradient archives were pruned at
user request. Their compact comparisons, execution sources and original hashes
remain; per-output retention records explain intentionally missing files.
Frozen legacy pipeline dependency hashes are historical pins, so those retired
stages should not be rerun. The current reference and `configs/optimized.json`
are the active comparison path. See `reports/gradient-retention*.json`.

## Active optimization reruns (2026-10-10)

Only Opt1–3 remain in the experiment plan: target-only logits, CCE exact with
both gradient filters disabled, then CCE exact plus direct dense attention bias.
Opt4–6 are dropped as separate experiments. Disk PLE, BF16 activation ports,
FP32 adapter masters/gradients and CPU expert prefetch remain the fixed reference
contract. Liger RMSNorm/SwiGLU and BF16 LoRA masters are disabled in all three.
Each config pins the current reference's gradient and initial-state manifests;
every initial tensor must match before forward/backward. Candidate initial
states are compared rather than duplicated on disk. Reference shards are
verified and streamed one at a time. Raw candidate gradients are never saved.

The consolidated result table is [v0.md](v0.md): every recorded phase's timing,
GPU allocation/reservation, RAM RSS/PSS/child/host counters, total wall time,
loss change and gradient L2/cosine/bitwise matches appear in this one table.
Opt1 matches all 74,472 reference gradients bitwise. Opt2 and Opt3 complete with
finite gradients but have 1.751894% and 1.784223% relative L2 differences against
the reference. Direct-bias operator fixtures pass; the full Opt3 result includes
CCE and does not separately prove full-model mask equivalence.

## Chunked activation replay (Opt7–10)

See [CHUNKING.md](CHUNKING.md) for component boundaries, the <1% gradient gate,
paired in-memory control and cumulative configurations. Use `configs/opt7.json`
through `configs/opt10.json`; `--chunk-tokens` controls window size.
