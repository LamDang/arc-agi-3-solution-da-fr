# Latest Opt4 capture — 2026-10-09

* CCE exact and Opt3 retained; primary directly executes, no subagents, Kaggle
  Jupyter API only. Opt4 code `df0947c`, attempt `20261009173647-1a7e7242`,
  config/stage `cce-opt4-ple-v5` / `cce_opt4_ple_v5`.
* `ple_preparation.py`: original native CPU hash forward, exact checkpoint hash
  buffers/EOS history, deduplicated read-only disk lookup from all 128 shards,
  one spawned DataLoader worker and two outstanding preparations. Source Dataset
  can tokenize full sequences; capture reuses three identical pre-encoded anchors
  to exercise the queue. Only the first is trained for one forward/backward.
* `opt4_ple_capture.py` bypasses the 95.367889 GiB table at native model loading
  via an empty marker/filtered checkpoint-index view. All non-table weights and
  native PLE projections/gating/convolution remain native. Full table stays on disk.
* All 16,249-token CPU hash IDs equal native CUDA IDs; every prepared row equals
  safetensors reference bytes. 259,984 IDs deduplicate to 69,280 rows. Payload
  83,194,880 bytes, SHA256
  `ac0f7b432b156a88d5168e9fb52484e690dbf01b204036c40473cb6fcda8cbae`.
  Real-weight PLE outputs/input gradients and EOS/unfamiliar-token cases match.
* Two queued copies finish before forward; initial preparation 4.389 s, first
  wait 5.232 s, warm duplicate lookahead 0.343/0.354 s. These are not measured
  new-trajectory throughput. Same CPU storage is used forward/recomputation;
  GPU path does zero disk reads. Worker shuts down after capture.
* GPU allocated peaks exactly unchanged: 49.529/58.277 GiB. Parent RSS peaks
  33.878/34.006 GiB (down 96.447/95.161 GiB); worker adds 0.952 GiB RSS.
  New process-tree PSS 34.393/34.518 GiB; cgroup anonymous/file/current counters
  also logged. Reclaimable OS file cache remains; do not confuse it with anon.
* Forward/backward 116.754/217.409 s, combined +0.403% vs v4. Total 733.234 s
  includes a new 141.775 s exhaustive PLE check before native loading.
* Loss 0.6256126165390015 (`3f202826`), one ULP below v4. All 744 finite/nonzero,
  0 exact vs v4; global L2 1.138583%, worst tensor 2.276983%, cosine
  0.999935189389238. Gradient SHA256
  `235d769690d07fe434ceb870ec8f5008e7be4b6e61e14c6f0f4dcc7bf483dc8b`.
* PLE checks pass, but full-model bitwise gate fails explicitly. Unchanged CCE
  probe also varies independently (0.014461% / 0.011021% mean/sum L2 vs v4).
  Do not attribute the whole full-model difference to CCE without more evidence.
  No tolerance, clipping or optimizer update, and no production promotion.
* At 130K: current + two BF16 payloads = 1.859665 GiB, replacing the table's
  95.367889 GiB before hash/lookup/worker/image/offload/cache overhead. Component
  reduction ~93.508 GiB; no complete 130K memory/throughput claim. Opt3's dense
  FP32 GPU bias remains. Actual-trajectory training requires its independent gate.
* All 86 files restored identically from DVC; tree
  `ebb3229748d601a248020122433ede0e.dir`. Cached reproduction does not rerun the GPU
  or change the failed numerical gate.
* `v5.md`, master `v0.md`, metrics/review retain all results. No successor GPU job.

# Latest Opt3 capture — 2026-10-09

* User explicitly selected **CCE exact** for subsequent comparisons, superseding
  the earlier default-v1 wording below. Primary executes directly, no subagents;
  Kaggle Jupyter API only. No production training is authorized by this capture.
* Opt3 code `50b83fc`, attempt `20261009171104-b3b5e641`, config/stage
  `cce-opt3-mask-v4` / `cce_opt3_mask_v4`. Pinned native source selection prefix
  unchanged; lazy causal rows and direct dense bias, dummy -1 column/alignment.
* Eight real native-indexer/SDPA cases (BF16/FP32, lengths 3/4/9/33) pass bitwise
  output/QKV-gradient and mask/selection/causal checks, with raw fixtures saved.
* Actual indexer hidden/bias dtype is **FP32**, 24 calls. At 130K the final logical
  bias is 62.957 GiB (62.961 GiB padded), not the conditional BF16 31.479 GiB.
* Loss 0.6256126761436462 (`3f202827`), three ULPs below CCE v3 (`3f20282a`). All
  744 gradients finite/nonzero, 0/744 exact vs CCE v3; global relative L2 1.154146%,
  worst tensor 2.232123%, cosine 0.9999333961936885. Raw SHA256
  `ab1decf082a64b8ee9e45cce81f18ca4500c3faf68faa9dafe3f2eb70c40d66f`.
* Strict full-model equality gate failed after preserving evidence. Retain this
  failed diagnostic through DVC; no tolerance was accepted or gate weakened.
* Pre-Opt3 unchanged CCE probe uses identical fixture/native gradients, but its
  CCE gradients vary from v3 by 0.014741%/0.013137% L2 mean/sum. This proves CCE
  repeat variability independently of Opt3, but does not attribute the full-model
  difference. Do not assert full-model mask equivalence based on this capture.
* Forward/backward 115.313/217.508 s; allocated GPU peaks 49.529/58.277 GiB,
  each 0.245776 GiB below CCE v3. Reserved peaks unchanged. Sampled CPU RSS
  130.325/129.167 GiB; no established RAM saving. Combined time +0.349%.
* All 70 files restored identically from DVC; tree
  `cc93515b1b120fc1bccd14ea1625f459.dir`. Cached reproduction skips the stage;
  this verifies artifacts, not numerical replay.
* `v4.md`, master `v0.md`, metrics/review preserve results, source identities,
  numerical limitations and 130K storage arithmetic. No successor GPU job started.

# Latest CCE exact experiment — 2026-10-09

* User authorized CCE exact without gradient filtering, compared with native,
  v1 and Liger. Primary agent executed directly through Kaggle Jupyter; no agents.
* `cce-exact-v3` completed at execution commit `c3da119`, attempt
  `20261009164944-23fb1fb3`. Official Apple source commit
  `3de376c106a1916bc5e1b619f9c77c87a461ee1c` (source version 25.9.3), isolated
  archive pinned by SHA256; PyPI 25.1.1 lacks the preset and was not used.
* Frozen head; `impl='cce_exact'`, `filter_eps=None`, both e/c filters false,
  both FP32 accumulation flags true. Actual effective options and all 15 imported
  CCE source hashes were independently checked against the archived source.
* Full-model loss `0.6256128549575806`, native delta -8.237361907958984e-5.
  All 744 A/B gradients finite/nonzero, 0/744 bitwise identical; global L2
  error 1.205940% (Liger 1.609730%, about 25.1% less error), worst tensor
  2.499718%, cosine 0.9999273038627808. No clipping/update or production promotion.
* Forward/backward 115.109 / 216.554 s; allocated GPU peaks 49.775 / 58.523 GiB;
  sampled process RSS peaks 129.066 / 129.270 GiB. No additional whole-model
  GPU/RAM peak saving over v1/v2. Zero vocabulary-shaped CPU saved tensors.
* Interleaved mean/sum operator losses match native. Hidden gradient errors
  0.333004% / 0.332454%, versus Liger 5.03–5.06%; operator GPU peak 2.598 GiB.
  Full-model adapter agreement remains much less close than this operator result.
* `v3.md`, master `v0.md`, config/stage, metrics and independent review record
  all results. All 54 files restored identically from DVC. Numerical qualification
  remains failed; successful pipeline execution is not acceptance of a tolerance.
  Production remains v1. Prior Liger stage is frozen; no successor GPU run started.

# Latest experiment update — 2026-10-09

* User authorized target-only Liger FLCE as the next optimization. Primary agent
  executed one configured capture through Kaggle Jupyter; no subagents.
* `liger-flce-v2` is a **diagnostic candidate, not promoted**. Execution commit
  `ee639cd`, attempt `20261009162024-bcec6367`, frozen head, same nonzero test
  fixture/sample/native identity, no clipping or optimizer update.
* Loss `0.6256953477859497` differs from native by 2 FP32 ULPs. All 744 adapter
  tensors are finite/nonzero, but 0/744 match byte for byte. Global relative L2
  error 1.609730%, worst tensor 4.545067%, cosine 0.9998704996893791.
* Forward/backward 114.392 / 215.512 s; GPU allocated peaks 49.775 / 58.523 GiB;
  sampled process RSS peaks 131.596 / 129.275 GiB. No whole-model GPU/RAM peak
  improvement over v1. Removing loss/head allocations leaves a peak elsewhere;
  this run did not identify that operator.
* Liger uses 8-row BF16 logits chunks and saves selected hidden gradients.
  Zero vocabulary-shaped CPU saved tensors; no full T-by-V backward scratch.
  FP32 CE statistics/scalar remain; only dense FP32 matrices are bypassed.
* Interleaved synthetic mean/sum operator losses match, but hidden gradients
  differ by 5.03–5.06% L2. Operator GPU peak 1.712 GiB. Raw operator and adapter
  gradients, wheel, source snapshots, timing/RAM telemetry are DVC outputs.
* `v2.md`, master `v0.md`, metrics and independent review record results. The
  successful DVC stage means finite recorded execution, not numerical acceptance.
  Validated production trajectory implementation remains v1. Its accepted DVC
  stage is frozen while the shared runner evolves. No follow-up GPU job started.

# Latest resume state — 2026-10-09

This section supersedes the original immediate-action instructions below.

* **Latest optimization:** `target-mask-v1` passed. Native loss
  `0.6256952285766602` and all 744 gradient archive bytes match exactly,
  including all nonzero A/B tensors. Forward/backward: 116.259 / 217.347 s;
  GPU allocated peaks 49.775 / 58.523 GiB; sampled CPU RSS peaks
  129.063 / 129.273 GiB. `v1.md` and the `v0.md` table record the result.
  The target-mask module handles arbitrary interleaved labels, preserves the
  native reduction positions using a tiny T-by-1 NLL buffer, and retains a
  full-shaped native head backward scratch. Two CPU-packed FP32 vocabulary
  matrices have 651 rows each, rather than 16,249 rows.
* Interleaved synthetic CUDA operator checks passed bitwise mean/sum loss and
  raw hidden gradients on four separated spans at 16K. Eight CPU target-mask
  tests and eight runner/worker tests passed. This plus the complete-model
  anchor pass does not bypass the independent all-assistant production gate.
  Its source identity now includes `target_only_head.py`.
* The `target_mask_v1` stage/config saves all raw operator and adapter gradients
  through Kaggle Jupyter. `reference_v0` is frozen to retain its accepted
  snapshot while shared runner code evolves. No additional optimization or
  optimizer-update run has been started; discuss the next change with the user.

* **Latest user instruction:** the primary agent now executes all work directly;
  do not use or restart subagents. Use the Kaggle Jupyter API, without computer
  use.
* Each new experiment must have a config, DVC stage and recorded artifacts.
  `kaggle_reference_run.mjs`, `kaggle_reference_worker.py`,
  `configs/reference-v0.json` and `dvc.yaml:reference_v0` implement the first
  version. `v0.md` is the experiment table and includes phase measurements,
  numerical identities, prerequisites and restore/recovery commands.
* The single config-driven `reference-runner-v0` capture completed with loss
  `0.6256952285766602` and the same raw gradient SHA256 as both previous
  nonzero references. Forward: 118.260 s, 77.802 GiB CUDA allocated peak,
  139.093 GiB sampled process RSS peak. Pure backward: 218.232 s,
  85.100 GiB CUDA allocated peak, 147.794 GiB sampled RSS peak.
  CPU/device use is sampled every 2 s; allocator peaks are exact. Loading,
  pre-backward adapter export and gradient export are separate phases.
  The Jupytext collector fix resumed the existing run, without another GPU
  pass. Complete evidence and frozen source/config bytes are DVC outputs;
  Git metrics are `metrics/reference-v0.json`. No candidate run is authorized.

* Branch: `codex/flash-next-full-context-training`, PR #24.
* The requested zero-B native capture completed and is preserved in
  `gradient-results/reference-new-server-one-run-v1`: loss
  `0.6243623495101929`, all 744 finite raw gradients, 372 zero A gradients,
  no clipping/update. Do not recapture it automatically.
* A **test-only** reproducible nonzero A/B fixture now exists in
  `gradient-results/nonzero-diagnostic-adapter-v2`. All 372 A and 372 B tensors
  are nonzero; two fresh CPU processes reproduced identical bytes. The pinned
  adapter SHA256 is
  `49e0960ba1f5a2435e47180262a27e652dd797397a5a11ab13425ef2cf041b6f`.
  Production initialization rejects this fixture. Read
  `NONZERO_DIAGNOSTIC_ADAPTER.md` before reuse.
* Its native capture `gradient-results/reference-nonzero-ab-v2` completed with
  loss `0.6256952285766602`, all 744 gradients finite and nonzero, no
  clipping/update. Root independently decoded and verified the raw tensor
  storages. Gradient SHA256:
  `91465990e81dd57bb29368510354674415c1f1d8bbbaba5225b62fdf4a8cced2`.
* The single authorized fresh-process replay completed in
  `gradient-results/reference-nonzero-ab-repeat-v1`. Loss FP32 bits (`3f202d90`)
  and all 744 tensors' raw bytes match exactly, including signed-zero bits.
  Both serialized gradient files have the SHA256 above. Source manifests,
  runtime versions, sample/adapter/config identities and environment match.
  Root independently verified the downloaded raw tensor storages and hashes.
  No clipping/update occurred and no candidate sweep is authorized.
* The user authorized committing the code/docs to Git and these session
  artifacts to DVC, then pushing both remotes. Raw `.pt` and ZIP files are
  excluded from Git by explicit directory ignore entries; `.dvc` pointers
  restore the complete evidence, including independent review reports.
* The executed native reference script is unchanged: last modifying commit
  `c7ff17e78776a9888b280531a0c53718efa45c02`, SHA256
  `f95893baa503ec446d4610878b7d5feb7ee975e9313e282349cac2d8952573a0`.
* Use the Kaggle Jupyter server API directly for kernel access and transfers.
  The user explicitly rejected computer-use automation for this work. Private
  URL is in `/tmp/kaggle_probe_url` for this local session only; never print or
  archive it.
* `GRADIENT_MEMORY_REVIEW.md` lists historical gradient evidence and 130K memory
  implications. The approximately 402 GiB stock-reference GPU estimate is an
  extrapolation of measured head allocation; the 668 GiB host calculation is
  illustrative only. The initially stated 600–700 GiB host range was withdrawn.
* PR #26's new dataset outputs were verified in the main local checkout against
  all four DVC MD5s: 58 trajectories, 25 games, 1,334 targets, 1,399,743 target
  tokens / 5,712,174 total (24.5%). Twelve trajectories exceed 32,768 targets;
  maximum 43,806. A fixed 32,768-row backward cannot cover this dataset. The
  trajectory output objects were absent from DVC remote at the time of review.

# Original resume prompt (historical)

Continue Qwen3.8-Flash-Next W4A16 + LoRA gradient qualification and long-context
training in `LamDang/arc-agi-3-solution-da-fr`, branch
`codex/flash-next-full-context-training`, draft PR #24:
https://github.com/LamDang/arc-agi-3-solution-da-fr/pull/24.

First read this handoff. The user stopped the previous work because it was too
slow, then authorized **one native reference forward/backward** to save its loss
and all adapter gradients, with the script commit hash. That run had **not
started** at handoff: the final server check showed **0 MiB GPU memory, 0%
utilization, no GPU compute processes, and no new one-run output directory**.
Agents are paused. Do not resume an old automatic experiment chain.

Use one **GPT-6.1 Sol** subagent for execution, with the main agent reviewing its
reports. The user explicitly requested this division. Timebox the next capture
to 20 minutes; report a concrete failure instead of waiting indefinitely. Do not
rebuild an already usable environment. Do not investigate the old/new scalar
loss difference before this capture: the user accepted the new-server number.

## Immediate next action: exactly one native capture

1. Check the supplied Kaggle server and existing processes before launching
   anything. Reuse its restored environment and 256-expert export. If another
   capture exists, inspect/collect it rather than launch a duplicate.
2. Run official HF + AutoRound + PEFT on the unchanged diagnostic sample and
   saved initial adapters, using stock native model operators, native layer
   checkpointing and PyTorch CPU saved-tensor storage. No custom model backend,
   candidate loss, optimizer update or clipping. Use the existing native capture
   mode in `overfit_hf_reference.py` (`--first-pass-only`), after checking its CLI.
3. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and deterministic PyTorch
   mode. The missing allocator setting caused the most recent OOM.
4. Use a NEW output directory, e.g.
   `/kaggle/working/gradient-audit-20261009/reference-new-server-one-run-v1`.
   Preserve all old evidence. Save all **744 raw named adapter gradients** before
   clipping/update, including zeros; loss; finite/norm/dtype/shape statistics;
   actual peak GPU/host memory and elapsed time.
5. Record repository HEAD, last commit modifying the executed capture script,
   actual executed script SHA256, imported source hashes, exact command,
   sample/adapter/model identity, package versions, GPU and numerical settings.
   A commit label alone does not prove the executed file matches Git.
6. Download the loss/gradients/metadata into the shared workspace, verify hashes,
   report paths and metrics, then STOP. No successor GPU job is authorized by
   this one-run instruction.

The accepted latest native forward loss is **0.6243623495101929**. It appeared
in both v3 and v4. **No complete gradients for that loss are saved yet.** The old
reference loss **0.6247151494026184** and its 744 gradients remain preserved,
but must not be silently mixed with the new-server reference. The user said the
new number is fine and asked to keep it.

## Repository and completed implementation

- Existing worktree: `/workspace/flash-next-training-pr`.
- Do not modify unrelated user edits in
  `/workspace/arc-agi-3-solution-da-fr`.
- `d82a0f0`: native head autocast fix, read-only dtype hook, isolated mixed-dtype
  harness and evidence.
- `ddd9906`: separate trajectory preparation/training/qualification pipeline.
- Last commit modifying `overfit_hf_reference.py`:
  `c7ff17e78776a9888b280531a0c53718efa45c02` (verify the executed remote file
  SHA256; no new capture has run yet).
- Final combined CPU suite: **113 passed, 5 CUDA skips**.
- Preserve the legacy fixed reference encoder, sample and objective.

Data files: `trajectory_data.py`, `trajectory_encode.py`,
`prepare_trajectories.py`, `TRAJECTORY_DATA.md`, and
`tests/test_trajectory_data.py` under `exp/sft-flash-next/train`.

Training files: `trajectory_head_loss.py`, `trajectory_training.py`,
`trajectory_checkpoints.py`, `trajectory_initialize.py`, `trajectory_run.py`,
`trajectory_gradient_audit.py`, `trajectory_gradient_gate.py`,
`TRAJECTORY_TRAINING.md`, and their tests.

Production policy: **one complete trajectory per game; supervise every assistant
turn**, including generated thinking, tool calls and final answer. User/tool/image
observations remain context with ignored labels. Fold 0 is validation, disjoint
by game. The user selected an **approximate supervised-token budget with whole
trajectories**: accumulate summed CE, divide gradients once by the actual target
count BEFORE clipping/update, log overshoot and final short updates. Do not
average trajectory means or carry stale gradients across updates.

Version-2 checkpoints preserve partial raw sums, target/trajectory counts,
optimizer/RNG/cursor and immutable plan identity. Evaluation explicitly loads a
trained checkpoint; it must not silently evaluate initial adapters. Production
initialization is clean random-A/zero-B with no validation exposure. The
nonzero diagnostic adapter may qualify operators, but cannot initialize actual
production training.

## Server, model and private access

Current Kaggle notebook ID: **356749920**, supplied by the user. RTX PRO 6000
Blackwell Server Edition, **94.97 GiB usable VRAM**, **175 GiB host limit**.
The old server was inaccessible; the new server started without temporary
exports/runtime files, hence the restoration work. It is the same intended
model, not a newly selected model family.

Private bearer URL is only in `/tmp/kaggle_probe_url`; NEVER print, commit or put
it in this handoff. `/tmp/kaggle_exec.py` executes a local Python file through
Jupyter and sanitizes outer exceptions. Local Python:
`/workspace/sft-venv/bin/python`. Read the available cloud-runtime skill for
network/credential setup if needed. Restricted-network commands have required
an explicit per-command network grant. Do not expose tokens/credential values.
If this is a fresh workspace, obtain the current server URL from the user;
do not assume private `/tmp` files carried over.

Remote paths:

- Source model:
  `/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1`.
- Byte-preserving 256-expert export: `/tmp/reference-256-hf` (already built).
- Training sources:
  `/kaggle/working/training-gradient-audit/exp/sft-flash-next/train`.
- Evidence root: `/kaggle/working/gradient-audit-20261009`.
- Sample: `<evidence>/overfit-sample-16k/sample.pt`.
- Initial adapters: `<evidence>/overfit-hf-v3/initial-adapter.pt`.
- Old reference: `<evidence>/native-deterministic-repeats`.
- `reap_model.py` dependency restored into the expected `exp/reap-flash-next`
  source path. Check imports; do not repeat the v2 missing-module failure.

Runtime restored: torch2.11.0+cu128, Transformers5.18.0, PEFT0.20.0,
AutoRound0.15.0, FLA/fla-core0.5.2, causal-conv1d1.7.0. The process-local package
view excludes incompatible torchao0.10.0. Do not bulk-upcast the frozen PLE
table using generic `prepare_model_for_kbit_training`.

Fixed sample: **16,249 tokens, 15,598 prompt, 651 targets, 7 images**.
Sample SHA256:
`48f6f88b7bff3b2be5b823c7394388d6b530e26febd25e4c21d3c0b13fdd49ff`.
Initial adapter SHA256:
`9d93ddb8676d5490d55fa9969d99352189883a4e53f8d189cc2b9d3bc4574a56`.
Model config SHA256:
`ed8de086c12d789969ff380ea41d353eacb13fd5431e40b66c79bde5257ecf17`.

Latest v3 native replay failed backward trying a 15.03 GiB allocation, with
70.07 GiB allocated and 20.37 GiB reserved unused. v4 restored the original
allocator, reached forward loss0.6243623495, then was terminated by user request
before backward finished. Its PID1276 was stopped; final GPU check was idle.
Old collector PID36488 was also stopped. Preserve v3/v4 failure/partial evidence.

The user explicitly approved uploading the prepared two-turn test sample
(including game images/generated thinking) and training code to this server,
**excluding raw source logs**. An earlier automatic upload rejection was
resolved by that explicit approval. Do not include `source.json` if it contains
raw logs. Remote production files may precede the final review fixes: verify
their hashes against Git before later production tests.

## Qualification status: do not overstate it

- Old deterministic native repeats matched all744 gradients bitwise, including
  fresh-process replay. Old learning reference reduced loss98.36% after5 updates.
- `native_mask_storage=True`, `offload="cpu"`, and their combination passed all744
  gradients bitwise on the old initial zero-B reference. Trained-adapter and
  new multi-span qualification remain pending.
- Original selected-logit and chunked-loss candidates FAILED (~1.2% gradient
  relative error). Do not promote them.
- Revised padded native head: **8 isolated mixed-dtype GPU cases passed bitwise
  loss/FP32 hidden gradients**. Operator peak10.40–10.47 GiB versus native46.50
  GiB. This is synthetic-hidden OPERATOR evidence, not full-model744 proof.
- Raw report/pairs:
  `/workspace/quant-compat-audit/native-gradient-audit/native-head-mixed-dtype-v1`;
  compact evidence is committed under `gradient-results`.
- Root independently reviewed raw mixed-dtype evidence and code. Further
  gradient tests must exercise nonzero A AND B adapters.
- New multi-span gate requires its own native summed-CE reference and raw744
  evidence. Legacy final-reply qualification cannot qualify it.

## Data blockers and locations

Full original corpus restored and MD5-verified: **1,334 requests /25 games**,
332,843,217-byte train.jsonl and192 raw-log files. All25 histories reconstruct
through verified chronological overlaps. Never use only the last trimmed
request as a complete trajectory. Historical thinking differs from regenerated
thinking; canonicalize each owned assistant turn consistently.

Sources live under `/workspace/trajectory-source`. Generated progressive source
is incoming PR25 branch `codex/progressive-sol25-20261008`, DVC directory
`ebbbafb37df3a9d557417bed1e8c015d.dir` for
`ARC3-Inference/runs/think-progressive-sol25`. Canonical records are
`turns/<game>_p0/<index>/final.json`. **585/1,334 verified;749 missing** due
persistent Envoy HTTP503. Low-concurrency bounded retries already attempted;
do not keep retrying indefinitely or silently substitute old thinking.

Production130K prep failed atomically for20/25 games; no production directory
was published. Four fully covered games exceed130K:
bp35=380,908; dc22=286,725; ka59=162,641; lf52=406,965 tokens.
Sixteen other games lacked generated-source coverage. Covered games within130K:
ar25=81,983/18,556 targets; cd82=107,567/27,523; cn04=90,672/25,124;
ft09=52,858/10,965; g50t=110,024/26,929.

Audits: `/workspace/trajectory-source/production-130k.audit.json`,
`length-audit.json`, `final-source-coverage.json`. Qualification-only coherent
two-assistant ar25 prefix: **7,595 total tokens/960 targets**, already supplied
to the training agent; locate its local/remote fixture files before testing.
It is not a production full-game sample.

Raw datasets and large gradients are NOT Git assets. Existing private DVC
remote: `s3://kaggle-arc-agi-3-dvc`. Pinned source DVC files and prior reference
archive are in the repository. Shared local evidence also lives in
`/workspace/quant-compat-audit/{native-gradient-audit,overfit-reference}`.
On a fresh workspace, restore from DVC/server where available; do not claim
that committed summary JSONs contain the raw gradients.

## Proposed experiments AFTER the one-run report

Discuss/choose the next test; no automatic chain:

1. Revised target-only head vs new captured gradients (roughly6–12min/pass).
2. Independent multi-span/nonzero-adapter native gate with **32,768 backward
   rows** (15–25min).16384 cannot cover actual18K–27K target counts.
3. Native checkpoint groups3/6 with CPU offload (6–12min/pass) to bound host
   activations alongside the95GiB PLE table.
4. Native bounded RMS/gated-normalization/residual mixing, separately and
   combined (6–12min/flag).
5. Disk offload only if host memory still requires it (10–20min;~32GiB scratch).
6. Qualified90K/130K capacity and speed probes (provisionally1–3GPU-hours, may
   OOM or take longer). Do not use old custom-backend two-minute timings to
   predict native HF speed.

Last night's custom backend completed130K composite steps around124s/74.28GiB
GPU, or131.8s/71.42GiB using whole-GDN blocks. Those runs do **not** establish
native gradient equivalence or new multi-span capacity. Many custom blocks,
sparse attention and disk/checkpoint combinations remain unqualified.

Complete games longer than130K require a user decision about context handling;
no truncation, segmentation or game exclusion has been authorized. Increasing
a preparation limit alone does not establish model/GPU capacity. Keep the
fixed-reference files unchanged and distinguish operator proof, full-model
proof, capacity success and learning quality in every report.
