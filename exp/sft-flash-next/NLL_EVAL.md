# Budgeted teacher-transcript NLL evaluation

Design and implementation notes, 2026-10-08. The user increased the core to
**30 requests**. The implementation and staging instructions are in
[nll/README.md](nll/README.md). This protocol supersedes
the full-fold and gameplay requirements previously proposed in PLAN/RESEARCH.
No GPU evaluation has been run for this design. Implement against the dataset
merge on main `37fadfeedfd54d2129db4525e4167596cc7337b6` or a descendant.

## 1. Recommendation and scope

Start with **30 validation requests, six per game, evaluated at 512, 448,
384, 320 and 256 experts**. Score every token of each selected final teacher
reply, conditioning on the complete recorded multimodal request. Select the
base from this paired NLL comparison and its memory requirements. There is no
gameplay, generation or training pilot in this selection budget.

Use the existing forward replay on **Kaggle's interactive Jupyter server,
RTX PRO 6000 Blackwell 96 GB**. Load the full W4A16 model once and change its
expert masks between replays. Keep an optional reserve of up to ten additional
requests for the full reference and at most two finalists. Do not spend that
reserve automatically.

At an illustrative 51,280 processed tokens per request and the repository's
historical replay rate, the first comparison costs **about 151 minutes of
scoring**, plus startup/checks for each session. At 500 tokens/s it is
closer to **256 scoring minutes**. The real panel's token sum, measured initial speed and
a hard session deadline control execution; this is not a promised runtime.

## 2. What the loss means

For request i, x_i is the complete prompt, including prior messages, tool
results and images; y_i is the final sol reply in the merged student format.
For model m and target token t:

    nll[m,i,t] = -log p_m(y_i[t] | x_i, y_i[:t])
    S[m,i] = sum_t nll[m,i,t]
    T[i] = number of target tokens
    L[m,i] = S[m,i] / T[i]

Use natural logarithms: units are nats/token. Lower is better; perplexity is
exp(NLL). This is exactly the supervised token prediction loss used for
transcript imitation, before averaging/weighting examples. It is a direct,
affordable measure of fit to the available teacher outputs. It does not
measure the teacher's full probability distribution or guarantee which model
will improve fastest under LoRA. We accept that limitation to save GPU time.

**“Every token” means every final-reply target token**, including its emitted
formatting and end-of-turn token. Earlier assistant replies, user/system
text, images and tool results are context, not scored targets. The generation
prefix, including an already supplied opening `<think>`, is also masked.
Computing losses on all prompt tokens would change the objective and greatly
increase vocabulary-head work; it is unnecessary for choosing this SFT base.

Teacher forcing uses the teacher's preceding target tokens, not newly sampled
student tokens. There are no decoding parameters, game episodes, tool
execution, gradient storage or optimizer states in this evaluation.

## 3. CPU preparation and reproducible sampling

### Inputs and split

Read `data/sft-gpt61sol-features-25games/index.json` and the existing
`data/game_folds/folds.json`. Use fold 0: **sk48, sp80, tn36, cd82, ar25**;
join on full game IDs. The other 20 games are training/calibration only.
The actual index and JSONL are DVC payloads not available in this workspace,
so this document does not invent selected row IDs or validation row counts.

Fetch and verify the locked payload on a CPU preparation machine. Index rows
already contain byte ranges, game, request index, level, approximate context
and output lengths, and image count. Read only selected JSONL rows by seek;
use the pinned real processor to verify their exact expanded token lengths,
images and target boundaries before a GPU is allocated. Do not shorten a
context to make a selected row fit the budget.

### Core panel: 30 requests

1. For each game, sort eligible requests by `(context_tokens, request_index,
   line)`. Divide the sorted list into three nearly equal-count strata:
   short, medium and long **relative to that game**. Ties are broken by the
   recorded identifiers. These strata are fixed from metadata, not losses.
2. Uniformly sample two requests per stratum, without replacement, using
   seed `20261008` and a pinned sampler implementation/version. Persist all
   population counts, boundaries, RNG details and selected IDs. With N_gb
   rows in a stratum and two draws, inclusion probability is 2/N_gb.
3. Use the exact same 30 requests, inputs, targets and order for all models.
   Do not reroll the seed to get cheaper requests or more favorable losses.
   An invalid row is a data-contract error, not permission to silently drop
   it. Empty/small games require an explicit revised allocation.
4. Emit a coverage table before GPU work: per game, level-at-request,
   absolute length bin, target length, image count and output type. Record
   unsampled cells explicitly. Quantile strata ensure useful comparisons even
   if, for example, one game's “short” requests are all longer than 32K.

This is a probability sample with known weights, not a claim that 30 requests
represent every behavior. It avoids evaluating every turn of the same game
and avoids selecting only early, short or successful-looking requests.

### Reserve: up to ten requests, selected before seeing losses

Prepare two more requests per game, excluding the core. Choose with a
deterministic metadata-only priority: an unrepresented level first; then
uncovered context extremes, unusually long replies or a missing output type;
break ties with the recorded seed. The first five form a one-per-game round,
as do the next five. Retain the terminal text-only sk48 reply as a tagged
candidate if it fills a missing output type, not as an extra mandatory job.

The reserve is a **targeted diagnostic sample**. Its inclusion probabilities
are not the core's probabilities: report it separately and never silently
pool it into an estimate of the full fold. Evaluate only the reference and
up to two finalists on the same reserve requests. Use it if the core's choice
is unstable or a level/length/category gap could change the decision, and only
if the remaining token/time budget covers a complete five-game round.

Fold 0 contains **35 game-level combinations** according to `folds.csv`.
Thirty requests can cover at most thirty distinct combinations, so the core
cannot cover all 35. A later reserve may improve coverage but does not guarantee
one request from every level. Report unobserved levels as
`not sampled`, never zero loss. Full level coverage would require at least
35 eligible requests, one per observed game-level, and still would give very
noisy level estimates. That is an optional later expense, not this baseline.

### Pruning maps without new GPU calibration

First try the saved per-run/per-game REAP statistics under
`exp/reap-flash-next/results/kaggle-20261006/calib.dvc` (about 122 MB on disk).
Verify run provenance, map run keys to games, and aggregate **only available
training-game runs**, excluding all five fold-0 games. Record which of the 20
training games actually have statistics. Reuse `analyze.py` with the fixed
`gate_norm` criterion and context/generated/image categories; select a single
stable per-layer ranking and nested 448/384/320/256 subsets. Keep top-10 routing
and shared experts unchanged. This aggregation needs no GPU replay.

The existing all-game `keep.json` files are validation-exposed. If separable
statistics cannot be recovered, do not silently schedule fresh calibration:
evaluate those deployed maps as an explicitly **validation-exposed operational
comparison**, retaining that limitation in selection.json. They cannot yield
a clean game-disjoint estimate. New train-only maps are different models from
the existing published pruned checkpoints and must be exported accordingly.

## 4. Correct rendering and token labels

Reuse the merged dataset's `messages`, code-only `tools`, template kwargs and
pinned tokenizer/processor. Render the prompt with `add_generation_prompt=True`
and the complete sample without it. Assert both string prefix equality and
processed token prefix equality, including image expansion. Let P be the
processed prompt length; score target positions P through N-1. Position P is
predicted by the hidden state at P-1, including across a replay chunk boundary.
No truncation, synthetic message packing or history loss is allowed.

The current `render.py` needs adaptation: its SGLang tool normalizer adds
fields to tool schemas, and its GENERATED scanner labels **all** historical
assistant replies. Neither behavior should be reused unchanged for this data.
Use an explicit final-target mask independent of the category annotations.

Assign mutually exclusive labels to final-target tokens:

| Label | Meaning |
|---|---|
| `thinking` | Recorded rationale plus `Next step :` description in `reasoning_content` |
| `tool_code` | Serialized value of the Python `code` argument |
| `tool_format` | Tool-call wrappers, function/argument names and surrounding syntax |
| `assistant_text` | Ordinary final assistant prose, including the terminal text turn |
| `turn_format` | Emitted thinking delimiters, other turn delimiters and end-of-turn |
| `boundary` | A token overlaps two semantic spans or has no unambiguous ownership |

“Thinking” here is the teacher's **logged rationale**, already converted into
student format; it is not unavailable private teacher reasoning. Tool results
are prompt context, so there is no tool-result target NLL.

Derive character/byte spans from structured target fields and the exact pinned
template serialization, then map them through tokenizer offsets. A span-aware
mirror of the template must reproduce the actual rendered text byte-for-byte;
it must never insert markers into model input. Do not locate spans with a naive
regex over `<think>` or code strings that can also occur in history/literals.
Account for serialization escapes and special-token offsets explicitly.
Overlapping tokens go into `boundary`, counted once, not arbitrarily reassigned.
Align the text-token mapping with the processor's expanded image positions.

All scored tokens must have exactly one label; category sums/counts must
reconstruct overall sums/counts. Reject a broken target boundary. If category
mapping alone is ambiguous, retain correct overall NLL, report affected
tokens as `boundary`, and withhold unreliable category conclusions.

Join every token to its request's game, **level at request time**, prompt
length, target length, image count and request index. A tool call may advance
multiple levels; without execution we cannot assign individual code tokens
to the levels they will later affect. Use fixed report bins for prompt length:
`<16K`, `16K–<48K`, `48K–<80K`, `>=80K`, where K=1024. Keep exact lengths too.

## 5. GPU execution: small changes to a working path

Use `exp/reap-flash-next/reap_model.py` and `replay.py`, not a new SFT framework
or a generation server for this job. Their bounded indexed-attention and PLE
offload paths already have long-context forward evidence on this GPU.

Proposed worker loop:

    load_model(..., record=False); model.eval()
    validate kernels and one representative scoring example
    for request in frozen interleaved_game_order:
        encode once; verify input and target hashes
        for candidate in fixed_rotating_expert_order:
            if matching completed result exists: continue
            set_pruning(model, candidate.map_or_none)
            replay complete request with a NEW cache, no gradients
            score only final-target predicting hidden positions
            atomically save token NLL and request summary
        refresh report and externally persist completed results

Start with the working replay chunk size 8192; lower it consistently if the
capacity check requires it. Use a vocabulary-head block of 256 target positions
initially. Compute exact full-vocabulary cross-entropy in FP32, reduction none;
sum saved FP32 token values in FP64 on CPU for reports. Do not approximate the
normalizer with top-k logits. The existing replay already computes this vector
and can return `token_nll` with `predictions=True`; the proposed change also
returns absolute target positions and avoids unnecessary argmax/hidden dumps.

The full model stays resident while router masks emulate retained experts;
all non-expert weights, quantization and kernels are held fixed. The repository
has a tiny-model masked-router versus physically-pruned parity test in
`exp/reap-flash-next/tests/test_prune_checkpoint.py`; run it during implementation.
This is not yet a numerical certification for every packed production export.
Never share a KV/recurrent cache across expert configurations: their entire
histories must be recomputed. Never change masks midway through a request.

Masking saves checkpoint loads, **not resident weight memory**. Report actual
sweep VRAM and separately estimated physically-pruned memory; this sweep does
not measure the final pruned checkpoints' deployment speed or training fit.
The A100 80 GB is not the default for this full-resident sweep: historical full
replay peaks exceed 80 GB. An A100-only run needs a separately qualified load/
offload strategy or physical pruned checkpoints; do not claim the 512 reference
fits or transfer the Kaggle time estimate to Colab.

Before the panel, use tiny synthetic tests for mask/shift/span/chunk correctness
and repeat one short selected real request at the full model to quantify
numerical variability. Count the first valid run toward the panel. Check the
longest selected input for capacity early, also retaining its valid result.
Use the existing kernel checks; compare alternative chunk boundaries on a
short input. Treat differences on the scale of the measured numerical noise
as unresolved, rather than running repeated copies of the entire panel.

Do not require prefix-tree cache reuse in version one. Requests can have
trimmed/rewritten histories, and accidental prefix equivalence would invalidate
NLL. Identical vision-feature caching is optional after equality checks, with
a bounded CPU cache. The budget below assumes **no reuse across requests or
models**, so neither optimization is needed to meet the stated estimate.

## 6. Aggregation, reports and choosing a base

For core request i from game g / stratum b, let w_i=N_gb/2 (more generally
N_gb/n_gb for stratified simple random samples). Report:

    game_request_nll[m,g] = sum_i(w_i * L[m,i]) / sum_i(w_i)
    primary[m] = mean over five games of game_request_nll[m,g]
    token_nll[m] = sum_i(w_i * S[m,i]) / sum_i(w_i * T[i])
    token_perplexity[m] = exp(token_nll[m])

The primary metric weights requests equally within each game's population,
then games equally. Equal-request loss matches the planned SFT averaging
within a game; the deliberate equal-game evaluation prevents one long game
dominating selection. Also report the fold-population-weighted mean request
loss and token NLL so the weighting choice is visible. The token-NLL ratio is
a sampled estimate, not an exactly unbiased population ratio. Do not average
request perplexities or change the selection metric after seeing rankings.

For each label, game, sampled level and length bin, retain loss sums, token
counts and request counts; report token-weighted NLL and paired delta against
512 on identical tokens. Missing categories have null NLL, not zero. Show
unweighted panel summaries alongside population estimates, and label targeted
reserve summaries separately. Keep granular cells descriptive: six requests
per game cannot support precise game × level × length × category comparisons.

Produce a CPU-only HTML report and CSV/JSON summaries containing:

- A model comparison: primary NLL, delta vs full and best, token NLL/PPL,
  thinking/code/format NLL, processed tokens, elapsed time, actual sweep VRAM.
- Per-game and length-bin paired deltas, with sample/token counts; a level
  coverage matrix that displays missing cells rather than interpolating them.
- A selectable transcript view colored by token NLL or delta against full,
  grouping tokens into rationale paragraphs and code lines for readability.
  Show raw nats in hover/details; common color scales across models. Local
  HTML can be generated without any additional GPU calls.
- The largest request/category regressions, numerical-check results, failed
  or unfinished jobs, and five leave-one-game-out comparisons of the ranking.

No token-bootstrap significance claims: tokens and nearby game turns are
correlated, and the core has only two requests per game/length stratum. With
five games, even a game bootstrap is weak. Paired deltas, individual games and
ranking sensitivity are more useful here than impressive-looking error bars.

**Predeclared practical selection rule:** find the best primary NLL, then
prefer the smallest expert count within **0.05 nats/token** of it, subject to
the intended hardware's plausible training-memory envelope. This tolerance
corresponds to about a 5.1% increase in exp(primary NLL), not a 5% drop in game
score; it is an engineering choice, not a literature-established optimum.
Show the complete NLL-versus-estimated-memory curve so the tradeoff is visible.

Flag a candidate if its tool-code delta against full exceeds 0.10 nats/token,
any game delta exceeds 0.15, or the choice changes under leave-one-game-out
analysis. These are diagnostic triggers, not statistical tests. If a trigger
could change the choice, spend one reserved five-game round on full plus the
two competing candidates, provided it fits the cap. Use the separately
reported reserve to detect contradiction or localized damage; do not claim
it refines the population estimate. If uncertainty remains and quota is
exhausted, choose the more conservative feasible candidate and record the
uncertainty instead of automatically expanding the experiment.

NLL alone cannot prove backward/training fit. A candidate is selected for
imitation quality with training feasibility still pending. Keep the W4A16
checkpoint and map hashes with the result. If later QLoRA qualification uses
a different NF4 base, this ranking remains evidence for the W4A16 family;
baseline the actual chosen training base before training rather than implying
that quantization cannot change its NLL.

## 7. Time, memory and logging estimates

### Evidence and assumptions

The merged dataset reports median context about 51K tokens, median reply
about 280, maxima about 108K and 4.3K respectively. A median is **not** the
sample mean: 51,280 total tokens/request below is an illustrative planning
assumption. Replace it with the sum of exact selected processor lengths.

`exp/reap-flash-next/README.md` reports full-model replay at 845–863 tokens/s
on 52K–118K multimodal sequences, about 950 tokens/s with FLA, and 730 seconds
for a cold load (12–21 seconds warm). Those measurements used older Qwen traces
and many more scored historical assistant tokens, not this new sol panel.
Budget at 850 tokens/s nominal and 500 tokens/s conservative. Hold speed
constant across expert counts: retaining top-10 routing does not imply
proportional speedup from removing half the expert bank.

| Work | Requests × models | Processed tokens, illustrative | Scoring @850 tok/s | Scoring @500 tok/s |
|---|---:|---:|---:|---:|
| Core | 30 × 5 = 150 | 7.692M | 150.8 min | 256.4 min |
| Optional first reserve round | 5 × up to 3 = 15 | 0.769M | 15.1 min | 25.6 min |
| Optional two reserve rounds | 10 × up to 3 = 30 | 1.538M | 30.2 min | 51.3 min |
| Core plus first round | 165 replays | 8.461M | 165.9 min | 282.0 min |
| Core plus both rounds | 180 replays | 9.230M | 181.0 min | 307.7 min |

Add **20–30 minutes per cold session** for loading, compilation/warmup,
correctness checks and persistence. A 120-minute cap means the nominal core
likely needs two sessions: about **3.2–3.5 hours total**. At 500 tokens/s it
may need three sessions: about **5.3–5.8 hours total**. These are planning
scenarios, not measured sol-panel performance. Package downloads, environment
repair and fresh calibration are excluded; stage inputs and wheels on CPU.
The implementation runs the core only; reserve work is a later explicit job.

For actual planning, use:

    scoring_seconds = sum_over_jobs((prompt_tokens + target_tokens) / rate_bucket)
    total_seconds = cold_start + checks + scoring_seconds + persistence_margin

Bucket rates by observed context/image load once available. At 850 tokens/s,
one 16K/51K/108K request costs roughly 19/60/127 seconds per model, before
unmeasured overhead. Saving one selected request saves five such forwards;
saving a few output tokens barely changes the dominant prompt work.

**Default operating cap: 120 GPU minutes per session.** The core remains
30 requests regardless of this cap. Do not fall back to ten or fifteen, reroll
for shorter requests, or truncate. Prepare all 150 jobs, estimate required
sessions from the exact token budget, and resume the same manifest until done.

Re-estimate after the first completed jobs. Stop launching work that cannot
finish and persist within the cap. Do not silently replace expensive rows
after observing them; resume their remaining jobs next session. Only a fully
matched core supports the planned five-model decision. Partial comparisons
must name their common rows and remain provisional. The 120-minute cap limits
spend; it does not guarantee completing the larger panel on slow hardware.

### Memory

Historical full replay peaks were about 86–88.1 GB with PLE tables on the host;
the recorded Kaggle machine had 176 GB host RAM. Verify available GPU/host
memory and page-cache behavior. Do not interpret 96 GB GPU memory as the only
resource requirement. Full-context inference caches remain live within each
request, while gradients and optimizer states are absent.

The existing capacity plan estimates resident device weights for **physically
pruned W4A16 exports** as follows; these are not whole-run peaks:

| Experts | Estimated device weights |
|---:|---:|
| 512 | 69.9 GB |
| 448 | 62.5 GB |
| 384 | 55.1 GB |
| 320 | 47.7 GB |
| 256 | 40.4 GB |

Use these only to illustrate the quality/memory tradeoff. The masked sweep
retains the full bank for every candidate. Training adds activations, adapters,
gradients and optimizer memory, and an NF4 implementation may have different
weight placement; none of these numbers establishes training capacity.

At vocabulary size 248,320, a 256-position FP32 logits block is **242.5 MiB**;
its BF16 precursor adds 121.25 MiB while live, plus CE/kernel temporaries.
At 512 positions those numbers double. Therefore start at 256 and discard
each block after saving its loss vector. Never retain `[context, vocabulary]`
logits, attention tensors or per-token hidden states.

### Logging is cheap

Store shared token IDs, target positions, span labels and display offsets
once per request. Store one FP32 NLL vector per `(request, model)` in NPZ or
another compact typed format; keep request/model metadata in JSONL. Derive
delta vectors offline. Do not repeat complete transcripts in every token row.

For 30 requests × 280 target tokens × 5 models, NLL values occupy **168 KB**
uncompressed. Even if all 30 replies were 4,300 tokens, they occupy **2.58 MB**.
Shared token annotations and reports add storage. Input/image bundles
and model weights dominate disk use. Saving per-token CE already computed by
the replay should add negligible arithmetic; measure transfer/I/O time rather
than inventing a precise percentage overhead.

## 8. Resume, artifacts and implementation boundary

One completion unit is a whole `(request, expert map)` replay. Write a temporary
result, flush/fsync, then atomically rename and mark complete only after finite
losses, target counts and hashes pass. Interrupted requests restart from their
beginning; do not serialize giant caches. At the historical longest input,
this typically costs minutes rather than losing an entire sweep. No model
checkpoint is needed because weights never change.

Bind resume identity to dataset/fold/sample hashes, model and map hashes,
quantization, processor/template, scoring code, kernels and chunk settings.
Persist a job ledger with pending/complete/error statuses; skip only exact
matching completed jobs. Rotate model order per request and interleave games
so interruption does not systematically leave only one model or game scored.

Launch and monitor the worker through the Kaggle Jupyter server, with a process
lock so reconnecting the notebook cannot duplicate it. Write a heartbeat with
current request/model, elapsed time, completed job count, ETA and deadline.
Local `/kaggle/working` files alone do **not** survive session loss. The external
Jupyter client downloads and verifies the small result/ledger bundle after
each completed request's model group (or every five minutes, whichever occurs
first), plus at shutdown. Resume by uploading the last verified bundle to the
next session. A Colab variant can persist to mounted Drive with the same
checksums, but requires separately qualified hardware/loading.

Proposed outputs:

    manifest.json              # immutable inputs, panel, reserve, weights, budget
    samples/<id>.json/npz       # shared target tokens/spans and request metadata
    results/<model>/<id>.npz    # token NLL, positions, checksum
    jobs.jsonl                 # append-only status/timing/error records
    comparison.csv/json/html   # CPU-generated summaries and token viewer
    selection.json             # chosen map, NLL tradeoff, caveats, capacity status

Implementation should add a small sampled-SFT entry point around the existing
replay, a CPU manifest/span builder and a CPU report generator. Keep old REAP
trace semantics intact. Acceptance checks cover final-only masking, one-token
shift, image alignment, span partitioning, no truncation, paired weighting,
chunk parity, map parity and interrupted-result recovery. These checks prevent
spending the scarce run on a wrong comparison; they are not extra benchmarks.

The CPU/GPU entry points and recovery/reporting code are now implemented in
[nll/](nll/README.md). Actual sampled IDs and the real-data preflight require
network access to the uncached DVC and Hugging Face inputs. Measured sol NLLs,
the chosen model and production runtime remain pending; no GPU has started.
