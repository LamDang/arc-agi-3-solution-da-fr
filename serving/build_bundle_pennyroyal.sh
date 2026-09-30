#!/usr/bin/env bash
# Pennyroyal v2.5.3 wheelhouse for OFFLINE Kaggle, built ONLINE in the matching
# Kaggle Docker image. No GPU is needed for this builder. Model weights,
# tokenizer assets and the existing adapted INT4 draft are separate datasets.
#
# Includes the three INT4 launcher patches, v2.2 prefill/unlock, and bounded prefetch:
#   1. Gabriel 0004: SM120 low-M BF16 GEMM (ported to Pennyroyal).
#   2. 0002 budget correction: do not reserve absent intermediate SSM buffers.
#   3. GPTQ/Marlin MoE scales: use params_dtype, fixing BF16/FP16 mismatch.
#   4. Sparse prefill v2.2: one cold-start branch + final checkpoint; MRU on unlock.
#   5. One-shard prefetch lookahead, parallel byte-range readers; default on.
#
# Example (use the same image/tag as the target Kaggle notebook):
#   mkdir -p out-penny
#   cp build_bundle_pennyroyal.sh out-penny/
#   docker run --name penny-builder -it -v "$PWD/out-penny:/out" \
#     gcr.io/kaggle-gpu-images/python:latest bash /out/build_bundle_pennyroyal.sh
# Output: /out/bundle-pennyroyal. Upload that directory as a Kaggle dataset.
# Keep the container to retain build/download caches if the build fails.
# Output is only published after a fresh offline installation passes.
#
# Controls: OUT=/out WORK=/build/pennyroyal PYTHON=python BUILD_JOBS=4
# Resume after SGLang built successfully (no SGLang rebuild):
#   RESUME_STAGE=/out/.penny-bundle.XXXXXXXX bash build_bundle_pennyroyal.sh
# Resume only stages already built with v2.5.3, prefill v2.2, and lookahead. An older SGLang
# wheel must be rebuilt. Upgrading from v2.5.2 also needs a new WORK checkout.
# No moving source branches, optional patch skips or FlashInfer overrides.
set -euo pipefail
readonly PENNY_SHA=d00d88efc8d6281b12be4f4073126aec95038c55
readonly PENNY_REPO=https://github.com/jpezzulli/sglang-rtxpro6000.git
readonly SGL_VERSION=0.5.19+gd00d88efc8d6
OUT="${OUT:-/out}"
WORK="${WORK:-/build/pennyroyal}"
PYTHON="${PYTHON:-python}"
BUILD_JOBS="${BUILD_JOBS:-4}"
RESUME_STAGE="${RESUME_STAGE:-}"
# Slow package indexes should not discard an otherwise successful build.
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}" UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-10}"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-300}" PIP_RETRIES="${PIP_RETRIES:-10}"
export UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy
# Every uv invocation selects its interpreter explicitly. An inherited
# UV_SYSTEM_PYTHON otherwise produces an irrelevant warning from uv venv.
unset UV_SYSTEM_PYTHON
fail() { echo "ERROR: $*" >&2; exit 1; }
[[ "$BUILD_JOBS" =~ ^[1-9][0-9]*$ ]] || fail 'BUILD_JOBS must be positive'
for cmd in "$PYTHON" git curl gcc g++ make pkg-config; do
  command -v "$cmd" >/dev/null || fail "Missing build prerequisite: $cmd"
done
PYTHON="$(command -v "$PYTHON")"
"$PYTHON" -S - <<'PY_PLATFORM'
import platform, sys
assert sys.platform == 'linux' and platform.machine() == 'x86_64', 'Use the Linux x86_64 Kaggle GPU image'
assert platform.python_implementation() == 'CPython' and sys.version_info >= (3, 10), 'CPython >=3.10 required'
print('Build interpreter:', sys.version)
print('Build platform:', platform.platform())
PY_PLATFORM
mkdir -p "$OUT" "$WORK"
OUT="$(cd "$OUT" && pwd)"
WORK="$(cd "$WORK" && pwd)"
BUNDLE="$OUT/bundle-pennyroyal"
[[ ! -e "$BUNDLE" ]] || fail "Output already exists: $BUNDLE; select a new OUT directory"
MODE=build
if [[ -n "$RESUME_STAGE" ]]; then
  [[ -d "$RESUME_STAGE" ]] || fail "Missing resume directory: $RESUME_STAGE"
  STAGE="$(cd "$RESUME_STAGE" && pwd)"
  [[ "$(dirname "$STAGE")" == "$OUT" && "$(basename "$STAGE")" == .penny-bundle.* ]] \
    || fail 'RESUME_STAGE must be a .penny-bundle.* directory immediately inside OUT'
  [[ -d "$STAGE/wheels" ]] || fail 'Resume wheel directory missing'
  MODE=finalize
  # These assets were written only after pip wheel completed, including in
  # older builders. Incomplete resolution/downloads need the online stage.
  for required in requirements.lock packages.json constraints.txt hot_tokens_64k.pt tracked-source.patch configs/upstream-build-env.sh; do
    [[ -s "$STAGE/$required" ]] || MODE=resolve
  done
else
  STAGE="$(mktemp -d "$OUT/.penny-bundle.XXXXXXXX")"
fi
SRC="$WORK/source"
mkdir -p "$STAGE/wheels" "$STAGE/patches" "$STAGE/configs"
trap 'echo "Build failed at line $LINENO; partial bundle retained at $STAGE" >&2' ERR
exec > >(tee -a "$STAGE/build.log") 2>&1
export PYTHONNOUSERSITE=1
unset PYTHONPATH
echo "Building Pennyroyal $PENNY_SHA with five patches. Staging: $STAGE"
echo "  1. SM120 low-M BF16 GEMM"
echo "  2. Intermediate SSM KV-budget correction"
echo "  3. GPTQ/Marlin MoE params_dtype scale fix"
echo "  4. Sparse prefill v2.2 and Mamba MRU refresh on unlock"
echo "  5. One-shard prefetch lookahead with parallel range readers (default on)"

# ---- exact embedded, hash-guarded patch ports and offline tools ----
cat > "$STAGE/patches/01-lowm.py" <<'PENNY_EMBEDDED_FILE_0'
import ast
import hashlib
import pathlib
import subprocess
import sys

PIN = 'd00d88efc8d6281b12be4f4073126aec95038c55'
BEFORE = {'python/sglang/kernels/ops/gemm/sm120_lowm_bf16_gemm.py': None, 'python/sglang/srt/environ.py': '84755bded97394a18537d8e329c3d05c15f7ec4aa8a801bd9971f34b4f11d7ed', 'python/sglang/srt/layers/quantization/unquant.py': 'fc7d7d24c939d08b0da8165767f4cbb96ed4741cbaa485442f45b84429d136ee'}
AFTER = {'python/sglang/kernels/ops/gemm/sm120_lowm_bf16_gemm.py': '4998de2a4a1309ef430444097f076f399865b41b56f4ce4b0ec337b81f082642', 'python/sglang/srt/environ.py': '0b98ad866be4da732833ef38d185eb8cb9e41329fe65caa82568c72810c94449', 'python/sglang/srt/layers/quantization/unquant.py': '28ce96e0baa7879bdeccc019946a99392b1db90a345612c00dd33f9c8d634a0d'}
PATCH = 'diff --git a/python/sglang/kernels/ops/gemm/sm120_lowm_bf16_gemm.py b/python/sglang/kernels/ops/gemm/sm120_lowm_bf16_gemm.py\nnew file mode 100644\n--- /dev/null\n+++ b/python/sglang/kernels/ops/gemm/sm120_lowm_bf16_gemm.py\n@@ -0,0 +1,181 @@\n+# SPDX-License-Identifier: Apache-2.0\n+"""SM120 (RTX PRO 6000 Blackwell / GeForce Blackwell) low-M BF16 GEMM.\n+\n+cuBLAS under CUDA-graph capture serves the qwen3.5/qwen4 decode projections\n+(m <= 8, k in {1536, 2560}) with 1-warp split-K WMMA kernels that reach only\n+~20-70% of DRAM bandwidth on sm120 (measured 61.9 us for m=4, n=4120, k=2560\n+= 320 GB/s on an RTX PRO 6000). The SM100 CuTeDSL/split-K paths in\n+``unquant.py`` need tcgen05 and cannot run here. This module provides a plain\n+Triton split-K GEMV tuned per shape for sm120, dispatched from\n+``bf16_gemm_dispatch``.\n+\n+Split-K partials are accumulated with fp32 atomics: non-deterministic\n+summation order, so this path is disabled under\n+``--enable-deterministic-inference`` (mirrors the SM100 split-K policy).\n+"""\n+\n+from __future__ import annotations\n+\n+from typing import Optional\n+\n+import torch\n+import triton\n+import triton.language as tl\n+\n+# (m, n, k) -> (split_k, block_n, block_k, num_warps, num_stages)\n+# Tuned on RTX PRO 6000 Blackwell (sm120, GDDR7 @ 13365 MHz) with HBM-cold\n+# weights and sequential (event-timed) launches; see qwen38fn/PERF_CEILING.md.\n+_SM120_TUNED_TACTICS: dict[tuple[int, int, int], tuple[int, int, int, int, int]] = {}\n+\n+# Shapes eligible for the generic fallback tactic even when not in the table.\n+# Covers decode (m = bs <= 4), bs=1 target-verify (m = 4-5) and bs<=4 C4\n+# verify (m <= 16), plus small eager draft-extend batches.\n+_MAX_M = 32\n+_MIN_WEIGHT_BYTES = 512 * 1024  # below this, launch overhead dominates; keep cuBLAS\n+# Above this, cuBLAS GEMV/split-K already streams at ~94% of DRAM bandwidth\n+# (measured on the 1.27 GB lm_head); keep it.\n+_MAX_WEIGHT_BYTES = 128 * 1024 * 1024\n+\n+\n+@triton.jit\n+def _sm120_lowm_gemm_kernel(\n+    x_ptr,\n+    w_ptr,\n+    out_ptr,\n+    M,\n+    N,\n+    K,\n+    stride_xm,\n+    stride_xk,\n+    stride_wn,\n+    stride_wk,\n+    stride_om,\n+    stride_on,\n+    BLOCK_N: tl.constexpr,\n+    BLOCK_K: tl.constexpr,\n+    BLOCK_M: tl.constexpr,\n+    SPLIT_K: tl.constexpr,\n+    NUM_STAGES: tl.constexpr,\n+    OUT_BF16: tl.constexpr,\n+):\n+    pid_n = tl.program_id(0)\n+    pid_k = tl.program_id(1)\n+    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)\n+    rm = tl.arange(0, BLOCK_M)\n+    n_mask = rn < N\n+    m_mask = rm < M\n+    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)\n+    k_per = tl.cdiv(K, SPLIT_K)\n+    k_start = pid_k * k_per\n+    k_end = tl.minimum(k_start + k_per, K)\n+    for k0 in tl.range(k_start, k_end, BLOCK_K, num_stages=NUM_STAGES):\n+        rk = k0 + tl.arange(0, BLOCK_K)\n+        k_mask = rk < k_end\n+        xb = tl.load(\n+            x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk,\n+            mask=m_mask[:, None] & k_mask[None, :],\n+            other=0.0,\n+        )\n+        wb = tl.load(\n+            w_ptr + rn[:, None] * stride_wn + rk[None, :] * stride_wk,\n+            mask=n_mask[:, None] & k_mask[None, :],\n+            other=0.0,\n+        )\n+        acc += tl.dot(xb, tl.trans(wb), out_dtype=tl.float32)\n+    if OUT_BF16:\n+        tl.store(\n+            out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on,\n+            acc.to(tl.bfloat16),\n+            mask=m_mask[:, None] & n_mask[None, :],\n+        )\n+    else:\n+        tl.atomic_add(\n+            out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on,\n+            acc,\n+            mask=m_mask[:, None] & n_mask[None, :],\n+            sem="relaxed",\n+        )\n+\n+\n+def _generic_tactic(m: int, n: int, k: int) -> tuple[int, int, int, int, int]:\n+    """(1, 16, 128) measured best for the wide shapes (81% of DRAM bandwidth\n+    at m=4, n=4120, k=2560); split K only when 16-wide N tiles cannot fill\n+    the 188 sm120 SMs."""\n+    n_ctas = triton.cdiv(n, 16)\n+    split_k = 1\n+    while split_k < 16 and n_ctas * split_k < 128 and (k // (split_k * 2)) >= 256:\n+        split_k *= 2\n+    return (split_k, 16, 128, 4, 4)\n+\n+\n+def use_sm120_lowm_bf16_gemm(m: int, n: int, k: int) -> bool:\n+    return m <= _MAX_M and _MIN_WEIGHT_BYTES <= n * k * 2 <= _MAX_WEIGHT_BYTES\n+\n+\n+def sm120_lowm_bf16_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:\n+    """out = x @ weight.T for bf16 x[m, k], weight[n, k]; bias-free, m <= 8."""\n+    x_2d = x.view(-1, x.shape[-1])\n+    m, k = x_2d.shape\n+    n = weight.shape[0]\n+    tactic = _SM120_TUNED_TACTICS.get((m, n, k)) or _generic_tactic(m, n, k)\n+    split_k, block_n, block_k, num_warps, num_stages = tactic\n+    block_m = max(16, triton.next_power_of_2(m))\n+    grid = (triton.cdiv(n, block_n), split_k)\n+    if split_k == 1:\n+        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)\n+        _sm120_lowm_gemm_kernel[grid](\n+            x_2d,\n+            weight,\n+            out,\n+            m,\n+            n,\n+            k,\n+            x_2d.stride(0),\n+            x_2d.stride(1),\n+            weight.stride(0),\n+            weight.stride(1),\n+            out.stride(0),\n+            out.stride(1),\n+            BLOCK_N=block_n,\n+            BLOCK_K=block_k,\n+            BLOCK_M=block_m,\n+            SPLIT_K=1,\n+            NUM_STAGES=num_stages,\n+            OUT_BF16=True,\n+            num_warps=num_warps,\n+        )\n+    else:\n+        acc = torch.zeros((m, n), dtype=torch.float32, device=x.device)\n+        _sm120_lowm_gemm_kernel[grid](\n+            x_2d,\n+            weight,\n+            acc,\n+            m,\n+            n,\n+            k,\n+            x_2d.stride(0),\n+            x_2d.stride(1),\n+            weight.stride(0),\n+            weight.stride(1),\n+            acc.stride(0),\n+            acc.stride(1),\n+            BLOCK_N=block_n,\n+            BLOCK_K=block_k,\n+            BLOCK_M=block_m,\n+            SPLIT_K=split_k,\n+            NUM_STAGES=num_stages,\n+            OUT_BF16=False,\n+            num_warps=num_warps,\n+        )\n+        out = acc.to(torch.bfloat16)\n+    return out.view(*x.shape[:-1], n)\n+\n+\n+def precompile_sm120_lowm_tactics(device: Optional[torch.device] = None) -> None:\n+    """JIT-compile the tuned specializations before CUDA graph capture."""\n+    for m, n, k in _SM120_TUNED_TACTICS:\n+        x = torch.zeros(m, k, dtype=torch.bfloat16, device=device or "cuda")\n+        w = torch.zeros(n, k, dtype=torch.bfloat16, device=device or "cuda")\n+        sm120_lowm_bf16_gemm(x, w)\n+    # generic-tactic block sizes used by unlisted shapes\n+    torch.cuda.synchronize()\ndiff --git a/python/sglang/srt/environ.py b/python/sglang/srt/environ.py\n--- a/python/sglang/srt/environ.py\n+++ b/python/sglang/srt/environ.py\n@@ -1012,6 +1012,9 @@\n     # Enable the allowlisted low-M BF16 Split-K GEMM path on Blackwell. Shapes\n     # outside the measured allowlist continue to use CuTe DSL/cuBLAS.\n     SGLANG_ENABLE_BF16_SPLITK_GEMM = EnvBool(True)\n+    # Gabriel Olympie\'s patch 0004: small BF16 GEMMs on exact SM120.\n+    # Disabled under deterministic inference; does not quantize weights.\n+    SGLANG_ENABLE_SM120_LOWM_BF16_GEMM = EnvBool(True)\n     # Opt-in Flash-Next-only conversion of eligible BF16 projections on exact\n     # SM120. Large linears use MXFP8; lm_head and HyperConnection mix weights\n     # use rowwise weight-only FP8. An explicit unsupported request fails boot.\ndiff --git a/python/sglang/srt/layers/quantization/unquant.py b/python/sglang/srt/layers/quantization/unquant.py\n--- a/python/sglang/srt/layers/quantization/unquant.py\n+++ b/python/sglang/srt/layers/quantization/unquant.py\n@@ -96,6 +96,9 @@\n _flashinfer_pr4266_run_splitk_dense = None\n _flashinfer_pr4266_splitk_tactic = None\n _enable_bf16_splitk_gemm = False\n+_sm120_lowm_bf16_gemm = None\n+_use_sm120_lowm_bf16_gemm = None\n+_enable_sm120_lowm_bf16_gemm = False\n _logged_bf16_gemm_shapes = set()\n \n # Qwen4-Exp TP4 decode tactics measured on B300 (sm103) under CUDA graph\n@@ -160,6 +163,8 @@\n     global _BF16_GEMM_BACKEND, _cutedsl_bf16_gemm, _use_cutedsl_bf16_gemm\n     global _flashinfer_pr4266_run_splitk_dense, _flashinfer_pr4266_splitk_tactic\n     global _enable_bf16_splitk_gemm\n+    global _sm120_lowm_bf16_gemm, _use_sm120_lowm_bf16_gemm\n+    global _enable_sm120_lowm_bf16_gemm\n \n     from sglang.srt.utils import is_sm100_supported\n \n@@ -234,6 +239,23 @@\n             "Flash-Next online FP8 enabled on SM120: eligible BF16 linears use "\n             "MXFP8; HyperConnection mix and lm_head use rowwise FP8"\n         )\n+\n+    _enable_sm120_lowm_bf16_gemm = False\n+    if (\n+        envs.SGLANG_ENABLE_SM120_LOWM_BF16_GEMM.get()\n+        and _is_cuda\n+        and torch.cuda.get_device_capability() == (12, 0)\n+        and not server_args.enable_deterministic_inference\n+    ):\n+        from sglang.kernels.ops.gemm.sm120_lowm_bf16_gemm import (\n+            sm120_lowm_bf16_gemm,\n+            use_sm120_lowm_bf16_gemm,\n+        )\n+\n+        _sm120_lowm_bf16_gemm = sm120_lowm_bf16_gemm\n+        _use_sm120_lowm_bf16_gemm = use_sm120_lowm_bf16_gemm\n+        _enable_sm120_lowm_bf16_gemm = True\n+        logger.info("SM120 low-M BF16 GEMM path enabled")\n \n     _BF16_GEMM_BACKEND = backend\n \n@@ -290,6 +312,12 @@\n         and use_flashinfer_pr4266_bf16_gemm(m, weight.shape[0], weight.shape[1])\n     ):\n         return _flashinfer_pr4266_bf16_gemm(x, weight)\n+    if (\n+        _enable_sm120_lowm_bf16_gemm\n+        and bias is None\n+        and _use_sm120_lowm_bf16_gemm(m, weight.shape[0], weight.shape[1])\n+    ):\n+        return _sm120_lowm_bf16_gemm(x, weight)\n     if _use_cutedsl_bf16_gemm is not None and _use_cutedsl_bf16_gemm(\n         m, weight.shape[0], weight.shape[1]\n     ):\n@@ -434,6 +462,23 @@\n                 return output.view(*x_shapes[:-1], -1)\n             return F.linear(x, layer.weight, bias)\n \n+        elif (\n+            _enable_sm120_lowm_bf16_gemm\n+            and bias is None\n+            and x.is_cuda\n+            and x.dtype == torch.bfloat16\n+            and layer.weight.dtype == torch.bfloat16\n+            and not layer.weight.requires_grad\n+        ):\n+            m = x.numel() // x.shape[-1]\n+            if envs.SGLANG_BF16_GEMM_LOG_SHAPES.get():\n+                _log_bf16_gemm_shape(m, layer.weight.shape[0], layer.weight.shape[1])\n+            if _use_sm120_lowm_bf16_gemm(\n+                m, layer.weight.shape[0], layer.weight.shape[1]\n+            ):\n+                return _sm120_lowm_bf16_gemm(x, layer.weight)\n+            return F.linear(x, layer.weight, bias)\n+\n         return F.linear(x, layer.weight, bias)\n \n     def apply_into(\n'

def fail(message):
    raise SystemExit('ERROR: ' + message)

def main():
    root = pathlib.Path(sys.argv[1]).expanduser().resolve()
    mode = sys.argv[2]
    if mode not in ('apply', 'check', 'revert'):
        fail('mode must be apply, check, or revert')
    def git(*args, **kwargs):
        return subprocess.run(['git', '-C', str(root), *args], text=True,
                              capture_output=True, **kwargs)
    top = git('rev-parse', '--show-toplevel')
    if top.returncode or pathlib.Path(top.stdout.strip()).resolve() != root:
        fail('ROOT must be the root of a Git checkout')
    rev = git('rev-parse', 'HEAD')
    if rev.returncode or rev.stdout.strip() != PIN:
        fail('expected Pennyroyal v2.5.3 at ' + PIN + '; found ' + rev.stdout.strip())
    def hashes():
        result = {}
        for name in BEFORE:
            p = root / name
            # Refuse symlinks anywhere in the patched path.
            if any(q.is_symlink() for q in (p, *p.parents) if q != root.parent):
                fail('symlink in target path: ' + name)
            if p.exists() and not p.is_file():
                fail('target is not a regular file: ' + name)
            result[name] = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
        return result
    state = hashes()
    if state == AFTER:
        if mode != 'revert':
            print('Already applied: all three file hashes match this port.')
            return
        reverse, expected = True, BEFORE
    elif state == BEFORE:
        if mode == 'revert':
            print('Already unpatched: no changes needed.')
            return
        reverse, expected = False, AFTER
    else:
        changed = [n for n in state if state[n] != BEFORE[n] and state[n] != AFTER[n]]
        fail('target files differ or port is partially applied; nothing changed. ' + ', '.join(changed))
    flags = ['--reverse'] if reverse else []
    checked = git('apply', '--check', *flags, '-', input=PATCH)
    if checked.returncode:
        fail('patch check failed; nothing changed.\n' + checked.stderr)
    if mode == 'check':
        print('Compatible: pinned release, pristine target files, patch applies cleanly.')
        return
    result = git('apply', *flags, '-', input=PATCH)
    if result.returncode:
        fail('git apply failed: ' + result.stderr)
    if hashes() != expected:
        fail('post-write hash verification failed; inspect checkout before serving')
    for name, digest in expected.items():
        if digest is not None:
            ast.parse((root / name).read_text(), filename=name)
    print('Reverted successfully.' if reverse else 'Applied successfully; Python syntax and all file hashes verified.')
    if not reverse:
        print('Restart the server using this source checkout. No auto-commit was made.')
        print('Startup marker: SM120 low-M BF16 GEMM path enabled')
        print('A/B disable: SGLANG_ENABLE_SM120_LOWM_BF16_GEMM=0 (restart required).')

if __name__ == '__main__':
    main()
PENNY_EMBEDDED_FILE_0

cat > "$STAGE/patches/02-kv-budget.py" <<'PENNY_EMBEDDED_FILE_1'
import ast
import hashlib
import pathlib
import subprocess
import sys
PIN = 'd00d88efc8d6281b12be4f4073126aec95038c55'
BEFORE = {'python/sglang/srt/mem_cache/kv_cache_configurator.py': 'a759ae971d6da987cf11bf81de85d6f67d577fc487bb52bf241c4019a825a138'}
AFTER = {'python/sglang/srt/mem_cache/kv_cache_configurator.py': '1751fd5464932ce1eadbdca19f069bbdc354b429432b43aabfc3e1fcb776dae7'}
PATCH = 'diff --git a/python/sglang/srt/mem_cache/kv_cache_configurator.py b/python/sglang/srt/mem_cache/kv_cache_configurator.py\n--- a/python/sglang/srt/mem_cache/kv_cache_configurator.py\n+++ b/python/sglang/srt/mem_cache/kv_cache_configurator.py\n@@ -2109,6 +2109,11 @@\n             self.hybrid_gdn_config is not None\n             or kimi_linear_config(self.model_config) is not None\n         )\n+        # RecoverSSM also skips intermediate_ssm allocation in MambaPool.\n+        # Do not charge the KV budget for this absent scratch buffer.\n+        no_intermediate_ssm = replayssm_active or (\n+            get_exec().mamba.gdn_mtp_cache_mode == "none"\n+        )\n         if replayssm_active:\n             # GDN sizes the fold window to the draft maximum; the KDA ring\n             # stays --linear-replayssm-cache-len long (mirrors MambaPool).\n@@ -2139,9 +2144,9 @@\n                 // self.ps.attn_dp_size,\n             )\n             # Reserve intermediate memory based on capped max_num_reqs (+1: the\n-            # pool\'s padding slot, see memory_pool.py). Skipped under replayssm\n-            # (no intermediate_ssm allocated).\n-            if has_spec_dec and not replayssm_active:\n+            # pool\'s padding slot, see memory_pool.py). Skipped for ReplaySSM\n+            # and RecoverSSM (no intermediate_ssm allocated).\n+            if has_spec_dec and not no_intermediate_ssm:\n                 ratio = self._calculate_mamba_ratio()\n                 capped_reqs = min(\n                     get_schedule().max_running_requests // self.ps.attn_dp_size,\n@@ -2164,8 +2169,8 @@\n                 // self.ps.attn_dp_size,\n             )\n             # Reserve intermediate memory based on capped max_num_reqs (+1: the\n-            # pool\'s padding slot). Skipped under replayssm.\n-            if has_spec_dec and not replayssm_active:\n+            # pool\'s padding slot). Skipped for ReplaySSM and RecoverSSM.\n+            if has_spec_dec and not no_intermediate_ssm:\n                 intermediate_size = (\n                     stage_per_req\n                     * (get_schedule().max_mamba_cache_size + 1)\n@@ -2187,7 +2192,7 @@\n             )\n             mamba_budget_bytes = mamba_budget * (1 << 30)\n \n-            if has_spec_dec and not replayssm_active:\n+            if has_spec_dec and not no_intermediate_ssm:\n                 ratio = self._calculate_mamba_ratio()\n                 D = get_spec().speculative_num_draft_tokens\n                 # Joint solve: main_state + intermediate = mamba_budget\n'

def fail(message):
    raise SystemExit('ERROR: ' + message)

def main():
    root = pathlib.Path(sys.argv[1]).expanduser().resolve()
    mode = sys.argv[2]
    if mode not in ('apply', 'check', 'revert'):
        fail('mode must be apply, check, or revert')

    def git(*args, **kwargs):
        return subprocess.run(['git', '-C', str(root), *args], text=True, capture_output=True, **kwargs)
    top = git('rev-parse', '--show-toplevel')
    if top.returncode or pathlib.Path(top.stdout.strip()).resolve() != root:
        fail('ROOT must be the root of a Git checkout')
    rev = git('rev-parse', 'HEAD')
    if rev.returncode or rev.stdout.strip() != PIN:
        fail('expected Pennyroyal v2.5.3 at ' + PIN + '; found ' + rev.stdout.strip())

    def hashes():
        result = {}
        for name in BEFORE:
            p = root / name
            if any((q.is_symlink() for q in (p, *p.parents) if q != root.parent)):
                fail('symlink in target path: ' + name)
            if p.exists() and (not p.is_file()):
                fail('target is not a regular file: ' + name)
            result[name] = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
        return result
    state = hashes()
    if state == AFTER:
        if mode != 'revert':
            print('Already applied: KV budget file hash matches this port.')
            return
        reverse, expected = (True, BEFORE)
    elif state == BEFORE:
        if mode == 'revert':
            print('Already unpatched: no changes needed.')
            return
        reverse, expected = (False, AFTER)
    else:
        changed = [n for n in state if state[n] != BEFORE[n] and state[n] != AFTER[n]]
        fail('target files differ or port is partially applied; nothing changed. ' + ', '.join(changed))
    flags = ['--reverse'] if reverse else []
    checked = git('apply', '--check', *flags, '-', input=PATCH)
    if checked.returncode:
        fail('patch check failed; nothing changed.\n' + checked.stderr)
    if mode == 'check':
        print('Compatible: pinned release, pristine target files, patch applies cleanly.')
        return
    result = git('apply', *flags, '-', input=PATCH)
    if result.returncode:
        fail('git apply failed: ' + result.stderr)
    if hashes() != expected:
        fail('post-write hash verification failed; inspect checkout before serving')
    for name, digest in expected.items():
        if digest is not None:
            ast.parse((root / name).read_text(), filename=name)
    print('Reverted successfully.' if reverse else 'Applied successfully; Python syntax and all file hashes verified.')
    if not reverse:
        print('Restart the server using this source checkout. No auto-commit was made.')
        print('Active after restart with --gdn-mtp-cache-mode none; compare max_total_num_tokens.')
        print('A/B undo: run this script with revert, then restart with the same settings.')
if __name__ == '__main__':
    main()
PENNY_EMBEDDED_FILE_1

cat > "$STAGE/patches/03-marlin-dtype.py" <<'PENNY_EMBEDDED_FILE_2'
"""Pinned, idempotent fix for GPTQ Marlin MoE scale allocation dtype."""
import hashlib
import os
import pathlib
import subprocess
import sys

PIN = 'd00d88efc8d6281b12be4f4073126aec95038c55'
BEFORE_SHA = '135e004f7613a59aa0765cbe3d779c9bb0c459f3a199b868358759e020aaeb5d'
AFTER_SHA = '47e65140e2480a8637b9ea5bb746ff77e9c09c01e448fd2056c0b485e4f3a10a'
REL = 'python/sglang/srt/layers/quantization/gptq/schemes/gptq_moe.py'


def main():
    root = pathlib.Path(sys.argv[1]).resolve()
    mode = sys.argv[2] if len(sys.argv) > 2 else 'apply'
    assert mode in ('apply', 'check'), 'Unknown patch action'
    pin = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    assert pin == PIN, 'GPTQ scale patch requires the pinned Pennyroyal v2.5.3 revision'
    path = root / REL
    b = path.read_bytes()
    digest = hashlib.sha256(b).hexdigest()
    if digest == AFTER_SHA:
        print('GPTQ Marlin MoE scale dtype fix: already applied (params_dtype).')
        return
    assert mode == 'apply', 'GPTQ scale dtype patch is not applied; run serve/setup'
    assert digest == BEFORE_SHA, 'GPTQ source differs from reviewed revision; refusing to overwrite'
    assert b.count(b'dtype=torch.half') == 2, 'Unexpected scale allocation count'
    patched = b.replace(b'dtype=torch.half', b'dtype=params_dtype')
    assert hashlib.sha256(patched).hexdigest() == AFTER_SHA, 'Unexpected patch result'
    temp = path.with_name(path.name + '.marlin-dtype.tmp')
    temp.write_bytes(patched)
    temp.chmod(path.stat().st_mode & 0o777)
    os.replace(temp, path)
    print('Applied GPTQ Marlin MoE scale dtype fix: scales follow params_dtype at load time.')


if __name__ == '__main__':
    try:
        main()
    except (AssertionError, OSError, subprocess.CalledProcessError) as e:
        sys.exit('GPTQ Marlin scale patch: ' + str(e))
PENNY_EMBEDDED_FILE_2

# Retire the previous generated helper names when reusing a staging directory.
rm -f "$STAGE/patches/04-prefill-v21.py" "$STAGE/patches/04-prefill-v21.patch"
cat > "$STAGE/patches/04-prefill-v22.py" <<'PENNY_EMBEDDED_FILE_3'
#!/usr/bin/env python3
"""Offline, reversible patch for Pennyroyal d00d88e: one cold-start branch plus the prefill endpoint (v2.2, low-M compatible).

Stop SGLang, run this with its checkout path, then restart the existing launcher.
No wheel/kernel rebuild is required for an editable source installation.
v2.2 refreshes a Mamba checkpoint to MRU immediately before its final device
lock is released. This includes the prefill endpoint held throughout decode;
host unlocks, skipped locks, ancestors, and shared locks still held are unchanged.
Set SGLANG_MAMBA_REFRESH_ON_UNLOCK=0 to disable the new LRU policy independently.
Set both SGLANG_MAMBA_PREFILL_FINAL_ONLY=0 and SGLANG_MAMBA_REFRESH_ON_UNLOCK=0
before launch for the original runtime behavior. Both options default to enabled.
Requires pinned Pennyroyal v2.5.3 source; --revert restores that baseline.
Validated on source/control-flow tests only; GPU integration testing is pending.
"""
import argparse
import ast
import hashlib
import os
from pathlib import Path
import tempfile
import subprocess

PIN = 'd00d88efc8d6281b12be4f4073126aec95038c55'

BEFORE = {'python/sglang/srt/mem_cache/unified_radix_cache.py': '643bb8e87b36ceb5a4398d222c91cd9e8be61c46aabae24da7ed96aa087ab83b', 'python/sglang/srt/managers/schedule_batch.py': '034c642d5139e645030884bd816ccaff095dfd4924d5729dd65267a903a7421c', 'python/sglang/srt/managers/schedule_policy.py': '64f3ea4ab0ac071d56639e407abd36ea392a656c2966ed92e6614dfa85cc02e8', 'python/sglang/srt/mem_cache/unified_cache/components/mamba_component.py': '6258302c506732d0357bcaa5dba7209131a23cc0bec68e1bb11ba22028db017b'}
AFTER = {'python/sglang/srt/mem_cache/unified_radix_cache.py': '9201c89377622efa281c9a9841e68bd96c31c110fee476331c7218749a026659', 'python/sglang/srt/managers/schedule_batch.py': '4eb9cdf4bd7064ce70ec0f726e670a4006a80befe1317adaa875e90e08cc03e5', 'python/sglang/srt/managers/schedule_policy.py': '2aacc1801884c9b0fb63cce26c2812de17ffe1977de2ed7dd52336c4139abc2d', 'python/sglang/srt/mem_cache/unified_cache/components/mamba_component.py': 'f529c0f966ff9bd69dcb27b79ae6b20d18eaa43860095b1585900f0771fb6a5d'}
RECORDS = [{'path': 'python/sglang/srt/mem_cache/unified_radix_cache.py', 'before': '643bb8e87b36ceb5a4398d222c91cd9e8be61c46aabae24da7ed96aa087ab83b', 'after': '9201c89377622efa281c9a9841e68bd96c31c110fee476331c7218749a026659', 'edits': [('import logging\nimport threading\n', 'import logging\nimport os\nimport threading\n'), ('        # SWA window size (None when SWA is not enabled).\n', '        # This policy is scoped to the regular FULL+MAMBA extra_buffer path.\n        self.mamba_prefill_final_only = (\n            os.environ.get("SGLANG_MAMBA_PREFILL_FINAL_ONLY", "1").lower()\n            not in ("0", "false")\n            and self.enable_mamba_extra_buffer\n            and not params.enable_mamba_extra_buffer_lazy\n            and set(self.tree_components) == {ComponentType.FULL, ComponentType.MAMBA}\n        )\n        if self.mamba_prefill_final_only:\n            logger.info("Mamba sparse-prefill v2: one cold-start branch plus final checkpoint")\n        # SWA window size (None when SWA is not enabled).\n'), ('        # components prepare insert data + return effective cache_len\n        insert_params = InsertParams(\n            prev_prefix_len=req.cache_protected_len,\n            chunked=chunked,\n', '        if chunked and self.mamba_prefill_final_only:\n            # Publish only the one branch chosen at the initial cold admission.\n            # The scheduler cuts a chunk exactly here, so the tracked state and\n            # the branch key agree. Never adopt later prefix-match hints.\n            branch = getattr(req, "mamba_prefill_branch_seqlen", None)\n            publish_branch = (\n                getattr(req, "mamba_prefill_initial_cached_tokens", None) == 0\n                and not getattr(req, "mamba_prefill_branch_used", False)\n                and branch is not None\n                and branch > 0\n                and len(token_ids) == branch\n                and req.mamba_last_track_seqlen == branch\n            )\n            if publish_branch:\n                # One attempt even if the pool cannot supply a donation slot.\n                req.mamba_prefill_branch_used = True\n            else:\n                # New KV remains request-owned; retain the existing tree lock.\n                req.prefix_indices = kv_indices_orig.to(dtype=torch.int64, copy=True)\n                # Preserve last_track_* for a final chunk shorter than the grid.\n                return\n\n        # components prepare insert data + return effective cache_len\n        insert_params = InsertParams(\n            prev_prefix_len=req.cache_protected_len,\n            chunked=chunked,\n'), ('        # SWA window size (None when SWA is not enabled).\n', '        # Refresh only the released Mamba state when its final device lock drops.\n        # Independent of sparse-prefill, including for extra_buffer_lazy.\n        self.mamba_refresh_on_unlock = (\n            self.is_mamba_enabled\n            and os.environ.get("SGLANG_MAMBA_REFRESH_ON_UNLOCK", "1").strip().lower()\n            not in ("0", "false", "off", "no")\n        )\n        if self.mamba_refresh_on_unlock:\n            logger.info("Mamba LRU: refresh checkpoint before final device unlock")\n        # SWA window size (None when SWA is not enabled).\n')], 'versions': []}, {'path': 'python/sglang/srt/managers/schedule_batch.py', 'before': '034c642d5139e645030884bd816ccaff095dfd4924d5729dd65267a903a7421c', 'after': '4eb9cdf4bd7064ce70ec0f726e670a4006a80befe1317adaa875e90e08cc03e5', 'edits': [('            if req.mamba_branching_seqlen is not None:\n', '            if req.mamba_branching_seqlen is not None and not getattr(\n                self.tree_cache, "mamba_prefill_final_only", False\n            ):\n'), ('        self.mamba_branching_seqlen: Optional[int] = None\n', '        self.mamba_branching_seqlen: Optional[int] = None\n        # Frozen on the first prefill forward, not on each chunk/rematch.\n        self.mamba_prefill_initial_cached_tokens: Optional[int] = None\n        self.mamba_prefill_branch_seqlen: Optional[int] = None\n        self.mamba_prefill_branch_used: bool = False\n'), ('        chunk_size = mamba_cache_chunk_size()\n', '        if (\n            getattr(self.tree_cache, "mamba_prefill_final_only", False)\n            and req.mamba_prefill_initial_cached_tokens is None\n        ):\n            req.mamba_prefill_initial_cached_tokens = len(req.prefix_indices)\n            req.mamba_prefill_branch_seqlen = (\n                req.mamba_branching_seqlen\n                if req.mamba_prefill_initial_cached_tokens == 0\n                else None\n            )\n        chunk_size = mamba_cache_chunk_size()\n')], 'versions': []}, {'path': 'python/sglang/srt/managers/schedule_policy.py', 'before': '64f3ea4ab0ac071d56639e407abd36ea392a656c2966ed92e6614dfa85cc02e8', 'after': '2aacc1801884c9b0fb63cce26c2812de17ffe1977de2ed7dd52336c4139abc2d', 'edits': [('    def add_chunked_req(self, req: Req):\n', '    def _cap_chunk_budget_at_mamba_branch(self, req: Req):\n        """End one cold prefill chunk at its initial shared-prefix divergence.\n\n        Cap the batch chunk budget too: consuming the branch chunk must stop\n        further admissions from creating a second unfinished chunked request.\n        """\n        if (\n            not getattr(self.tree_cache, "mamba_prefill_final_only", False)\n            or self.rem_chunk_tokens is None\n            or self.dllm_config is not None\n            or getattr(req, "mamba_prefill_branch_used", False)\n        ):\n            return\n        initial = getattr(req, "mamba_prefill_initial_cached_tokens", None)\n        if initial is None:\n            initial = len(req.prefix_indices)\n            branch = req.mamba_branching_seqlen\n        else:\n            branch = req.mamba_prefill_branch_seqlen\n        if initial != 0 or branch is None:\n            return\n        prefix_len = len(req.prefix_indices)\n        if prefix_len < branch < len(req.full_untruncated_fill_ids):\n            self.rem_chunk_tokens = min(self.rem_chunk_tokens, branch - prefix_len)\n\n    def add_chunked_req(self, req: Req):\n        self._cap_chunk_budget_at_mamba_branch(req)\n'), ('        # Reserve page_size for page-alignment overhead: the paged allocator may\n', '        self._cap_chunk_budget_at_mamba_branch(req)\n\n        # Reserve page_size for page-alignment overhead: the paged allocator may\n')], 'versions': []}, {'path': 'python/sglang/srt/mem_cache/unified_cache/components/mamba_component.py', 'before': '6258302c506732d0357bcaa5dba7209131a23cc0bec68e1bb11ba22028db017b', 'after': 'f529c0f966ff9bd69dcb27b79ae6b20d18eaa43860095b1585900f0771fb6a5d', 'edits': [('        if cd.lock_ref > 0:\n            if cd.lock_ref == 1:\n                vlen = len(value)\n', '        if cd.lock_ref > 0:\n            if cd.lock_ref == 1:\n                # A prefill checkpoint can stay locked for a long decode. Make\n                # its age reflect release, not the start of that decode, before\n                # it becomes evictable. Touch only this state, never ancestors.\n                if getattr(self.cache, "mamba_refresh_on_unlock", False):\n                    self.tree_core.lru_lists[ct].reset_node_mru(node)\n                vlen = len(value)\n')], 'versions': []}]

def digest(data):
    return hashlib.sha256(data).hexdigest()

# Normalize only this known, independent low-M addition for verification.
# The actual source text is preserved byte-for-byte outside our own edits.
LOWM_ENV_ADDITION = "    # Gabriel Olympie's patch 0004: small BF16 GEMMs on exact SM120.\n    # Disabled under deterministic inference; does not quantize weights.\n    SGLANG_ENABLE_SM120_LOWM_BF16_GEMM = EnvBool(True)\n"

def source_digest(record, source):
    if record['path'] == 'python/sglang/srt/environ.py':
        count = source.count(LOWM_ENV_ADDITION)
        if count > 1:
            raise RuntimeError('Duplicate low-M environment declaration. No files changed.')
        source = source.replace(LOWM_ENV_ADDITION, '', 1)
    return digest(source.encode('utf-8'))


def replace(path, data):
    mode = path.stat().st_mode & 0o777
    fd, name = tempfile.mkstemp(prefix=path.name + '.tmp-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        os.chmod(name, mode)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', default='/root/sglang-pennyroyal-intel', type=Path)
    parser.add_argument('--check', action='store_true', help='validate without changing files')
    parser.add_argument('--revert', action='store_true', help='restore the original source')
    args = parser.parse_args()
    root = args.root.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != PIN:
        raise RuntimeError("Expected pinned Pennyroyal v2.5.3 at " + PIN + "; found " + revision)
    pending = []
    for record in RECORDS:
        path = root / record['path']
        original = path.read_bytes()
        source = original.decode('utf-8')
        state = source_digest(record, source)
        wanted = record['before'] if args.revert else record['after']
        if state == wanted:
            print('Already in requested state:', record['path'])
            continue
        if state == record['after']:
            undo = record['edits']
        elif state == record['before']:
            undo = []
        else:
            known = next((v for v in record['versions'] if v['after'] == state), None)
            if known is None:
                raise RuntimeError(f'Source mismatch: {path}\nExpected pinned Pennyroyal d00d88e with the known low-M patch and/or this v2.2 patch. No files changed.')
            undo = known['edits']
        for before, after in reversed(undo):
            if source.count(after) != 1:
                raise RuntimeError(f'Ambiguous reverse anchor in {path}. No files changed.')
            source = source.replace(after, before, 1)
        if source_digest(record, source) != record['before']:
            raise RuntimeError(f'Cannot recover baseline for {path}. No files changed.')
        if not args.revert:
            for before, after in record['edits']:
                if source.count(before) != 1:
                    raise RuntimeError(f'Ambiguous patch anchor in {path}. No files changed.')
                source = source.replace(before, after, 1)
        updated = source.encode('utf-8')
        if source_digest(record, source) != wanted:
            raise RuntimeError(f'Unexpected patch result in {path}. No files changed.')
        ast.parse(source, filename=str(path))
        pending.append((path, original, updated))
    if args.check:
        print(f'Check passed: {len(pending)} file(s) would change. No files changed.')
        return
    completed = []
    try:
        for path, original, updated in pending:
            # Refuse to overwrite a concurrent edit after preflight.
            if path.read_bytes() != original:
                raise RuntimeError(f'Source changed during patch: {path}')
            replace(path, updated)
            completed.append((path, original))
    except Exception:
        for path, original in reversed(completed):
            replace(path, original)
        raise
    print(('Reverted' if args.revert else 'Applied') + f': {len(pending)} file(s).')
    print('Restart SGLang to load the change; no CUDA rebuild is needed for an editable install.')
    if not args.revert:
        print('Default v2.2: sparse prefill plus Mamba MRU refresh on final device unlock.')
        print('Disable sparse prefill: export SGLANG_MAMBA_PREFILL_FINAL_ONLY=0')
        print('Disable unlock refresh: export SGLANG_MAMBA_REFRESH_ON_UNLOCK=0')
        print('Startup marker: Mamba sparse-prefill v2: one cold-start branch plus final checkpoint')

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 3 and sys.argv[2] in ('apply', 'check', 'revert'):
        mode = sys.argv.pop()
        if mode != 'apply':
            sys.argv.append('--' + mode)
    try:
        main()
    except (OSError, RuntimeError, SyntaxError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error))
PENNY_EMBEDDED_FILE_3

cat > "$STAGE/patches/04-prefill-v22.patch" <<'PENNY_EMBEDDED_FILE_4'
--- a/python/sglang/srt/mem_cache/unified_radix_cache.py
+++ b/python/sglang/srt/mem_cache/unified_radix_cache.py
@@ -2,6 +2,7 @@
 
 import atexit
 import logging
+import os
 import threading
 import time
 from dataclasses import replace
@@ -184,6 +185,25 @@
         self.enable_mamba_extra_buffer = (
             params.enable_mamba_extra_buffer if self.is_mamba_enabled else False
         )
+        # This policy is scoped to the regular FULL+MAMBA extra_buffer path.
+        self.mamba_prefill_final_only = (
+            os.environ.get("SGLANG_MAMBA_PREFILL_FINAL_ONLY", "1").lower()
+            not in ("0", "false")
+            and self.enable_mamba_extra_buffer
+            and not params.enable_mamba_extra_buffer_lazy
+            and set(self.tree_components) == {ComponentType.FULL, ComponentType.MAMBA}
+        )
+        if self.mamba_prefill_final_only:
+            logger.info("Mamba sparse-prefill v2: one cold-start branch plus final checkpoint")
+        # Refresh only the released Mamba state when its final device lock drops.
+        # Independent of sparse-prefill, including for extra_buffer_lazy.
+        self.mamba_refresh_on_unlock = (
+            self.is_mamba_enabled
+            and os.environ.get("SGLANG_MAMBA_REFRESH_ON_UNLOCK", "1").strip().lower()
+            not in ("0", "false", "off", "no")
+        )
+        if self.mamba_refresh_on_unlock:
+            logger.info("Mamba LRU: refresh checkpoint before final device unlock")
         # SWA window size (None when SWA is not enabled).
         self._sliding_window_size = (
             params.sliding_window_size if self.is_swa_enabled else None
@@ -916,6 +936,28 @@
             req.req_pool_idx, : len(token_ids)
         ]
 
+        if chunked and self.mamba_prefill_final_only:
+            # Publish only the one branch chosen at the initial cold admission.
+            # The scheduler cuts a chunk exactly here, so the tracked state and
+            # the branch key agree. Never adopt later prefix-match hints.
+            branch = getattr(req, "mamba_prefill_branch_seqlen", None)
+            publish_branch = (
+                getattr(req, "mamba_prefill_initial_cached_tokens", None) == 0
+                and not getattr(req, "mamba_prefill_branch_used", False)
+                and branch is not None
+                and branch > 0
+                and len(token_ids) == branch
+                and req.mamba_last_track_seqlen == branch
+            )
+            if publish_branch:
+                # One attempt even if the pool cannot supply a donation slot.
+                req.mamba_prefill_branch_used = True
+            else:
+                # New KV remains request-owned; retain the existing tree lock.
+                req.prefix_indices = kv_indices_orig.to(dtype=torch.int64, copy=True)
+                # Preserve last_track_* for a final chunk shorter than the grid.
+                return
+
         # components prepare insert data + return effective cache_len
         insert_params = InsertParams(
             prev_prefix_len=req.cache_protected_len,
--- a/python/sglang/srt/managers/schedule_batch.py
+++ b/python/sglang/srt/managers/schedule_batch.py
@@ -959,6 +959,10 @@
         # the branching point seqlen to track mamba state. If set, given by prefix match,
         # it will be the tracked seqlen in the ping pong buffer for the right prefill pass.
         self.mamba_branching_seqlen: Optional[int] = None
+        # Frozen on the first prefill forward, not on each chunk/rematch.
+        self.mamba_prefill_initial_cached_tokens: Optional[int] = None
+        self.mamba_prefill_branch_seqlen: Optional[int] = None
+        self.mamba_prefill_branch_used: bool = False
         # Deferred COW: source mamba pool index from radix cache node (copy on forward stream)
         self.mamba_cow_src_index: Optional[torch.Tensor] = None
         # Deferred clear: newly allocated mamba slot needs zeroing on forward stream
@@ -2673,6 +2677,16 @@
         self,
         req: Req,
     ) -> _MambaRadixCacheV2TrackEntry:
+        if (
+            getattr(self.tree_cache, "mamba_prefill_final_only", False)
+            and req.mamba_prefill_initial_cached_tokens is None
+        ):
+            req.mamba_prefill_initial_cached_tokens = len(req.prefix_indices)
+            req.mamba_prefill_branch_seqlen = (
+                req.mamba_branching_seqlen
+                if req.mamba_prefill_initial_cached_tokens == 0
+                else None
+            )
         chunk_size = mamba_cache_chunk_size()
         # The donated depth has to be a radix node boundary. Read the tree's own
         # page rather than re-deriving how DCP widens it; the kernel still
@@ -2739,7 +2753,9 @@
                         req.mamba_next_track_idx
                     )
                 )
-            if req.mamba_branching_seqlen is not None:
+            if req.mamba_branching_seqlen is not None and not getattr(
+                self.tree_cache, "mamba_prefill_final_only", False
+            ):
                 # track branching point in this forward if the branching point
                 # is within the current extend batch.
                 branching_seqlen_aligned_mask = (
--- a/python/sglang/srt/managers/schedule_policy.py
+++ b/python/sglang/srt/managers/schedule_policy.py
@@ -1001,7 +1001,33 @@
             else AddReqResult.CONTINUE
         )
 
+    def _cap_chunk_budget_at_mamba_branch(self, req: Req):
+        """End one cold prefill chunk at its initial shared-prefix divergence.
+
+        Cap the batch chunk budget too: consuming the branch chunk must stop
+        further admissions from creating a second unfinished chunked request.
+        """
+        if (
+            not getattr(self.tree_cache, "mamba_prefill_final_only", False)
+            or self.rem_chunk_tokens is None
+            or self.dllm_config is not None
+            or getattr(req, "mamba_prefill_branch_used", False)
+        ):
+            return
+        initial = getattr(req, "mamba_prefill_initial_cached_tokens", None)
+        if initial is None:
+            initial = len(req.prefix_indices)
+            branch = req.mamba_branching_seqlen
+        else:
+            branch = req.mamba_prefill_branch_seqlen
+        if initial != 0 or branch is None:
+            return
+        prefix_len = len(req.prefix_indices)
+        if prefix_len < branch < len(req.full_untruncated_fill_ids):
+            self.rem_chunk_tokens = min(self.rem_chunk_tokens, branch - prefix_len)
+
     def add_chunked_req(self, req: Req):
+        self._cap_chunk_budget_at_mamba_branch(req)
         if self.dllm_config is not None:
             _rem_tokens = self._get_dllm_remain_tokens()
         else:
@@ -1220,6 +1246,8 @@
         if req.sampling_params.ignore_eos and getattr(self.tree_cache, "disable", True):
             return self.add_one_req_ignore_eos(req)
 
+        self._cap_chunk_budget_at_mamba_branch(req)
+
         # Reserve page_size for page-alignment overhead: the paged allocator may
         # consume one extra page per request (see alloc_extend), which
         # _update_prefill_budget also deducts.
--- a/python/sglang/srt/mem_cache/unified_cache/components/mamba_component.py
+++ b/python/sglang/srt/mem_cache/unified_cache/components/mamba_component.py
@@ -489,6 +489,11 @@
 
         if cd.lock_ref > 0:
             if cd.lock_ref == 1:
+                # A prefill checkpoint can stay locked for a long decode. Make
+                # its age reflect release, not the start of that decode, before
+                # it becomes evictable. Touch only this state, never ancestors.
+                if getattr(self.cache, "mamba_refresh_on_unlock", False):
+                    self.tree_core.lru_lists[ct].reset_node_mru(node)
                 vlen = len(value)
                 self.tree_core.component_evictable_size_[ct] += vlen
                 self.tree_core.component_protected_size_[ct] -= vlen
PENNY_EMBEDDED_FILE_4

cat > "$STAGE/patches/05-prefetch-lookahead.py" <<'PENNY_EMBEDDED_FILE_10'
import ast
import hashlib
import pathlib
import subprocess
import sys
PIN = 'd00d88efc8d6281b12be4f4073126aec95038c55'
BEFORE = {'python/sglang/srt/environ.py': '0b98ad866be4da732833ef38d185eb8cb9e41329fe65caa82568c72810c94449', 'python/sglang/srt/server_args.py': '39a0472496b813bb8ac26eeb4e83fd894753de3f7264683531b01d8a9f7334ea', 'python/sglang/srt/model_loader/weight_utils.py': '3f9ade94beda68d16ad8a3c629a067f13b124ed2fab97a9ee47b9a7debe0881d', 'python/sglang/srt/model_loader/loader.py': '1016b2c6c2d7eb50ef25e0a731e20096e4b79239bccb6fea26aa59d3f29bf8fe', 'python/sglang/srt/model_executor/model_runner_components/startup_weight_load.py': '40b44a547a454de3dd6d4ca7556336a1e70c9a73ee73977345e4c433543d2279'}
AFTER = {'python/sglang/srt/environ.py': '7431c4e0e679d1983ade2c3d9a8689c0ac78666d166d6743188434f2f25c5ed1', 'python/sglang/srt/server_args.py': '68137576b47a67e909ca953f5d8c094a5561ca4a4acd4bf24e6d234ea62d8018', 'python/sglang/srt/model_loader/weight_utils.py': 'b92cc460f69fb3e37a9ac9f909f5f8691cb3dcfdc02d2a00cf85fd55cb24494b', 'python/sglang/srt/model_loader/loader.py': 'd12e619c9a334f1af61d79d1958c59554e0931986da4942b1f570562d0752bb1', 'python/sglang/srt/model_executor/model_runner_components/startup_weight_load.py': '26a9a1ac26daf3a372fcfb4ecae90a77ba9cbd4de82653cf5c6c593a45c99ea5'}
PATCH = 'diff --git a/python/sglang/srt/environ.py b/python/sglang/srt/environ.py\n--- a/python/sglang/srt/environ.py\n+++ b/python/sglang/srt/environ.py\n@@ -303,6 +303,9 @@\n     # path is ported, so setting this fails loudly instead of degrading.\n     SGLANG_QWEN_DSA_USE_FP8_INDEXER = EnvBool(False)\n     SGLANG_PREFETCH_BLOCK_SIZE_MB = EnvInt(16)\n+    # With checkpoint prefetch enabled, keep at most one shard ahead of the\n+    # serial mmap loader instead of staging the entire checkpoint independently.\n+    SGLANG_WEIGHT_LOADER_PREFETCH_LOOKAHEAD = EnvBool(True)\n     SGLANG_GEMMA_OUT_OF_PLACE_POSITION_MUTATION = EnvBool(False)\n     SGLANG_ENABLE_WEIGHT_LOADER_V2 = EnvBool(False)\n     # Copy rank-local MoE slices into independent CPU storage before H2D when\ndiff --git a/python/sglang/srt/server_args.py b/python/sglang/srt/server_args.py\n--- a/python/sglang/srt/server_args.py\n+++ b/python/sglang/srt/server_args.py\n@@ -3388,7 +3388,7 @@\n     ] = False\n     weight_loader_prefetch_num_threads: A[\n         int,\n-        "Number of threads per rank for checkpoint prefetching (default: 4).",\n+        "Number of threads per rank for checkpoint prefetching (default: 4). In lookahead mode, threads read disjoint byte ranges of the same shard.",\n         NS("model"),\n     ] = 4\n     weight_loader_drop_cache_after_load: A[\ndiff --git a/python/sglang/srt/model_loader/weight_utils.py b/python/sglang/srt/model_loader/weight_utils.py\n--- a/python/sglang/srt/model_loader/weight_utils.py\n+++ b/python/sglang/srt/model_loader/weight_utils.py\n@@ -1080,41 +1080,152 @@\n             os.close(fd)\n \n \n+def _prefetch_checkpoint_range(\n+    file_path: str,\n+    start: int,\n+    end: int,\n+    cancel_event: threading.Event,\n+) -> None:\n+    """Read one disjoint byte range sequentially, without userspace read-ahead."""\n+    with open(file_path, "rb", buffering=0) as f:\n+        f.seek(start)\n+        remaining = end - start\n+        block_size = _get_prefetch_block_size()\n+        if block_size <= 0:\n+            raise ValueError("checkpoint prefetch block size must be positive")\n+        while remaining and not cancel_event.is_set():\n+            data = f.read(min(block_size, remaining))\n+            if not data:\n+                raise OSError(f"Checkpoint file shrank while prefetching {file_path!r}")\n+            remaining -= len(data)\n+\n+\n+def _lookahead_prefetch_files(\n+    files: List[str], num_threads: int = 1\n+) -> Generator[str, None, None]:\n+    """Warm the current shard, then at most one successor in loader order.\n+\n+    This is per-process staging: unlike the eager prefetcher it does not split\n+    files across ranks, whose loaders may be progressing at different speeds.\n+    Workers read disjoint contiguous ranges of ONE shard. No page-cache\n+    eviction is performed here.\n+    """\n+    if num_threads < 1:\n+        raise ValueError("weight loader prefetch num_threads must be >= 1")\n+    if not files:\n+        return\n+    logger.info(\n+        "Checkpoint prefetch: one-shard lookahead in loader order "\n+        "(%d shards, up to %d range-reader threads per shard)",\n+        len(files),\n+        num_threads,\n+    )\n+    cancel_event = threading.Event()\n+    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:\n+        def submit_shard(path):\n+            try:\n+                if num_threads == 1:\n+                    return [executor.submit(_prefetch_checkpoint_file, path, cancel_event)]\n+                size = os.path.getsize(path)\n+                block_size = _get_prefetch_block_size()\n+                if block_size <= 0:\n+                    raise ValueError("checkpoint prefetch block size must be positive")\n+                blocks = (size + block_size - 1) // block_size\n+                workers = min(num_threads, max(1, blocks))\n+                if workers == 1:\n+                    return [executor.submit(_prefetch_checkpoint_file, path, cancel_event)]\n+                return [\n+                    executor.submit(\n+                        _prefetch_checkpoint_range,\n+                        path,\n+                        (blocks * i // workers) * block_size,\n+                        min(size, (blocks * (i + 1) // workers) * block_size),\n+                        cancel_event,\n+                    )\n+                    for i in range(workers)\n+                ]\n+            except OSError as exc:\n+                logger.warning(\n+                    "Could not stage checkpoint file %r: %s; loading it normally",\n+                    path,\n+                    exc,\n+                )\n+                return []\n+\n+        try:\n+            pending = submit_shard(files[0])\n+            for index, path in enumerate(files):\n+                error = None\n+                for future in pending:\n+                    try:\n+                        future.result()\n+                    except Exception as exc:\n+                        error = error or exc\n+                if error is not None:\n+                    # Staging is only an optimization. The real loader remains\n+                    # responsible for reporting missing or unreadable weights.\n+                    logger.warning(\n+                        "Failed to prefetch checkpoint file %r: %s; "\n+                        "loading it normally",\n+                        path,\n+                        error,\n+                    )\n+                if index + 1 < len(files):\n+                    pending = submit_shard(files[index + 1])\n+                yield path\n+        finally:\n+            # Closing the weights iterator early must also stop the lookahead\n+            # reader. Reads observe cancellation between sequential blocks.\n+            cancel_event.set()\n+\n+\n def safetensors_weights_iterator(\n     hf_weights_files: List[str],\n     disable_mmap: bool = False,\n     prefetch: bool = False,\n     prefetch_num_threads: int = 4,\n     drop_cache_after_load: bool = False,\n+    prefetch_lookahead: bool = False,\n ) -> Generator[Tuple[str, torch.Tensor], None, None]:\n     """Iterate over the weights in the model safetensor files."""\n     enable_tqdm = (\n         not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0\n     )\n \n-    if prefetch and not disable_mmap:\n+    lookahead = prefetch and prefetch_lookahead and not disable_mmap\n+    if prefetch and not disable_mmap and not lookahead:\n         _prefetch_all_checkpoints(\n             sorted(hf_weights_files), num_threads=prefetch_num_threads\n         )\n \n-    for st_file in tqdm(\n-        hf_weights_files,\n-        desc="Loading safetensors checkpoint shards",\n-        disable=not enable_tqdm,\n-        bar_format=BAR_FORMAT,\n-        position=tqdm._get_free_pos(),\n-    ):\n-        if disable_mmap:\n-            with open(st_file, "rb") as f:\n-                result = safetensors.torch.load(f.read())\n-                for name in sorted(result.keys()):\n-                    yield name, result[name]\n-        else:\n-            with safetensors.safe_open(st_file, framework="pt", device="cpu") as f:\n-                for name in f.keys():\n-                    yield name, f.get_tensor(name)\n-        if drop_cache_after_load:\n-            _drop_file_cache_after_load(st_file)\n+    files = (\n+        _lookahead_prefetch_files(hf_weights_files, num_threads=prefetch_num_threads)\n+        if lookahead\n+        else iter(hf_weights_files)\n+    )\n+    try:\n+        for st_file in tqdm(\n+            files,\n+            total=len(hf_weights_files),\n+            desc="Loading safetensors checkpoint shards",\n+            disable=not enable_tqdm,\n+            bar_format=BAR_FORMAT,\n+            position=tqdm._get_free_pos(),\n+        ):\n+            if disable_mmap:\n+                with open(st_file, "rb") as f:\n+                    result = safetensors.torch.load(f.read())\n+                    for name in sorted(result.keys()):\n+                        yield name, result[name]\n+            else:\n+                with safetensors.safe_open(st_file, framework="pt", device="cpu") as f:\n+                    for name in f.keys():\n+                        yield name, f.get_tensor(name)\n+            if drop_cache_after_load:\n+                _drop_file_cache_after_load(st_file)\n+    finally:\n+        if lookahead:\n+            files.close()\n \n \n def fastsafetensors_weights_iterator(\ndiff --git a/python/sglang/srt/model_loader/loader.py b/python/sglang/srt/model_loader/loader.py\n--- a/python/sglang/srt/model_loader/loader.py\n+++ b/python/sglang/srt/model_loader/loader.py\n@@ -575,8 +575,13 @@\n         elif use_safetensors:\n             weight_loader_disable_mmap = get_model().weight_loader_disable_mmap\n             configured_prefetch = get_model().weight_loader_prefetch_checkpoints\n+            lookahead_prefetch = (\n+                envs.SGLANG_WEIGHT_LOADER_PREFETCH_LOOKAHEAD.get()\n+                and (configured_prefetch or startup_prefetch_started)\n+            )\n             start_iterator_prefetch = (\n-                configured_prefetch and not startup_prefetch_started\n+                (configured_prefetch and not startup_prefetch_started)\n+                or lookahead_prefetch\n             )\n             concurrent_prefetch_active = (\n                 startup_prefetch_active or start_iterator_prefetch\n@@ -585,6 +590,18 @@\n             weight_loader_drop_cache_after_load = (\n                 get_model().weight_loader_drop_cache_after_load\n             )\n+            if lookahead_prefetch:\n+                if (\n+                    weight_loader_disable_mmap\n+                    or self.load_config.load_format == LoadFormat.FASTSAFETENSORS\n+                ):\n+                    raise ValueError(\n+                        "SGLANG_WEIGHT_LOADER_PREFETCH_LOOKAHEAD requires the "\n+                        "standard safetensors mmap loader"\n+                    )\n+                # A single consumer defines the bound. Even an explicit\n+                # multithread setting must not start reading later shards.\n+                use_multithread = False\n \n             # Prefetch and multi-threaded loading both read the same shards,\n             # competing for I/O on shared/network storage. When prefetch is\n@@ -639,6 +656,7 @@\n                     prefetch=start_iterator_prefetch,\n                     prefetch_num_threads=prefetch_num_threads,\n                     drop_cache_after_load=weight_loader_drop_cache_after_load,\n+                    prefetch_lookahead=lookahead_prefetch,\n                 )\n \n         else:\ndiff --git a/python/sglang/srt/model_executor/model_runner_components/startup_weight_load.py b/python/sglang/srt/model_executor/model_runner_components/startup_weight_load.py\n--- a/python/sglang/srt/model_executor/model_runner_components/startup_weight_load.py\n+++ b/python/sglang/srt/model_executor/model_runner_components/startup_weight_load.py\n@@ -482,16 +482,27 @@\n             )\n         assert self._capture_ready_at is not None\n         prefetch_started_at = time.perf_counter()\n-        self._prefetch_handle = self._loader.start_checkpoint_prefetch(\n-            self._resolved_sources,\n-            num_threads=self._options.prefetch_num_threads,\n-        )\n+        from sglang.srt.environ import envs\n+\n+        if envs.SGLANG_WEIGHT_LOADER_PREFETCH_LOOKAHEAD.get():\n+            # The commit iterator owns bounded staging. An independent startup\n+            # reader would otherwise warm the whole checkpoint during capture.\n+            self._prefetch_handle = None\n+            logger.info(\n+                "Deferring checkpoint prefetch to one-shard lookahead at weight commit"\n+            )\n+        else:\n+            self._prefetch_handle = self._loader.start_checkpoint_prefetch(\n+                self._resolved_sources,\n+                num_threads=self._options.prefetch_num_threads,\n+            )\n         self._prefetch_started_at = prefetch_started_at\n         self._state = StartupWeightLoadState.PREFETCHING\n-        logger.info(\n-            "Started checkpoint prefetching %.2f s after capture-safe model prep",\n-            self._prefetch_started_at - self._capture_ready_at,\n-        )\n+        if self._prefetch_handle is not None:\n+            logger.info(\n+                "Started checkpoint prefetching %.2f s after capture-safe model prep",\n+                self._prefetch_started_at - self._capture_ready_at,\n+            )\n \n     def finalize(self) -> None:\n         if self._state == StartupWeightLoadState.READY:\n@@ -544,7 +555,9 @@\n         )\n \n     def _prepare_prefetch_for_commit(self) -> bool:\n-        assert self._prefetch_handle is not None\n+        if self._prefetch_handle is None:\n+            # Lookahead staging is started by the commit\'s weights iterator.\n+            return False\n         if not self._prefetch_handle.failed:\n             return not self._prefetch_handle.done\n \n'

def fail(message):
    raise SystemExit('ERROR: ' + message)

def main():
    root = pathlib.Path(sys.argv[1]).expanduser().resolve()
    mode = sys.argv[2]
    if mode not in ('apply', 'check', 'revert'):
        fail('mode must be apply, check, or revert')

    def git(*args, **kwargs):
        return subprocess.run(['git', '-C', str(root), *args], text=True, capture_output=True, **kwargs)
    top = git('rev-parse', '--show-toplevel')
    if top.returncode or pathlib.Path(top.stdout.strip()).resolve() != root:
        fail('ROOT must be the root of a Git checkout')
    rev = git('rev-parse', 'HEAD')
    if rev.returncode or rev.stdout.strip() != PIN:
        fail('expected Pennyroyal v2.5.3 at ' + PIN + '; found ' + rev.stdout.strip())

    def hashes():
        result = {}
        for name in BEFORE:
            p = root / name
            if any((q.is_symlink() for q in (p, *p.parents) if q != root.parent)):
                fail('symlink in target path: ' + name)
            if p.exists() and (not p.is_file()):
                fail('target is not a regular file: ' + name)
            result[name] = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
        return result
    state = hashes()
    if state == AFTER:
        if mode != 'revert':
            print('Already applied: all five lookahead source hashes match this port.')
            return
        reverse, expected = (True, BEFORE)
    elif state == BEFORE:
        if mode == 'revert':
            print('Already unpatched: no changes needed.')
            return
        reverse, expected = (False, AFTER)
    else:
        changed = [n for n in state if state[n] != BEFORE[n] and state[n] != AFTER[n]]
        fail('target files differ or port is partially applied; nothing changed. ' + ', '.join(changed))
    flags = ['--reverse'] if reverse else []
    checked = git('apply', '--check', *flags, '-', input=PATCH)
    if checked.returncode:
        fail('patch check failed; nothing changed.\n' + checked.stderr)
    if mode == 'check':
        print('Compatible: pinned release, pristine target files, patch applies cleanly.')
        return
    result = git('apply', *flags, '-', input=PATCH)
    if result.returncode:
        fail('git apply failed: ' + result.stderr)
    if hashes() != expected:
        fail('post-write hash verification failed; inspect checkout before serving')
    for name, digest in expected.items():
        if digest is not None:
            ast.parse((root / name).read_text(), filename=name)
    print('Reverted successfully.' if reverse else 'Applied successfully; Python syntax and all file hashes verified.')
    if not reverse:
        print('Restart the server using this source checkout. No auto-commit was made.')
        print('Active with --weight-loader-prefetch-checkpoints; lookahead defaults on, with four range readers per shard.')
        print('A/B disable: SGLANG_WEIGHT_LOADER_PREFETCH_LOOKAHEAD=0 (restart required).')
if __name__ == '__main__':
    main()
PENNY_EMBEDDED_FILE_10

cat > "$STAGE/patches/05-prefetch-lookahead.patch" <<'PENNY_EMBEDDED_FILE_11'
diff --git a/python/sglang/srt/environ.py b/python/sglang/srt/environ.py
--- a/python/sglang/srt/environ.py
+++ b/python/sglang/srt/environ.py
@@ -303,6 +303,9 @@
     # path is ported, so setting this fails loudly instead of degrading.
     SGLANG_QWEN_DSA_USE_FP8_INDEXER = EnvBool(False)
     SGLANG_PREFETCH_BLOCK_SIZE_MB = EnvInt(16)
+    # With checkpoint prefetch enabled, keep at most one shard ahead of the
+    # serial mmap loader instead of staging the entire checkpoint independently.
+    SGLANG_WEIGHT_LOADER_PREFETCH_LOOKAHEAD = EnvBool(True)
     SGLANG_GEMMA_OUT_OF_PLACE_POSITION_MUTATION = EnvBool(False)
     SGLANG_ENABLE_WEIGHT_LOADER_V2 = EnvBool(False)
     # Copy rank-local MoE slices into independent CPU storage before H2D when
diff --git a/python/sglang/srt/server_args.py b/python/sglang/srt/server_args.py
--- a/python/sglang/srt/server_args.py
+++ b/python/sglang/srt/server_args.py
@@ -3388,7 +3388,7 @@
     ] = False
     weight_loader_prefetch_num_threads: A[
         int,
-        "Number of threads per rank for checkpoint prefetching (default: 4).",
+        "Number of threads per rank for checkpoint prefetching (default: 4). In lookahead mode, threads read disjoint byte ranges of the same shard.",
         NS("model"),
     ] = 4
     weight_loader_drop_cache_after_load: A[
diff --git a/python/sglang/srt/model_loader/weight_utils.py b/python/sglang/srt/model_loader/weight_utils.py
--- a/python/sglang/srt/model_loader/weight_utils.py
+++ b/python/sglang/srt/model_loader/weight_utils.py
@@ -1080,41 +1080,152 @@
             os.close(fd)
 
 
+def _prefetch_checkpoint_range(
+    file_path: str,
+    start: int,
+    end: int,
+    cancel_event: threading.Event,
+) -> None:
+    """Read one disjoint byte range sequentially, without userspace read-ahead."""
+    with open(file_path, "rb", buffering=0) as f:
+        f.seek(start)
+        remaining = end - start
+        block_size = _get_prefetch_block_size()
+        if block_size <= 0:
+            raise ValueError("checkpoint prefetch block size must be positive")
+        while remaining and not cancel_event.is_set():
+            data = f.read(min(block_size, remaining))
+            if not data:
+                raise OSError(f"Checkpoint file shrank while prefetching {file_path!r}")
+            remaining -= len(data)
+
+
+def _lookahead_prefetch_files(
+    files: List[str], num_threads: int = 1
+) -> Generator[str, None, None]:
+    """Warm the current shard, then at most one successor in loader order.
+
+    This is per-process staging: unlike the eager prefetcher it does not split
+    files across ranks, whose loaders may be progressing at different speeds.
+    Workers read disjoint contiguous ranges of ONE shard. No page-cache
+    eviction is performed here.
+    """
+    if num_threads < 1:
+        raise ValueError("weight loader prefetch num_threads must be >= 1")
+    if not files:
+        return
+    logger.info(
+        "Checkpoint prefetch: one-shard lookahead in loader order "
+        "(%d shards, up to %d range-reader threads per shard)",
+        len(files),
+        num_threads,
+    )
+    cancel_event = threading.Event()
+    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
+        def submit_shard(path):
+            try:
+                if num_threads == 1:
+                    return [executor.submit(_prefetch_checkpoint_file, path, cancel_event)]
+                size = os.path.getsize(path)
+                block_size = _get_prefetch_block_size()
+                if block_size <= 0:
+                    raise ValueError("checkpoint prefetch block size must be positive")
+                blocks = (size + block_size - 1) // block_size
+                workers = min(num_threads, max(1, blocks))
+                if workers == 1:
+                    return [executor.submit(_prefetch_checkpoint_file, path, cancel_event)]
+                return [
+                    executor.submit(
+                        _prefetch_checkpoint_range,
+                        path,
+                        (blocks * i // workers) * block_size,
+                        min(size, (blocks * (i + 1) // workers) * block_size),
+                        cancel_event,
+                    )
+                    for i in range(workers)
+                ]
+            except OSError as exc:
+                logger.warning(
+                    "Could not stage checkpoint file %r: %s; loading it normally",
+                    path,
+                    exc,
+                )
+                return []
+
+        try:
+            pending = submit_shard(files[0])
+            for index, path in enumerate(files):
+                error = None
+                for future in pending:
+                    try:
+                        future.result()
+                    except Exception as exc:
+                        error = error or exc
+                if error is not None:
+                    # Staging is only an optimization. The real loader remains
+                    # responsible for reporting missing or unreadable weights.
+                    logger.warning(
+                        "Failed to prefetch checkpoint file %r: %s; "
+                        "loading it normally",
+                        path,
+                        error,
+                    )
+                if index + 1 < len(files):
+                    pending = submit_shard(files[index + 1])
+                yield path
+        finally:
+            # Closing the weights iterator early must also stop the lookahead
+            # reader. Reads observe cancellation between sequential blocks.
+            cancel_event.set()
+
+
 def safetensors_weights_iterator(
     hf_weights_files: List[str],
     disable_mmap: bool = False,
     prefetch: bool = False,
     prefetch_num_threads: int = 4,
     drop_cache_after_load: bool = False,
+    prefetch_lookahead: bool = False,
 ) -> Generator[Tuple[str, torch.Tensor], None, None]:
     """Iterate over the weights in the model safetensor files."""
     enable_tqdm = (
         not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
     )
 
-    if prefetch and not disable_mmap:
+    lookahead = prefetch and prefetch_lookahead and not disable_mmap
+    if prefetch and not disable_mmap and not lookahead:
         _prefetch_all_checkpoints(
             sorted(hf_weights_files), num_threads=prefetch_num_threads
         )
 
-    for st_file in tqdm(
-        hf_weights_files,
-        desc="Loading safetensors checkpoint shards",
-        disable=not enable_tqdm,
-        bar_format=BAR_FORMAT,
-        position=tqdm._get_free_pos(),
-    ):
-        if disable_mmap:
-            with open(st_file, "rb") as f:
-                result = safetensors.torch.load(f.read())
-                for name in sorted(result.keys()):
-                    yield name, result[name]
-        else:
-            with safetensors.safe_open(st_file, framework="pt", device="cpu") as f:
-                for name in f.keys():
-                    yield name, f.get_tensor(name)
-        if drop_cache_after_load:
-            _drop_file_cache_after_load(st_file)
+    files = (
+        _lookahead_prefetch_files(hf_weights_files, num_threads=prefetch_num_threads)
+        if lookahead
+        else iter(hf_weights_files)
+    )
+    try:
+        for st_file in tqdm(
+            files,
+            total=len(hf_weights_files),
+            desc="Loading safetensors checkpoint shards",
+            disable=not enable_tqdm,
+            bar_format=BAR_FORMAT,
+            position=tqdm._get_free_pos(),
+        ):
+            if disable_mmap:
+                with open(st_file, "rb") as f:
+                    result = safetensors.torch.load(f.read())
+                    for name in sorted(result.keys()):
+                        yield name, result[name]
+            else:
+                with safetensors.safe_open(st_file, framework="pt", device="cpu") as f:
+                    for name in f.keys():
+                        yield name, f.get_tensor(name)
+            if drop_cache_after_load:
+                _drop_file_cache_after_load(st_file)
+    finally:
+        if lookahead:
+            files.close()
 
 
 def fastsafetensors_weights_iterator(
diff --git a/python/sglang/srt/model_loader/loader.py b/python/sglang/srt/model_loader/loader.py
--- a/python/sglang/srt/model_loader/loader.py
+++ b/python/sglang/srt/model_loader/loader.py
@@ -575,8 +575,13 @@
         elif use_safetensors:
             weight_loader_disable_mmap = get_model().weight_loader_disable_mmap
             configured_prefetch = get_model().weight_loader_prefetch_checkpoints
+            lookahead_prefetch = (
+                envs.SGLANG_WEIGHT_LOADER_PREFETCH_LOOKAHEAD.get()
+                and (configured_prefetch or startup_prefetch_started)
+            )
             start_iterator_prefetch = (
-                configured_prefetch and not startup_prefetch_started
+                (configured_prefetch and not startup_prefetch_started)
+                or lookahead_prefetch
             )
             concurrent_prefetch_active = (
                 startup_prefetch_active or start_iterator_prefetch
@@ -585,6 +590,18 @@
             weight_loader_drop_cache_after_load = (
                 get_model().weight_loader_drop_cache_after_load
             )
+            if lookahead_prefetch:
+                if (
+                    weight_loader_disable_mmap
+                    or self.load_config.load_format == LoadFormat.FASTSAFETENSORS
+                ):
+                    raise ValueError(
+                        "SGLANG_WEIGHT_LOADER_PREFETCH_LOOKAHEAD requires the "
+                        "standard safetensors mmap loader"
+                    )
+                # A single consumer defines the bound. Even an explicit
+                # multithread setting must not start reading later shards.
+                use_multithread = False
 
             # Prefetch and multi-threaded loading both read the same shards,
             # competing for I/O on shared/network storage. When prefetch is
@@ -639,6 +656,7 @@
                     prefetch=start_iterator_prefetch,
                     prefetch_num_threads=prefetch_num_threads,
                     drop_cache_after_load=weight_loader_drop_cache_after_load,
+                    prefetch_lookahead=lookahead_prefetch,
                 )
 
         else:
diff --git a/python/sglang/srt/model_executor/model_runner_components/startup_weight_load.py b/python/sglang/srt/model_executor/model_runner_components/startup_weight_load.py
--- a/python/sglang/srt/model_executor/model_runner_components/startup_weight_load.py
+++ b/python/sglang/srt/model_executor/model_runner_components/startup_weight_load.py
@@ -482,16 +482,27 @@
             )
         assert self._capture_ready_at is not None
         prefetch_started_at = time.perf_counter()
-        self._prefetch_handle = self._loader.start_checkpoint_prefetch(
-            self._resolved_sources,
-            num_threads=self._options.prefetch_num_threads,
-        )
+        from sglang.srt.environ import envs
+
+        if envs.SGLANG_WEIGHT_LOADER_PREFETCH_LOOKAHEAD.get():
+            # The commit iterator owns bounded staging. An independent startup
+            # reader would otherwise warm the whole checkpoint during capture.
+            self._prefetch_handle = None
+            logger.info(
+                "Deferring checkpoint prefetch to one-shard lookahead at weight commit"
+            )
+        else:
+            self._prefetch_handle = self._loader.start_checkpoint_prefetch(
+                self._resolved_sources,
+                num_threads=self._options.prefetch_num_threads,
+            )
         self._prefetch_started_at = prefetch_started_at
         self._state = StartupWeightLoadState.PREFETCHING
-        logger.info(
-            "Started checkpoint prefetching %.2f s after capture-safe model prep",
-            self._prefetch_started_at - self._capture_ready_at,
-        )
+        if self._prefetch_handle is not None:
+            logger.info(
+                "Started checkpoint prefetching %.2f s after capture-safe model prep",
+                self._prefetch_started_at - self._capture_ready_at,
+            )
 
     def finalize(self) -> None:
         if self._state == StartupWeightLoadState.READY:
@@ -544,7 +555,9 @@
         )
 
     def _prepare_prefetch_for_commit(self) -> bool:
-        assert self._prefetch_handle is not None
+        if self._prefetch_handle is None:
+            # Lookahead staging is started by the commit's weights iterator.
+            return False
         if not self._prefetch_handle.failed:
             return not self._prefetch_handle.done
 
PENNY_EMBEDDED_FILE_11

cat > "$STAGE/verify_bundle.py" <<'PENNY_EMBEDDED_FILE_5'
"""Verify wheel contents, the offline bundle, or its installed environment."""
import ast
import hashlib
import importlib.metadata as md
import json
import pathlib
import platform
import sys
import sysconfig
import zipfile


def require(ok, message):
    if not ok:
        raise SystemExit(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_patches(bundle):
    result = {}
    required = {'01-lowm.py', '02-kv-budget.py', '03-marlin-dtype.py', '04-prefill-v22.py', '05-prefetch-lookahead.py'}
    require({p.name for p in (bundle / 'patches').glob('*.py')} == required,
            'Expected exactly the five reviewed patch helpers, including prefill v2.2 and lookahead')
    # Read literal assignments, never execute the patch helpers to verify a wheel.
    for path in sorted((bundle / 'patches').glob('*.py')):
        constants = {}
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                try:
                    constants[node.targets[0].id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
        expected = constants.get('AFTER', {constants.get('REL'): constants.get('AFTER_SHA')})
        require(all(k and v for k, v in expected.items()), 'Missing patch hashes: ' + str(path))
        # Later patches may extend an earlier file only when the recorded
        # BEFORE hash matches that earlier patch's exact AFTER hash.
        for name in expected:
            key = name.removeprefix('python/')
            if key in result:
                require(constants.get('BEFORE', {}).get(name) == result[key],
                        'Patch hash chain mismatch: ' + str(path) + ': ' + name)
        result.update({k.removeprefix('python/'): v for k, v in expected.items()})
    require(len(result) == 13, 'Expected thirteen source files across five patches')
    return result


def check_wheel(bundle):
    wheels = list((bundle / 'wheels').glob('sglang-*.whl'))
    require(len(wheels) == 1, 'Expected exactly one SGLang wheel')
    expected = load_patches(bundle)
    with zipfile.ZipFile(wheels[0]) as z:
        for name, digest in expected.items():
            require(name in z.namelist(), 'Wheel lacks patched source: ' + name)
            require(hashlib.sha256(z.read(name)).hexdigest() == digest, 'Wheel patch mismatch: ' + name)
        require(any(n.endswith('.so') and ('_core' in n or 'rust' in n or '_grpc' in n) for n in z.namelist()),
                'SGLang wheel is missing compiled Rust extensions')
        # Kernels use packaged source files at JIT time; do not accept a Python-only shell.
        require(any(n.startswith('sglang/kernels/') and n.endswith(('.cuh', '.cu', '.cpp')) for n in z.namelist()),
                'SGLang JIT C++/CUDA sources missing from wheel')
    print('SGLang wheel: all five patches (including prefill v2.2 and lookahead), Rust extensions and JIT sources verified.')


def check_platform(manifest):
    require(platform.python_implementation() == 'CPython', 'Use CPython from the Kaggle image')
    require(list(sys.version_info[:2]) == manifest['python_version'],
            'Python major/minor differs from build image; rebuild in the matching Kaggle image')
    require(platform.machine() == manifest['machine'], 'CPU architecture differs from build image')
    require(sysconfig.get_config_var('SOABI') == manifest['soabi'], 'Python ABI differs from build image')
    built = manifest['libc']
    running = platform.libc_ver()
    if built[0] == 'glibc' and running[0] == 'glibc':
        require(tuple(map(int, running[1].split('.'))) >= tuple(map(int, built[1].split('.'))),
                'Runtime glibc is older than the build image; rebuild in the matching image')


def check_installed(bundle):
    manifest = json.loads((bundle / 'manifest.json').read_text())
    check_platform(manifest)
    for name, expected in manifest['packages'].items():
        require(md.version(name) == expected, 'Installed version mismatch: ' + name)
    dist = md.distribution('sglang')
    for name, digest in load_patches(bundle).items():
        require(sha(pathlib.Path(dist.locate_file(name))) == digest, 'Installed patch mismatch: ' + name)
    # Import high-value dependencies without requiring a GPU or a model.
    import torch
    import transformers
    from transformers import PretrainedConfig
    require(torch.__version__ == '2.13.0+cu130', 'Wrong PyTorch build')
    require(torch.version.cuda == '13.0', 'Wrong PyTorch CUDA family')
    print('Installed versions, patch hashes, PyTorch and Transformers imports verified.')


def main():
    mode, root = sys.argv[1:3]
    bundle = pathlib.Path(root).resolve()
    if mode == 'wheel':
        check_wheel(bundle)
    elif mode == 'installed':
        check_installed(bundle)
    elif mode == 'bundle':
        manifest = json.loads((bundle / 'manifest.json').read_text())
        check_platform(manifest)
        for name, digest in manifest['files'].items():
            rel = pathlib.PurePosixPath(name)
            require(not rel.is_absolute() and '..' not in rel.parts, 'Invalid manifest path')
            require(sha(bundle / name) == digest, 'Bundle checksum mismatch: ' + name)
        wheels = sorted(p.name for p in (bundle / 'wheels').iterdir())
        require(wheels == sorted(pathlib.PurePosixPath(n).name for n in manifest['files'] if n.startswith('wheels/')),
                'Unexpected files in wheelhouse')
        check_wheel(bundle)
        print('Bundle checksums and build/runtime ABI verified.')
    else:
        raise SystemExit('Expected wheel, installed, or bundle')


if __name__ == '__main__':
    main()
PENNY_EMBEDDED_FILE_5

cat > "$STAGE/make_cuda_overlay.py" <<'PENNY_EMBEDDED_FILE_6'
"""Assemble split CUDA 13 wheels into a writable, nvcc-compatible toolkit root."""
import importlib.metadata as md
import pathlib
import subprocess
import sys

root = pathlib.Path(sys.argv[1]).absolute()
if root.exists():
    raise SystemExit('CUDA overlay already exists: ' + str(root))
root.mkdir()
count = 0
for dist in sorted(md.distributions(), key=lambda d: d.metadata['Name']):
    name = dist.metadata['Name'].lower().replace('_', '-')
    if not name.startswith('nvidia-'):
        continue
    for entry in dist.files or ():
        parts = entry.parts
        if len(parts) < 4 or parts[0] != 'nvidia':
            continue
        # CUDA 13 uses nvidia/cu13/{bin,include,lib,nvvm}; support legacy
        # per-component layouts too. Exclude package Python metadata.
        category = parts[2]
        if category not in ('bin', 'include', 'lib', 'lib64', 'nvvm'):
            continue
        source = pathlib.Path(dist.locate_file(entry)).absolute()
        if not source.is_file():
            continue
        dest = root / ('lib64' if category == 'lib' else category) / pathlib.Path(*parts[3:])
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_symlink() or dest.exists():
            # Namespace wheel overlap is acceptable only when bytes agree.
            if dest.read_bytes() != source.read_bytes():
                raise SystemExit('Conflicting CUDA wheel files: ' + str(dest))
        else:
            dest.symlink_to(source)
            count += 1
(root / 'lib').symlink_to('lib64')
required = ['bin/nvcc', 'bin/ptxas', 'include/cuda.h', 'include/cuda_runtime.h',
            'include/crt/host_config.h', 'nvvm/libdevice/libdevice.10.bc']
for name in required:
    if not (root / name).is_file():
        raise SystemExit('CUDA toolkit wheel missing: ' + name)
if not (root / 'include/cccl/cuda/std').is_dir():
    raise SystemExit('CUDA CCCL headers missing')
version = subprocess.check_output([str(root / 'bin/nvcc'), '--version'], text=True)
if 'release 13.0,' not in version:
    raise SystemExit('Expected CUDA 13.0 nvcc: ' + version)
print(f'CUDA 13.0 toolkit overlay ready ({count} links): {root}')
PENNY_EMBEDDED_FILE_6

cat > "$STAGE/install_offline.sh" <<'PENNY_EMBEDDED_FILE_7'
#!/usr/bin/env bash
# Run with the SAME Kaggle image Python used to build the wheels.
# Does not replace notebook/global packages; installs into an isolated venv.
set -euo pipefail
BUNDLE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV="${1:-/kaggle/working/penny-venv}"
PYTHON="${PYTHON:-python}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
fail() { echo "ERROR: $*" >&2; exit 1; }
for cmd in gcc g++; do command -v "$cmd" >/dev/null || fail "Missing $cmd in Kaggle image (required for JIT)"; done
[[ ! -e "$VENV" ]] || fail "VENV already exists: $VENV. Choose a fresh path."
"$PYTHON" -S "$BUNDLE/verify_bundle.py" bundle "$BUNDLE"
# Run uv straight from its wheel, avoiding ensurepip and an online bootstrap.
UV_BIN="$("$PYTHON" -S - "$BUNDLE" "$VENV" <<'PY'
import os, pathlib, sys, zipfile
b, v = map(pathlib.Path, sys.argv[1:])
v = v.absolute()
v.parent.mkdir(parents=True, exist_ok=True)
uv = list((b/'wheels').glob('uv-*.whl'))
assert len(uv) == 1, 'Expected exactly one uv wheel'
dest = v.parent / (v.name + '.uv-bootstrap')
assert not dest.exists(), 'Bootstrap file already exists: ' + str(dest)
with zipfile.ZipFile(uv[0]) as z:
    names = [n for n in z.namelist() if n.endswith('/scripts/uv') or n == 'uv/uv']
    assert len(names) == 1, 'Cannot locate uv binary in wheel'
    with dest.open('xb') as f:
        f.write(z.read(names[0]))
dest.chmod(0o755)
print(dest)
PY
)"
trap 'rm -f -- "$UV_BIN"' EXIT
export UV_OFFLINE=1 UV_NO_CACHE=1 UV_PYTHON_DOWNLOADS=never
export PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
"$UV_BIN" --no-config venv --python "$(command -v "$PYTHON")" "$VENV"
"$UV_BIN" --no-config pip install --python "$VENV/bin/python" \
  --offline --no-index --no-build --no-cache --find-links "$BUNDLE/wheels" \
  -r "$BUNDLE/requirements.lock"
"$VENV/bin/python" -m pip check
"$VENV/bin/python" "$BUNDLE/verify_bundle.py" installed "$BUNDLE"
"$VENV/bin/python" "$BUNDLE/make_cuda_overlay.py" "$VENV/cuda"
echo "Installed. In the shell that starts the server, run:"
printf '  source %q %q\n' "$BUNDLE/runtime_env.sh" "$VENV"
echo 'Use local model/draft/tokenizer paths; model weights are separate datasets.'
PENNY_EMBEDDED_FILE_7

cat > "$STAGE/runtime_env.sh" <<'PENNY_EMBEDDED_FILE_8'
#!/usr/bin/env bash
# Source AFTER installing. This must run in the shell that starts SGLang.
#   source /kaggle/input/MY_DATASET/runtime_env.sh /kaggle/working/penny-venv
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo 'Source this file in the shell that will launch SGLang.' >&2
  exit 1
fi
_penny_venv="${1:-/kaggle/working/penny-venv}"
if [[ ! -x "$_penny_venv/bin/python" || ! -x "$_penny_venv/cuda/bin/nvcc" ]]; then
  echo "Install the offline bundle first: $_penny_venv" >&2
  return 1
fi
export VIRTUAL_ENV="$_penny_venv"
export CUDA_HOME="$_penny_venv/cuda" CUDA_PATH="$_penny_venv/cuda"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$_penny_venv/bin:$CUDA_HOME/bin:$PATH"
export CC="${CC:-$(command -v gcc)}" CXX="${CXX:-$(command -v g++)}"
export CUDAHOSTCXX="$CXX" NVCC_CCBIN="$CXX"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CPLUS_INCLUDE_PATH="$CUDA_HOME/include:$CUDA_HOME/include/cccl${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export TORCH_CUDA_ARCH_LIST=12.0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export SGLANG_ENABLE_SM120_LOWM_BF16_GEMM=1
export SGLANG_SM120_ONLINE_MXFP8=0
export SGLANG_MAMBA_PREFILL_FINAL_ONLY="${SGLANG_MAMBA_PREFILL_FINAL_ONLY:-1}"
export SGLANG_MAMBA_REFRESH_ON_UNLOCK="${SGLANG_MAMBA_REFRESH_ON_UNLOCK:-1}"
export SGLANG_MAMBA_CONV_DTYPE=bfloat16
export SGLANG_MM_PREPROCESS_DEVICE=cpu
export PENNY_CACHE_BASE="${PENNY_CACHE_BASE:-${TMPDIR:-/tmp}/pennyroyal-d00d88efc8d6}"
export XDG_CACHE_HOME="$PENNY_CACHE_BASE/xdg"
export HF_HOME="$PENNY_CACHE_BASE/hf"
export TRITON_CACHE_DIR="$PENNY_CACHE_BASE/triton"
export CUDA_CACHE_PATH="$PENNY_CACHE_BASE/cuda"
export FLASHINFER_WORKSPACE_BASE="$PENNY_CACHE_BASE/flashinfer"
export TORCH_EXTENSIONS_DIR="$PENNY_CACHE_BASE/torch-extensions"
export SGLANG_CACHE_DIR="$PENNY_CACHE_BASE/sglang" SGLANG_JIT_CACHE_DIR="$PENNY_CACHE_BASE/sglang/jit"
export TORCHINDUCTOR_CACHE_DIR="$PENNY_CACHE_BASE/torchinductor"
export MAX_JOBS="${MAX_JOBS:-4}" FLASHINFER_NINJA_JOBS="${FLASHINFER_NINJA_JOBS:-4}"
export FLASHINFER_NVCC_THREADS=1
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" \
  "$FLASHINFER_WORKSPACE_BASE" "$TORCH_EXTENSIONS_DIR"
unset _penny_venv
PENNY_EMBEDDED_FILE_8

cat > "$STAGE/README.txt" <<'PENNY_EMBEDDED_FILE_9'
Pennyroyal v2.5.3: offline Kaggle wheelhouse
=========================================

Source: jpezzulli/sglang-rtxpro6000
Revision: d00d88efc8d6281b12be4f4073126aec95038c55
Includes the upstream native-MTP checkpoint-boundary correction from v2.5.3.

Included patches (INT4 launcher, sparse prefill v2.2, and bounded prefetch):
1. Gabriel 0004 low-M BF16 GEMM, ported to this exact Pennyroyal revision.
2. KV budget correction from 0002: excludes intermediate SSM scratch that
   RecoverSSM does not allocate. Use --gdn-mtp-cache-mode none as before.
3. GPTQ Marlin MoE scale allocations follow params_dtype. This fixes the
   FP16-scale/BF16-activation assertion. Loading FP16 scales into BF16 can
   round values; it is not a lossless representation change.
4. Sparse prefill v2.2: ordinary intermediate prefill chunks do not publish
   shared checkpoints. Requests starting with zero cached tokens may publish
   one initial branching checkpoint; cache-hit requests skip that insertion.
   Final prefill and generation-end checkpointing are retained. This policy
   targets the regular FULL+MAMBA extra_buffer configuration used here.
   v2.2 also refreshes the Mamba checkpoint to MRU before its final device
   lock is released, including the prefill checkpoint held during decode.
   This refresh is independent of sparse prefill and works with lazy mode.
   It leaves environ.py and all low-M file hashes unchanged.
5. One-shard checkpoint prefetch lookahead. Enabled by default when using
   --weight-loader-prefetch-checkpoints. Reads disjoint contiguous byte ranges
   of a shard using --weight-loader-prefetch-num-threads (default 4), waits
   for the whole shard, and prefetches only its immediate successor while
   loading tensors. Uses the loader's file order; forces the serial mmap
   tensor loader and defers independent startup staging. Cache-drop settings
   are preserved. Disable your external prefetcher. This requires the standard
   mmap safetensors loader, not FASTSAFETENSORS or --weight-loader-disable-mmap.
   Set SGLANG_WEIGHT_LOADER_PREFETCH_LOOKAHEAD=0 for legacy eager prefetch.
   Its environ.py change is verified on top of patch 1's exact output.

This is a runtime/dependency bundle, not a model bundle. It serves both the
Intel BF16-PLE checkpoint and the current albucino FP8-PLE checkpoint with
appropriate local model configuration. It does not switch or quantize the
draft. Keep the current separately configured compressed-tensors INT4 g32
draft and its SGLang-compatible expert target/ignore patterns.

Build online
------------
Run build_bundle_pennyroyal.sh inside the same Kaggle GPU Docker image used
by the offline notebook. Prefer a fixed image digest to a moving latest tag.
Do not run Docker with --network=none for the build: sources, Rust crates,
Python dependencies and FlashInfer kernel wheels must first be downloaded.
No GPU or NVIDIA Docker passthrough is needed for building the wheelhouse.

Default output: /out/bundle-pennyroyal
Default work/cache directory: /build/pennyroyal
BUILD_JOBS defaults to 4. PYTHON defaults to the image's python interpreter.
The Python major/minor, SOABI, architecture and glibc are recorded and checked.
No replacement Python interpreter is downloaded or bundled.

All five patches are mandatory and source/hash guarded. A pre-existing
output bundle is not overwritten; choose a different OUT to build another.
The builder does not publish bundle-pennyroyal until validation succeeds.
If it fails, it prints the staging directory; retain the Docker container
to reuse its caches while investigating. Sufficient disk is needed for
the source, wheelhouse, resolver venv and a second offline validation venv.

If SGLang built successfully but dependency resolution, downloads or final
validation failed, replace the builder and resume in the same container:

  RESUME_STAGE=/out/.penny-bundle.XXXXXXXX bash /out/build_bundle_pennyroyal.sh

Resume is only for a stage built with v2.5.3, prefill v2.2, and lookahead.
An existing four-patch v2.5.3 wheel must be rebuilt with this builder.
It cannot upgrade a v2.5.2-based or older wheelhouse. For this release upgrade,
run an ordinary build with new OUT and WORK directories. Within v2.5.3,
retain WORK to reuse build caches after a failed build.
The wheel version is 0.5.19+gd00d88efc8d6; source-hash checks verify all five patches.

Use the exact partial directory printed by the failed build. This mode
refreshes the bundled helpers, rechecks the patched wheel, regenerates the
manifest and performs a new offline installation. It never rebuilds SGLang.
If dependency resolution/downloads did not finish, it reruns those steps in
a fresh resolver venv, reusing package-manager caches and any saved lock.
This requires the original WORK/source checkout. If downloads and asset
staging already finished, it skips the online dependency steps entirely.
With non-default paths, set OUT and WORK to the original values.

uv HTTP timeout defaults to 300 seconds and retries to 10; pip uses the same
values. Override UV_HTTP_TIMEOUT, UV_HTTP_RETRIES, PIP_DEFAULT_TIMEOUT and
PIP_RETRIES if needed. These tolerate slow/intermittent indexes; a persistent
network outage still needs to be resolved before the online build can finish.
uv's generated wheels/.gitignore marker is removed before wheel validation;
other non-wheel artifacts still cause an error.

The following are checked automatically at build time:
- Every patch's exact resulting bytes inside the built SGLang wheel.
- Rust extensions and C++/CUDA JIT sources are present in the wheel.
- Exactly one wheel per pinned dependency; no source archives.
- A fresh venv installation using --offline --no-index --no-build --no-cache.
- pip check, installed package versions, patch hashes and Torch/Transformers imports.
- A CUDA 13.0 nvcc compile targeting sm_120, including CUDA and CCCL headers.

The build does not execute GPU kernels or run model inference. FlashInfer's
0.6.17+cu130 JIT-cache and 0.6.17 cubin wheels are mandatory; other supported
kernels can still compile locally from their packaged sources at first use.
Optional workflows that download additional models or remote code are not
covered. No NIXL/POSIX source build or NVMe PLE reader is included, matching
our current launcher with those paths disabled.

Upload
------
Upload the contents of bundle-pennyroyal as a Kaggle dataset. Include the
whole directory, not just wheels: install_offline.sh, runtime_env.sh,
verify_bundle.py, make_cuda_overlay.py, requirements.lock, manifest.json,
patches, configs and hot_tokens_64k.pt are also used or retained for audit.
If needed, archive it with zip -r -0 or tar; wheels are already compressed.
Upload checkpoint, tokenizer, processor/chat-template files and current
INT4 draft separately. Existing absolute symlinks from another machine are
not portable: recreate the draft's local view using Kaggle dataset paths.

Install in the offline notebook
-------------------------------
Use a bash cell, replacing the dataset path as necessary:

  bash /kaggle/input/YOUR_BUNDLE/install_offline.sh /kaggle/working/penny-venv

If the dataset contains a bundle-pennyroyal subdirectory, include it above.
The installer uses uv bundled inside a wheel; it needs neither internet
nor ensurepip. It does not alter the notebook's global Python environment.
It requires gcc/g++ from the Kaggle image for local runtime JIT compilation.

Start the server from the same bash cell/shell that sources the environment:

  source /kaggle/input/YOUR_BUNDLE/runtime_env.sh /kaggle/working/penny-venv
  sglang serve ...your existing server arguments with local paths...

Sourcing an environment in one notebook ! command will not persist it into
another ! command. Use one %%bash cell or a wrapper script. The runtime_env
sets CUDA_HOME to the installed CUDA 13.0 wheel toolkit, low-M on, online
MXFP8 off, BF16 Mamba convolution and CPU image preprocessing. Compiler
caches default to /tmp; set PENNY_CACHE_BASE to another writable location
before sourcing if desired. Sparse prefill and unlock refresh are enabled by default.
Set SGLANG_MAMBA_REFRESH_ON_UNLOCK=0 to disable only the v2.2 LRU change.
Set both SGLANG_MAMBA_PREFILL_FINAL_ONLY=0 and SGLANG_MAMBA_REFRESH_ON_UNLOCK=0
before starting SGLang for the original runtime behavior.
Startup reports: Mamba sparse-prefill v2: one cold-start branch plus final checkpoint.
With unlock refresh enabled: Mamba LRU: refresh checkpoint before final device unlock.
The manifest records mamba_prefill_policy_version=2.2. The newest Intel
notebook launcher detects this policy directly in the installed wheel.
The server still needs a compatible NVIDIA
driver and the RTX PRO 6000 visible in Kaggle.

Retain the serving choices from our current launcher, including:
  --quantization auto-round
  --dtype bfloat16
  --moe-runner-backend auto
  --kv-cache-dtype fp8_e4m3
  --speculative-draft-model-path /LOCAL/ADAPTED/INT4/DRAFT
  --speculative-draft-model-quantization compressed-tensors
  --gdn-mtp-cache-mode none
  --speculative-token-map /kaggle/input/YOUR_BUNDLE/hot_tokens_64k.pt

Those are reminders, not a complete launch command. Keep CPU PLE offload,
the existing MTP settings and lossless acceptance thresholds. Use the
checkpoint's own local chat template. The old online serve script's setup
and download actions cannot run offline; invoke the installed server from
your notebook using local paths instead. This builder does not alter the
number of streams, KV budget or model/draft checkpoint metadata.

BF16 PLE alone consumes about 95.4 GiB of host RAM. The larger Kaggle host
needs loading/image-processing headroom beyond that; the 90 GB host does
not fit it. FP8 PLE remains supported by this same bundle.
PENNY_EMBEDDED_FILE_9


if [[ "$MODE" != build ]]; then
  # Fail before network activity if the saved wheel has stale/missing patches.
  "$PYTHON" -S "$STAGE/verify_bundle.py" wheel "$STAGE"
fi
if [[ "$MODE" != finalize ]]; then
  if ! command -v uv >/dev/null; then
    "$PYTHON" -m pip install uv || "$PYTHON" -m pip install --break-system-packages uv
  fi
fi

# ---- pinned source; retain original HEAD so patch guards are meaningful ----
if [[ "$MODE" == build ]]; then
if [[ ! -d "$SRC" ]]; then
  git init -q "$SRC"
  git -C "$SRC" remote add origin "$PENNY_REPO"
fi
[[ -d "$SRC/.git" ]] || fail 'Source path is not a Git checkout'
if ! git -C "$SRC" rev-parse --verify HEAD >/dev/null 2>&1; then
  git -C "$SRC" fetch --depth 1 origin "$PENNY_SHA"
  git -C "$SRC" checkout --detach FETCH_HEAD
fi
[[ "$(git -C "$SRC" rev-parse HEAD)" == "$PENNY_SHA" ]] || fail 'Wrong source revision; choose a fresh WORK directory'
# Check the final hash set before applying: patch 5 extends environ.py from
# patch 1, so replaying patch 1 against a fully patched tree would rightly fail.
"$PYTHON" -S - "$SRC" "$STAGE" <<'PY_APPLY_PATCHES'
import hashlib, importlib.util, pathlib, subprocess, sys
src, bundle = map(pathlib.Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location('verify', bundle/'verify_bundle.py')
v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
expected = v.load_patches(bundle)
def matches_final():
    for name, digest in expected.items():
        path = src/'python'/name
        assert not any(p.is_symlink() for p in (path, *path.parents)), 'Symlink in patched source path: '+name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            return False
    return True
if matches_final():
    print('All five patches already present; exact final source hashes verified.')
else:
    for helper in sorted((bundle/'patches').glob('*.py')):
        subprocess.run([sys.executable, str(helper), str(src), 'apply'], check=True)
    assert matches_final(), 'Final five-patch source hash mismatch'
PY_APPLY_PATCHES
# Reject unrelated edits as well as partial/stale patches.
"$PYTHON" -S - "$SRC" "$STAGE" <<'PY_SOURCE'
import importlib.util, pathlib, subprocess, sys
src, bundle = map(pathlib.Path, sys.argv[1:])
spec=importlib.util.spec_from_file_location('verify', bundle/'verify_bundle.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
allowed={'python/'+p for p in v.load_patches(bundle)}
changed=set(subprocess.check_output(['git','-C',str(src),'diff','--name-only','HEAD'],text=True).splitlines())
untracked=set(subprocess.check_output(['git','-C',str(src),'ls-files','--others','--exclude-standard'],text=True).splitlines())
assert not ((changed|untracked)-allowed), 'Unexpected source edits: '+str((changed|untracked)-allowed)
PY_SOURCE

# ---- build tools; use the image interpreter, never download another Python ----
export CARGO_HOME="${CARGO_HOME:-$WORK/cargo}" RUSTUP_HOME="${RUSTUP_HOME:-$WORK/rustup}"
export PATH="$CARGO_HOME/bin:$PATH"
if ! command -v cargo >/dev/null; then
  curl --fail --silent --show-error --location --retry 3 \
    https://sh.rustup.rs -o "$WORK/rustup-init.sh"
  sh "$WORK/rustup-init.sh" -y --profile minimal --no-modify-path
fi
cargo --version
export SGLANG_BUILD_RUST_EXTS=all CARGO_BUILD_JOBS="$BUILD_JOBS"
export CARGO_PROFILE_RELEASE_DEBUG=0 CARGO_INCREMENTAL=0
export MAX_JOBS="$BUILD_JOBS" CMAKE_BUILD_PARALLEL_LEVEL="$BUILD_JOBS"
# Keep native Rust extensions portable across x86_64 build and Kaggle CPUs.
export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=x86-64"
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG="$SGL_VERSION"
uv --no-config build --wheel "$SRC/python" --python "$PYTHON" -o "$STAGE/wheels"
"$PYTHON" -S "$STAGE/verify_bundle.py" wheel "$STAGE"

fi

if [[ "$MODE" != finalize ]]; then
if [[ "$MODE" == resolve ]]; then
  echo "=== resuming dependencies: $STAGE (reusing verified SGLang wheel) ==="
  [[ -d "$SRC/.git" ]] || fail "Dependency resume needs the original $SRC checkout; run in the same container with the same WORK"
  [[ "$(git -C "$SRC" rev-parse HEAD)" == "$PENNY_SHA" ]] || fail 'Resume source revision mismatch'
  for patch in "$STAGE"/patches/*.py; do "$PYTHON" -S "$patch" "$SRC" check; done
fi
# Keep already-downloaded dependency versions if resolution previously
# completed. This prevents mixing an old partial download with a newer lock.
RESOLVE_LOCK_ARGS=()
if [[ -e "$STAGE/requirements.lock" || -e "$STAGE/packages.json" ]]; then
  "$PYTHON" -S - "$STAGE" "$SGL_VERSION" <<'PY_RESUME_LOCK'
import json,pathlib,sys
b=pathlib.Path(sys.argv[1])
packages=json.loads((b/'packages.json').read_text())
assert packages.get('sglang') == sys.argv[2], 'Wrong SGLang version in saved dependency lock'
assert (b/'requirements.lock').read_text() == ''.join(f'{n}=={v}\n' for n,v in sorted(packages.items())), 'Saved package metadata and lock disagree'
PY_RESUME_LOCK
  RESOLVE_LOCK_ARGS=(--constraint "$STAGE/requirements.lock")
fi
# ---- resolve in an empty venv, never against Kaggle's global packages ----
RESOLVE="$(mktemp -d "$WORK/resolve.XXXXXXXX")/venv"
uv --no-config venv --python "$PYTHON" "$RESOLVE"
INDEX_ARGS=(--extra-index-url https://download.pytorch.org/whl/cu130
  --extra-index-url https://docs.sglang.ai/whl/cu130/
  --extra-index-url https://pypi.nvidia.com)
cp "$SRC/docker/pennyroyal/constraints.txt" "$STAGE/constraints.txt"
cat >> "$STAGE/constraints.txt" <<'PINS'
# Compatibility bound from our working launcher.
huggingface-hub>=1.5.0,<2.0
# Pin all CUDA compiler/runtime headers to the cu130 family.
nvidia-cuda-nvcc==13.0.88
nvidia-cuda-crt==13.0.88
nvidia-nvvm==13.0.88
nvidia-cuda-runtime>=13.0,<13.1
nvidia-cuda-cccl>=13.0,<13.1
PINS
uv --no-config pip install --python "$RESOLVE/bin/python" \
  --prerelease=allow --index-strategy unsafe-best-match "${INDEX_ARGS[@]}" \
  --constraint "$STAGE/constraints.txt" "${RESOLVE_LOCK_ARGS[@]}" "$STAGE"/wheels/sglang-*.whl \
  pip uv 'setuptools>=80.9' wheel packaging \
  'huggingface-hub>=1.5.0,<2.0' \
  nvidia-cuda-nvcc nvidia-cuda-crt nvidia-nvvm nvidia-cuda-runtime nvidia-cuda-cccl

# Both are mandatory: a network fallback at first inference is not acceptable.
uv --no-config pip install --python "$RESOLVE/bin/python" --no-deps \
  --index-url https://flashinfer.ai/whl/cu130 'flashinfer-jit-cache==0.6.17+cu130'
uv --no-config pip install --python "$RESOLVE/bin/python" --no-deps \
  --index-url https://flashinfer.ai/whl 'flashinfer-cubin==0.6.17'
"$RESOLVE/bin/python" -m pip check
# Source-only CUDA dependencies must also build with the bundled toolkit,
# rather than accidentally selecting the image's older /usr/local/cuda.
"$RESOLVE/bin/python" "$STAGE/make_cuda_overlay.py" "$RESOLVE/cuda"
source "$STAGE/runtime_env.sh" "$RESOLVE"

# Freeze installed names/versions, including the local patched SGLang wheel.
# Do not discard direct URLs silently: their installed version must be findable
# as a wheel below, otherwise the build fails before publishing a bundle.
"$RESOLVE/bin/python" - "$STAGE" <<'PY_LOCK'
import importlib.metadata as md, json, pathlib, re, sys
b=pathlib.Path(sys.argv[1])
packages={re.sub(r'[-_.]+','-',d.metadata['Name']).lower():d.version for d in md.distributions()}
(b/'packages.json').write_text(json.dumps(dict(sorted(packages.items())),indent=2)+'\n')
(b/'requirements.lock').write_text(''.join(f'{n}=={v}\n' for n,v in sorted(packages.items())))
# The local SGLang wheel is already present; the special FlashInfer index is
# supplied explicitly to pip when materializing the rest of the lock.
(b/'dependencies.lock').write_text(''.join(f'{n}=={v}\n' for n,v in sorted(packages.items()) if n!='sglang'))
PY_LOCK

# Build wheels for any source-only dependencies NOW, while online. No sdists
# are allowed in the delivered wheelhouse. The clean offline reinstall below
# also catches metadata drift from locally-built wheels.
"$RESOLVE/bin/python" -m pip wheel --pre --no-deps \
  -r "$STAGE/dependencies.lock" --wheel-dir "$STAGE/wheels" \
  "${INDEX_ARGS[@]}" --extra-index-url https://flashinfer.ai/whl/cu130 \
  --extra-index-url https://flashinfer.ai/whl

cp "$SRC/configs/pennyroyal/frspec/flash-next-64k.pt" "$STAGE/hot_tokens_64k.pt"
"$PYTHON" -S - "$STAGE/hot_tokens_64k.pt" <<'PY_MAP'
import hashlib,pathlib,sys
assert hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest() == 'becfa41d394b86c26c632bea8f3c6ea64bbb76d7b238d8673c06afae21269f25', 'FR-Spec map mismatch'
PY_MAP
cp "$SRC/scripts/pennyroyal/build-env.sh" "$STAGE/configs/upstream-build-env.sh"
git -C "$SRC" diff --binary HEAD > "$STAGE/tracked-source.patch"
else
  echo "=== resuming completed wheels: $STAGE (no source build or downloads) ==="
fi
# Includes the untracked new low-M kernel, which git diff alone omits.
"$PYTHON" -S - "$STAGE" <<'PY_NEW'
import ast,pathlib,sys
b=pathlib.Path(sys.argv[1])
for name in ('01-lowm.py','02-kv-budget.py'):
    tree=ast.parse((b/'patches'/name).read_text())
    patch=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PATCH' for t in n.targets))
    (b/'patches'/name.replace('.py','.patch')).write_text(patch)
PY_NEW

# ---- manifest/checksums: identify the exact source, ABI and dependency set ----
"$PYTHON" -S - "$STAGE" "$PENNY_SHA" "$SGL_VERSION" <<'PY_MANIFEST'
import email.parser,hashlib,json,pathlib,platform,re,sys,sysconfig,zipfile
b=pathlib.Path(sys.argv[1]);packages=json.loads((b/'packages.json').read_text())
assert packages.get('sglang') == sys.argv[3], 'Wrong SGLang version in saved dependency lock'
lock=''.join(f'{n}=={v}\n' for n,v in sorted(packages.items()))
assert (b/'requirements.lock').read_text()==lock, 'Saved package metadata and lock disagree'
# uv build writes this housekeeping file alongside its output. It is not a
# package, nor an sdist. Remove only this known marker; reject other debris.
marker=b/'wheels'/'.gitignore'
if marker.exists() or marker.is_symlink():
    assert marker.is_file() and not marker.is_symlink(), 'Unexpected .gitignore type'
    marker.unlink()
seen={}
for p in (b/'wheels').iterdir():
    assert p.is_file() and p.suffix=='.whl', 'Non-wheel artifact: '+p.name
    with zipfile.ZipFile(p) as z:
        # setuptools vendors dependencies with their own nested dist-info.
        # Only the wheel's root-level dist-info describes this distribution.
        metadata=[n for n in z.namelist()
                  if n.count('/')==1 and n.endswith('.dist-info/METADATA')]
        assert len(metadata)==1, 'Ambiguous wheel metadata: '+p.name
        m=email.parser.BytesParser().parsebytes(z.read(metadata[0]))
    n=re.sub(r'[-_.]+','-',m['Name']).lower();v=m['Version']
    assert n not in seen, 'Duplicate wheel: '+n
    assert packages.get(n)==v, 'Unpinned/wrong wheel: '+p.name
    seen[n]=p.name
assert set(seen)==set(packages), 'Missing wheel(s): '+str(set(packages)-set(seen))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
m={'format':1,'source_repository':'https://github.com/jpezzulli/sglang-rtxpro6000',
   'source_revision':sys.argv[2],'sglang_version':sys.argv[3],
   'patches':['Gabriel 0004 low-M','0002 intermediate SSM budget correction','GPTQ Marlin params_dtype scales','Sparse prefill v2.2: one cold-start branch plus final checkpoint; MRU on final device unlock','One-shard prefetch lookahead with parallel byte-range readers'],
   'mamba_prefill_policy_version':'2.2',
   'mamba_refresh_on_unlock_default':True,
   'checkpoint_prefetch_lookahead_default':True,
   'checkpoint_prefetch_threads_default':4,
   'python_version':list(sys.version_info[:2]),'python_full':sys.version,
   'soabi':sysconfig.get_config_var('SOABI'),'machine':platform.machine(),'libc':platform.libc_ver(),
   'packages':packages,'files':{str(p.relative_to(b)):sha(p) for p in sorted(b.rglob('*'))
       if p.is_file() and p.name not in ('manifest.json','build.log','BUILD_COMPLETE') and '__pycache__' not in p.parts}}
(b/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
PY_MANIFEST

# Exercise the EXACT installer the offline notebook will use. No cached wheels,
# online indexes, build isolation downloads or global site-packages are allowed.
CHECK_ROOT="$(mktemp -d "$WORK/offline-check.XXXXXXXX")"
PYTHON="$PYTHON" bash "$STAGE/install_offline.sh" "$CHECK_ROOT/venv"
# Verify nvcc+headers compile without a GPU; catches mismatched CUDA/CCCL wheels.
(
  source "$STAGE/runtime_env.sh" "$CHECK_ROOT/venv"
  printf '#include <cuda_runtime.h>\n#include <cuda/std/type_traits>\n__global__ void k() {}\n' > "$CHECK_ROOT/check.cu"
  "$CUDACXX" -arch=sm_120 -c "$CHECK_ROOT/check.cu" -o "$CHECK_ROOT/check.o"
)
echo 'Offline wheel install, imports, patch checks and SM120 nvcc compile: PASSED' > "$STAGE/BUILD_COMPLETE"
chmod +x "$STAGE/install_offline.sh"
mv "$STAGE" "$BUNDLE"
trap - ERR
echo "=== COMPLETE: $BUNDLE ==="
du -sh "$BUNDLE/wheels"
echo 'Upload bundle-pennyroyal as a Kaggle dataset. See README.txt inside.'
echo 'Actual GPU kernel execution/model inference still needs testing in Kaggle.'
