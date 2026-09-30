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
