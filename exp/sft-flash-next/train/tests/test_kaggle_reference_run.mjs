import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { validateConfig, resolveAttempt, preflightCode, artifactNames } from '../kaggle_reference_run.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const config = JSON.parse(fs.readFileSync(path.join(here, '../configs/reference-v0.json')));

test('v0 config pins test-only source, sample, adapter and loss/gradient gate', () => {
  assert.equal(validateConfig(config), config);
  assert.equal(config.expected_loss_float32_bits, '3f202d90');
  assert.equal(config.expected_gradients_sha256.length, 64);
  assert.equal(config.local_metrics, 'exp/sft-flash-next/train/metrics/reference-v0.json');
});

test('each invocation gets unique remote output while DVC local output remains stable', () => {
  const first = resolveAttempt(config, '20261009170000-1234abcd');
  const second = resolveAttempt(config, '20261009170001-1234abcd');
  assert.notEqual(first.remote_output, second.remote_output);
  assert.notEqual(first.remote_launch, second.remote_launch);
  assert.notEqual(first.remote_config, second.remote_config);
  assert.equal(first.local_output, second.local_output);
  assert.match(preflightCode(first, 'abc'), /not Path\(c\['remote_output'\]\)\.exists\(\)/);
});

test('unsafe edits fail before server access', () => {
  assert.throws(() => validateConfig({ ...config, timeout_seconds: 0 }), /time\/token/);
  assert.throws(() => validateConfig({ ...config, local_output: '../escape' }), /Unsafe local/);
  assert.throws(() => validateConfig({ ...config, expected_sha256: { ...config.expected_sha256, adapter: 'bad' } }), /adapter/);
});

test('Kaggle Jupytext Python sources are collected as raw files', () => {
  assert.deepEqual(artifactNames([
    { name: 'reference.py', type: 'notebook' },
    { name: 'gradients.pt', type: 'file' },
    { name: 'nested', type: 'directory' },
  ]), ['reference.py', 'gradients.pt']);
  assert.throws(() => artifactNames([{ name: '../escape', type: 'file' }]), /Unsafe artifact/);
});

test('target-mask config pins candidate sources and retains native numerical gate', () => {
  const candidate = JSON.parse(fs.readFileSync(path.join(here, '../configs/target-mask-v1.json')));
  assert.equal(validateConfig(candidate), candidate);
  assert.equal(candidate.expected_gradients_sha256, config.expected_gradients_sha256);
  assert.throws(() => validateConfig({ ...candidate, candidate_sources_sha256: {} }), /Missing target-mask/);
});

test('Liger diagnostic pins wheel and comparison while preserving exact reference identity', () => {
  const candidate = JSON.parse(fs.readFileSync(path.join(here, '../configs/liger-flce-v2.json')));
  assert.equal(validateConfig(candidate), candidate);
  assert.equal(candidate.expected_gradients_sha256, config.expected_gradients_sha256);
  assert.throws(() => validateConfig({ ...candidate, liger_dependency: {} }), /pinned Liger/);
  assert.throws(() => validateConfig({ ...candidate, native_gradients: '/tmp/other.pt' }), /pinned Liger/);
});

test('CCE exact diagnostic pins official source archive and native gradient comparison', () => {
  const candidate = JSON.parse(fs.readFileSync(path.join(here, '../configs/cce-exact-v3.json')));
  assert.equal(validateConfig(candidate), candidate);
  assert.equal(candidate.expected_gradients_sha256, config.expected_gradients_sha256);
  assert.throws(() => validateConfig({ ...candidate, cce_dependency: {} }), /pinned CCE/);
  assert.throws(() => validateConfig({ ...candidate, native_gradients: '/tmp/other.pt' }), /pinned CCE/);
});

test('Opt3 pins the saved CCE baseline and every attention patch source', () => {
  const candidate = JSON.parse(fs.readFileSync(path.join(here, '../configs/cce-opt3-mask-v4.json')));
  assert.equal(validateConfig(candidate), candidate);
  assert.throws(() => validateConfig({ ...candidate, expected_gradients_sha256: config.expected_gradients_sha256 }), /pinned Opt3/);
  assert.throws(() => validateConfig({ ...candidate, native_gradients: '/tmp/other.pt' }), /pinned CCE/);
  const sources = { ...candidate.candidate_sources_sha256 };
  delete sources['opt3_mask_capture.py'];
  assert.throws(() => validateConfig({ ...candidate, candidate_sources_sha256: sources }), /pinned Opt3/);
});

test('Opt4 pins Opt3 baseline and disk/preparation implementation sources', () => {
  const candidate = JSON.parse(fs.readFileSync(path.join(here, '../configs/cce-opt4-ple-v5.json')));
  assert.equal(validateConfig(candidate), candidate);
  assert.throws(() => validateConfig({ ...candidate, expected_loss_float32_bits: '3f20282a' }), /pinned Opt4/);
  const sources = { ...candidate.candidate_sources_sha256 };
  delete sources['ple_preparation.py'];
  assert.throws(() => validateConfig({ ...candidate, candidate_sources_sha256: sources }), /pinned Opt4/);
  assert.throws(() => validateConfig({ ...candidate, native_gradients: '/tmp/other.pt' }), /pinned CCE/);
});
