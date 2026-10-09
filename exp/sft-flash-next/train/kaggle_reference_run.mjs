/** Run and collect a pinned native gradient capture through Kaggle Jupyter API.
 *
 * The private bearer URL is read only from --url-file and never written to an
 * artifact. DVC invokes this runner as a stage; this script never pushes DVC.
 */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, '../../..');
const remoteWorker = '/kaggle/working/training-gradient-audit/exp/sft-flash-next/train/kaggle_reference_worker.py';
const remoteRoot = '/kaggle/working/gradient-audit-20261009';

export function resolveAttempt(config, nonce = `${new Date().toISOString().replace(/[^0-9]/g, '').slice(0, 14)}-${crypto.randomBytes(4).toString('hex')}`) {
  if (!/^[0-9]{14}-[0-9a-f]{8}$/.test(nonce)) throw new Error('Invalid attempt identifier');
  return { ...config, attempt_id: nonce,
    remote_output: `${config.remote_output}-attempt-${nonce}`,
    remote_launch: `${config.remote_launch}-attempt-${nonce}`,
    remote_config: `${remoteRoot}/${config.run_id}-attempt-${nonce}-config.json` };
}

export function validateConfig(config) {
  if (config.schema_version !== 1 || config.purpose !== 'diagnostic-gradient-qualification-only') throw new Error('Unsupported diagnostic config');
  if (!/^[a-zA-Z0-9_-]+$/.test(config.run_id)) throw new Error('Invalid run_id');
  if (config.timeout_seconds !== 1200 || config.prompt_tokens !== 15598) throw new Error('Pinned time/token settings differ');
  if (config.reference_script_commit !== 'c7ff17e78776a9888b280531a0c53718efa45c02') throw new Error('Reference commit differs');
  if (config.reference_seed !== 20261009) throw new Error('Native reference seed differs');
  if (config.expected_sha256?.reference_script !== 'f95893baa503ec446d4610878b7d5feb7ee975e9313e282349cac2d8952573a0') throw new Error('Native script hash differs');
  if (config.expected_sha256?.adapter !== '49e0960ba1f5a2435e47180262a27e652dd797397a5a11ab13425ef2cf041b6f') throw new Error('Pinned adapter differs');
  if (config.expected_sha256?.sample !== '48f6f88b7bff3b2be5b823c7394388d6b530e26febd25e4c21d3c0b13fdd49ff') throw new Error('Pinned sample differs');
  if (JSON.stringify(config.environment) !== JSON.stringify({ CUBLAS_WORKSPACE_CONFIG: ':4096:8', PYTORCH_CUDA_ALLOC_CONF: 'expandable_segments:True' })) throw new Error('Pinned environment differs');
  if (!config.remote_output.startsWith('/kaggle/working/gradient-audit-20261009/') ||
      !config.remote_launch.startsWith('/kaggle/working/gradient-audit-20261009/') ||
      config.remote_output === config.remote_launch) throw new Error('Unsafe remote output paths');
  if (!config.local_output.startsWith('exp/sft-flash-next/train/gradient-results/') ||
      config.local_output.includes('..')) throw new Error('Unsafe local output path');
  if (!config.local_metrics.startsWith('exp/sft-flash-next/train/metrics/') ||
      config.local_metrics.includes('..')) throw new Error('Unsafe local metrics path');
  if (!/^[0-9a-f]{8}$/.test(config.expected_loss_float32_bits) ||
      !/^[0-9a-f]{64}$/.test(config.expected_gradients_sha256)) throw new Error('Missing exact reference gate');
  return config;
}

export function sha256(data) { return crypto.createHash('sha256').update(data).digest('hex'); }

function privateBase(urlFile) {
  const value = fs.readFileSync(urlFile, 'utf8').trim().replace(/\/$/, '');
  if (!/^https:\/\//.test(value)) throw new Error('Private Jupyter URL must be HTTPS');
  return value;
}

async function request(base, endpoint, options = {}) {
  let response;
  try { response = await fetch(`${base}${endpoint}`, { signal: AbortSignal.timeout(90000), ...options }); }
  catch { throw new Error('Jupyter request failed; private URL omitted'); }
  if (!response.ok) throw new Error(`Jupyter request failed: HTTP ${response.status}; private URL omitted`);
  return response;
}

async function execute(base, code) {
  const kernels = await (await request(base, '/api/kernels')).json();
  if (kernels.length !== 1) throw new Error(`Expected one Jupyter kernel, found ${kernels.length}`);
  const socketUrl = `${base.replace(/^https:/, 'wss:')}/api/kernels/${kernels[0].id}/channels`;
  const msgId = crypto.randomUUID();
  const session = crypto.randomUUID();
  return await new Promise((resolve, reject) => {
    let output = '';
    let settled = false;
    const socket = new WebSocket(socketUrl);
    const timer = setTimeout(() => { if (!settled) { settled = true; socket.close(); reject(new Error('Jupyter execution timed out')); } }, 90000);
    socket.addEventListener('open', () => socket.send(JSON.stringify({
      header: { msg_id: msgId, username: 'codex', session, date: new Date().toISOString(), msg_type: 'execute_request', version: '5.3' },
      parent_header: {}, metadata: {}, channel: 'shell',
      content: { code, silent: false, store_history: false, user_expressions: {}, allow_stdin: false, stop_on_error: true }
    })));
    socket.addEventListener('message', event => {
      let msg; try { msg = JSON.parse(event.data); } catch { return; }
      if (msg.parent_header?.msg_id !== msgId) return;
      if (msg.msg_type === 'stream') output += msg.content.text;
      if (msg.msg_type === 'error') output += `${msg.content.ename}: ${msg.content.evalue}\n`;
      if (msg.msg_type === 'execute_reply') {
        settled = true; clearTimeout(timer); socket.close();
        if (msg.content.status === 'ok') resolve(output.trim());
        else reject(new Error(`Remote execution failed: ${output.slice(-1000)}`));
      }
    });
    socket.addEventListener('error', () => {
      if (!settled) { settled = true; clearTimeout(timer); reject(new Error('Jupyter WebSocket failed; private URL omitted')); }
    });
  });
}

async function upload(base, remotePath, bytes) {
  const relative = remotePath.replace(/^\/kaggle\/working\//, '');
  if (relative === remotePath) throw new Error('Upload path must be under /kaggle/working');
  const payload = JSON.stringify({ type: 'file', format: 'base64', content: bytes.toString('base64') });
  await request(base, `/api/contents/${relative}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: payload });
}

function pyLiteral(value) { return JSON.stringify(value); }

export function preflightCode(config, workerHash) {
  return `import json,hashlib,subprocess\nfrom pathlib import Path\nc=json.loads(Path(${pyLiteral(config.remote_config)}).read_text())\nwith Path(${pyLiteral(remoteWorker)}).open('rb') as stream: actual=hashlib.file_digest(stream,'sha256').hexdigest()\nassert actual==${pyLiteral(workerHash)}\nassert not Path(c['remote_output']).exists() and not Path(c['remote_launch']).exists()\nr=subprocess.run(['/usr/bin/python3',${pyLiteral(remoteWorker)},'validate',${pyLiteral(config.remote_config)}],capture_output=True,text=True,timeout=60)\nassert r.returncode==0,r.stderr[-1000:]\nprint(json.dumps({'preflight':'ok','run_id':c['run_id'],'attempt_id':c['attempt_id'],'worker_sha256':actual}))`;
}

async function preflight(base, config, workerBytes) {
  await upload(base, remoteWorker, workerBytes);
  await upload(base, config.remote_config, Buffer.from(JSON.stringify(config, null, 2) + '\n'));
  const result = await execute(base, preflightCode(config, sha256(workerBytes)));
  if (!result.includes('"preflight": "ok"')) throw new Error('Remote preflight did not confirm readiness');
  return result;
}

async function launch(base, config) {
  const code = `import subprocess,json\nr=subprocess.run(['/usr/bin/python3',${pyLiteral(remoteWorker)},'launch',${pyLiteral(config.remote_config)}],capture_output=True,text=True,timeout=60)\nprint(json.dumps({'returncode':r.returncode,'stdout':r.stdout[-1500:],'stderr':r.stderr[-1500:]}))\nassert r.returncode==0`;
  return await execute(base, code);
}

async function jsonIfExists(base, remotePath) {
  const relative = remotePath.replace(/^\/kaggle\/working\//, '');
  let response;
  try { response = await fetch(`${base}/files/${relative}`, { signal: AbortSignal.timeout(15000) }); }
  catch { throw new Error('Jupyter status request failed; private URL omitted'); }
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`Jupyter status HTTP ${response.status}; private URL omitted`);
  return await response.json();
}

async function waitForMonitor(base, config) {
  const monitorPath = `${config.remote_launch}/monitor.json`;
  const deadline = Date.now() + (config.timeout_seconds + 150) * 1000;
  let lastUpdate = 0;
  while (true) {
    const status = await jsonIfExists(base, monitorPath);
    if (status) return status;
    if (Date.now() > deadline) throw new Error('Monitor completion deadline exceeded; use --action collect after inspecting remote attempt');
    if (Date.now() - lastUpdate > 60000) {
      console.log(JSON.stringify({ run_id: config.run_id, attempt_id: config.attempt_id, status: 'waiting' }));
      lastUpdate = Date.now();
    }
    await new Promise(resolve => setTimeout(resolve, 10000));
  }
}

export function artifactNames(entries) {
  // Kaggle's Jupytext contents manager reports .py source files as notebooks.
  return entries.filter(entry => ['file', 'notebook'].includes(entry.type)).map(entry => {
    if (path.basename(entry.name) !== entry.name) throw new Error('Unsafe artifact name');
    return entry.name;
  });
}

async function listRemote(base, remoteDirectory) {
  const relative = remoteDirectory.replace(/^\/kaggle\/working\//, '');
  const item = await (await request(base, `/api/contents/${relative}?content=1`)).json();
  if (item.type !== 'directory') throw new Error('Expected remote artifact directory');
  return artifactNames(item.content);
}

async function downloadFile(base, remotePath, localPath) {
  const relative = remotePath.replace(/^\/kaggle\/working\//, '');
  const response = await request(base, `/files/${relative}`);
  const temporary = `${localPath}.part`;
  await pipeline(Readable.fromWeb(response.body), fs.createWriteStream(temporary));
  fs.renameSync(temporary, localPath);
  const bytes = fs.readFileSync(localPath);
  return { bytes: bytes.length, sha256: sha256(bytes) };
}

async function collect(base, config) {
  const monitor = await jsonIfExists(base, `${config.remote_launch}/monitor.json`);
  if (!monitor) throw new Error('Remote monitor has not finished');
  const local = path.join(repoRoot, config.local_output);
  fs.mkdirSync(local, { recursive: true });
  const frozen = JSON.parse(fs.readFileSync(path.join(local, 'frozen-identities.json')));
  if (sha256(fs.readFileSync(path.join(local, 'frozen-runner.mjs'))) !== frozen.runner_sha256 ||
      sha256(fs.readFileSync(path.join(local, 'frozen-worker.py'))) !== frozen.worker_sha256 ||
      sha256(fs.readFileSync(path.join(local, 'frozen-config-template.json'))) !== frozen.config_template_sha256 ||
      sha256(fs.readFileSync(path.join(local, 'frozen-reference.py'))) !== frozen.native_source_sha256)
    throw new Error('Locally frozen runner/source/config bytes changed during capture');
  const files = {};
  for (const [remoteDirectory, prefix] of [[config.remote_output, ''], [config.remote_launch, 'launch-']]) {
    for (const name of await listRemote(base, remoteDirectory)) {
      const localName = `${prefix}${name}`;
      files[localName] = await downloadFile(base, `${remoteDirectory}/${name}`, path.join(local, localName));
    }
  }
  const result = JSON.parse(fs.readFileSync(path.join(local, 'result.json')));
  const summary = JSON.parse(fs.readFileSync(path.join(local, 'gradient-summary.json')));
  const timing = JSON.parse(fs.readFileSync(path.join(local, 'launch-timing-summary.json')));
  const launchRecord = JSON.parse(fs.readFileSync(path.join(local, 'launch-launch.json')));
  const a = Object.entries(summary).filter(([name]) => name.includes('lora_A'));
  const b = Object.entries(summary).filter(([name]) => name.includes('lora_B'));
  const gradients = { a_tensors: a.length, b_tensors: b.length,
    a_nonzero_tensors: a.filter(([, row]) => row.nonzero > 0).length,
    b_nonzero_tensors: b.filter(([, row]) => row.nonzero > 0).length };
  const sourceValid = files['reference.py'].sha256 === config.expected_sha256.reference_script &&
    files['launch-reference.py'].sha256 === config.expected_sha256.reference_script &&
    files['launch-worker.py'].sha256 === launchRecord.worker_sha256 &&
    files['launch-instrumented-bootstrap.py'].sha256 === launchRecord.instrumented_bootstrap_sha256 &&
    files['launch-bootstrap.py'].sha256 === config.expected_sha256.bootstrap &&
    files['launch-sample.pt'].sha256 === config.expected_sha256.sample &&
    files['launch-model-config.json'].sha256 === config.expected_sha256.model_config &&
    files['launch-worker.py'].sha256 === frozen.worker_sha256 &&
    JSON.stringify(launchRecord.expected_sha256) === JSON.stringify(config.expected_sha256) &&
    JSON.stringify(launchRecord.environment) === JSON.stringify(config.environment) &&
    launchRecord.reference_script_commit === config.reference_script_commit &&
    launchRecord.no_optimizer_update === true && launchRecord.no_clipping === true;
  const structuralValid = files['gradients.pt'].sha256 === result.gradients_sha256 &&
      result.gradient_tensors === 744 && Object.keys(summary).length === 744 &&
      Object.values(summary).every(row => row.finite) &&
      a.length === 372 && b.length === 372 && sourceValid &&
      result.optimizer_updates === 0 && result.clipping_applied === false &&
      monitor.returncode === 0 && monitor.timed_out === false;
  const lossBits = Buffer.alloc(4); lossBits.writeFloatBE(result.measured_loss);
  const equality = { expected_loss_float32_bits: config.expected_loss_float32_bits,
    observed_loss_float32_bits: lossBits.toString('hex'),
    expected_gradients_sha256: config.expected_gradients_sha256,
    observed_gradients_sha256: files['gradients.pt'].sha256 };
  equality.loss_exact = equality.expected_loss_float32_bits === equality.observed_loss_float32_bits;
  equality.gradients_file_exact = equality.expected_gradients_sha256 === equality.observed_gradients_sha256;
  const record = { run_id: config.run_id, loss: result.measured_loss,
    raw_gradient_tensors: result.gradient_tensors, gradients_sha256: result.gradients_sha256,
    gradient_counts: gradients,
    no_clipping: true, optimizer_updates: 0, timings: timing,
    monitor, expected_sha256: config.expected_sha256, equality,
    structural_valid: structuralValid,
    source_identities: {
      runner_sha256: frozen.runner_sha256,
      worker_sha256: frozen.worker_sha256,
      config_template_sha256: frozen.config_template_sha256,
      repository_head: frozen.repository_head,
      collector_sha256: sha256(fs.readFileSync(fileURLToPath(import.meta.url))),
      collector_repository_head: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: repoRoot, encoding: 'utf8' }).trim(),
      reference_script_commit: config.reference_script_commit },
    files,
    dvc_stage: 'exp/sft-flash-next/train/dvc.yaml:reference_v0' };
  fs.writeFileSync(path.join(local, 'collector-runner.mjs'), fs.readFileSync(fileURLToPath(import.meta.url)));
  fs.writeFileSync(path.join(local, 'runner-result.json'), JSON.stringify(record, null, 2) + '\n');
  fs.writeFileSync(path.join(local, 'file-hashes.json'), JSON.stringify(files, null, 2) + '\n');
  const metricsPath = path.join(repoRoot, config.local_metrics);
  fs.mkdirSync(path.dirname(metricsPath), { recursive: true });
  fs.writeFileSync(metricsPath, JSON.stringify({ run_id: config.run_id,
    attempt_id: config.attempt_id, loss: record.loss, gradients_sha256: record.gradients_sha256,
    raw_gradient_tensors: record.raw_gradient_tensors, gradient_counts: gradients,
    equality, timings: timing,
    structural_valid: record.structural_valid }, null, 2) + '\n');
  if (!record.structural_valid || !equality.loss_exact || !equality.gradients_file_exact)
    throw new Error(`Native reference equality gate failed; evidence retained at ${local}`);
  return { local, loss: record.loss, equality, structural_valid: record.structural_valid,
    gradients_sha256: record.gradients_sha256, attempt_id: config.attempt_id };
}

export async function main(argv) {
  const options = { action: 'run', config: path.join(here, 'configs/reference-v0.json'), urlFile: '/tmp/kaggle_probe_url' };
  for (let index = 0; index < argv.length; index += 2) {
    const key = argv[index], value = argv[index + 1];
    if (!value || !['--action', '--config', '--url-file'].includes(key)) throw new Error('Use --action/--config/--url-file with values');
    options[{ '--action': 'action', '--config': 'config', '--url-file': 'urlFile' }[key]] = value;
  }
  if (!['preflight', 'launch', 'collect', 'run'].includes(options.action)) throw new Error('Invalid action');
  const templateBytes = fs.readFileSync(options.config);
  const template = validateConfig(JSON.parse(templateBytes));
  const local = path.join(repoRoot, template.local_output);
  let config;
  let workerBytes;
  if (options.action === 'collect') {
    config = JSON.parse(fs.readFileSync(path.join(local, 'resolved-config.json')));
    const savedTemplate = JSON.parse(fs.readFileSync(path.join(local, 'frozen-config-template.json')));
    validateConfig(savedTemplate);
    if (JSON.stringify(resolveAttempt(savedTemplate, config.attempt_id)) !== JSON.stringify(config))
      throw new Error('Saved resolved attempt differs from frozen template');
    workerBytes = fs.readFileSync(path.join(local, 'frozen-worker.py'));
  } else {
    config = resolveAttempt(template);
    workerBytes = fs.readFileSync(path.join(here, 'kaggle_reference_worker.py'));
    if (options.action !== 'preflight') {
      fs.mkdirSync(local, { recursive: false });
      fs.writeFileSync(path.join(local, 'resolved-config.json'), JSON.stringify(config, null, 2) + '\n');
      fs.writeFileSync(path.join(local, 'frozen-config-template.json'), templateBytes);
      fs.writeFileSync(path.join(local, 'frozen-worker.py'), workerBytes);
      const runnerBytes = fs.readFileSync(fileURLToPath(import.meta.url));
      fs.writeFileSync(path.join(local, 'frozen-runner.mjs'), runnerBytes);
      const nativeBytes = fs.readFileSync(path.join(here, 'overfit_hf_reference.py'));
      if (sha256(nativeBytes) !== config.expected_sha256.reference_script) throw new Error('Local native source hash differs');
      fs.writeFileSync(path.join(local, 'frozen-reference.py'), nativeBytes);
      const identities = { runner_sha256: sha256(runnerBytes), worker_sha256: sha256(workerBytes),
        config_template_sha256: sha256(templateBytes), native_source_sha256: sha256(nativeBytes),
        repository_head: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: repoRoot, encoding: 'utf8' }).trim(),
        reference_script_commit: config.reference_script_commit };
      fs.writeFileSync(path.join(local, 'frozen-identities.json'), JSON.stringify(identities, null, 2) + '\n');
    }
  }
  const base = privateBase(options.urlFile);
  if (options.action === 'preflight') return console.log(await preflight(base, config, workerBytes));
  if (options.action === 'collect') return console.log(JSON.stringify(await collect(base, config)));
  console.log(await preflight(base, config, workerBytes));
  console.log(await launch(base, config));
  if (options.action === 'launch') return;
  console.log(JSON.stringify(await waitForMonitor(base, config)));
  console.log(JSON.stringify(await collect(base, config)));
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  main(process.argv.slice(2)).catch(error => { console.error(error.message); process.exitCode = 1; });
}
