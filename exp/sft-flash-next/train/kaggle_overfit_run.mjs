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
import { validateConfig as validateGradientConfig } from './kaggle_reference_run.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, '../../..');
const remoteWorker = '/kaggle/working/training-gradient-audit/exp/sft-flash-next/train/kaggle_overfit_worker.py';
const remoteRoot = '/kaggle/working/gradient-audit-20261009';

export function resolveAttempt(config, nonce = `${new Date().toISOString().replace(/[^0-9]/g, '').slice(0, 14)}-${crypto.randomBytes(4).toString('hex')}`) {
  if (!/^[0-9]{14}-[0-9a-f]{8}$/.test(nonce)) throw new Error('Invalid attempt identifier');
  return { ...config, attempt_id: nonce,
    remote_output: `${config.remote_output}-attempt-${nonce}`,
    remote_launch: `${config.remote_launch}-attempt-${nonce}`,
    remote_config: `${remoteRoot}/${config.run_id}-attempt-${nonce}-config.json` };
}

export function validateConfig(config) {
  validateGradientConfig({ ...config, purpose: 'diagnostic-gradient-qualification-only', timeout_seconds: 1200 });
  if (config.purpose !== 'diagnostic-single-sample-overfit' || config.timeout_seconds !== 9000 ||
      config.precision_policy?.implementation !== 'model_activations_native_statistics' ||
      JSON.stringify(config.learning) !== JSON.stringify({initialization:'seeded-random-A-zero-B',seed:20261009,
        optimizer:'torch.optim.AdamW',learning_rate:0.0002,weight_decay:0,clip_grad_norm:1,max_updates:20,
        target_loss_ratio:0.05,acceptance:'finite-learning-and-loss-reduction; gradient equality not required'}))
    throw new Error('Pinned BF16 one-sample learning contract differs');
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
  if (['liger_target_flce', 'cce_target_exact', 'cce_opt3_mask', 'cce_opt4_ple', 'cce_opt5_bf16'].includes(config.objective)) {
    const dep = config.liger_dependency ?? config.cce_dependency;
    const response = await fetch(dep.url, { signal: AbortSignal.timeout(90000) });
    if (!response.ok) throw new Error('Pinned Liger wheel download failed');
    const wheel = Buffer.from(await response.arrayBuffer());
    if (sha256(wheel) !== dep.sha256) throw new Error('Liger wheel checksum differs');
    await upload(base, dep.remote_wheel ?? dep.remote_archive, wheel);
  }
  for (const [name, hash] of Object.entries(config.candidate_sources_sha256 ?? {})) {
    const bytes = fs.readFileSync(path.join(here, name));
    if (sha256(bytes) !== hash) throw new Error(`Local candidate source hash differs: ${name}`);
    await upload(base, `${path.posix.dirname(remoteWorker)}/${name}`, bytes);
  }
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
  const monitor=await jsonIfExists(base, `${config.remote_launch}/monitor.json`);
  if (!monitor) throw new Error('Overfit monitor has not finished');
  const local=path.join(repoRoot,config.local_output);
  const frozen=JSON.parse(fs.readFileSync(path.join(local,'frozen-identities.json')));
  for (const [name,hash] of Object.entries(config.candidate_sources_sha256)) {
    if (sha256(fs.readFileSync(path.join(local,`frozen-candidate-${name}`)))!==hash) throw new Error('Frozen source mismatch');
  }
  for (const [name,key] of [['frozen-runner.mjs','runner_sha256'],['frozen-worker.py','worker_sha256'],
      ['frozen-config-template.json','config_template_sha256'],['frozen-reference.py','native_source_sha256']]) {
    if (sha256(fs.readFileSync(path.join(local,name)))!==frozen[key]) throw new Error('Frozen launcher mismatch');
  }
  const files={};
  for (const [directory,prefix] of [[config.remote_output,''],[config.remote_launch,'launch-']]) {
    for (const name of await listRemote(base,directory)) files[prefix+name]=await downloadFile(base,`${directory}/${name}`,path.join(local,prefix+name));
  }
  const read=name=>JSON.parse(fs.readFileSync(path.join(local,name)));
  const result=read('result.json'),audit=read('learning-audit.json'),precision=read('bf16-policy.json');
  const events=fs.readFileSync(path.join(local,'events.jsonl'),'utf8').trim().split('\n').map(JSON.parse);
  const losses=events.filter(x=>x.event==='loss'),updates=events.filter(x=>x.event==='update');
  const steps=read('launch-timing-summary.json');
  const head=read('target-head.json'),ple=read('ple-preparation.json');
  const original=read('launch-launch.json');
  const sourcesValid=Object.entries(config.candidate_sources_sha256).every(([name,hash])=>files[`launch-candidate-${name}`]?.sha256===hash);
  const structurallyValid=monitor.returncode===0&&!monitor.timed_out&&sourcesValid&&
    original.worker_sha256===frozen.worker_sha256&&files['reference.py'].sha256===config.expected_sha256.reference_script&&
    audit.initialization==='seeded-random-A-zero-B'&&audit.adapter_tensors===744&&audit.all_gradients_finite&&
    audit.completed_updates===result.optimizer_updates&&updates.length===result.optimizer_updates&&
    audit.optimizer_states_finite&&audit.parameters_changed&&audit.all_parameters_bf16&&audit.all_parameters_finite&&
    losses.length===result.optimizer_updates+1&&losses.every(x=>Number.isFinite(x.loss)&&x.loss>=0)&&
    precision.finalized&&precision.saved_tensor_conversion===false&&
    precision.saved_fp32_statistics_compressed===false&&precision.adapter_parameter_dtype_counts['torch.bfloat16']===744&&
    precision.layer_calls.every(x=>x.dtype==='torch.bfloat16')&&precision.cce_fp32_lse.length===losses.length&&
    precision.cce_fp32_lse.every(x=>x.dtype==='torch.float32'&&x.roundtrip_values_exact)&&
    JSON.stringify(precision.saved_original_bytes)===JSON.stringify(precision.saved_storage_bytes)&&
    head.head_calls.length===losses.length&&head.head_calls.every(x=>x.hidden_dtype==='torch.bfloat16'&&x.target_tokens===651)&&
    head.vocabulary_saved_tensor_calls.length===0&&ple.current_retained_through_backward&&ple.workers_shutdown;
  const passed=structurallyValid&&result.passed===true&&result.ratio<=config.learning.target_loss_ratio;
  const record={run_id:config.run_id,attempt_id:config.attempt_id,passed,structurally_valid:structurallyValid,
    criterion:config.learning.acceptance,gradient_equality_required:false,loss_curve:losses.map(({step,loss,ratio})=>({step,loss,ratio})),
    result,learning_audit:audit,timings:steps,execution_commit:frozen.repository_head,
    native_script_commit:config.reference_script_commit,native_source_sha256:config.expected_sha256.reference_script,files};
  fs.writeFileSync(path.join(local,'file-hashes.json'),JSON.stringify(files,null,2)+'\n');
  fs.writeFileSync(path.join(local,'runner-result.json'),JSON.stringify(record,null,2)+'\n');
  fs.writeFileSync(path.join(repoRoot,config.local_metrics),JSON.stringify({...record,files:undefined},null,2)+'\n');
  if(!passed) throw new Error(`One-sample overfit criterion failed; complete evidence retained at ${local}`);
  return {passed,loss_curve:record.loss_curve,optimizer_updates:result.optimizer_updates,ratio:result.ratio,local};
}

export async function main(argv) {
  const options = { action: 'run', config: path.join(here, 'configs/bf16-overfit-v8.json'), urlFile: '/tmp/kaggle_probe_url' };
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
    workerBytes = fs.readFileSync(path.join(here, 'kaggle_overfit_worker.py'));
    if (options.action !== 'preflight') {
      fs.mkdirSync(local, { recursive: false });
      fs.writeFileSync(path.join(local, 'resolved-config.json'), JSON.stringify(config, null, 2) + '\n');
      fs.writeFileSync(path.join(local, 'frozen-config-template.json'), templateBytes);
      fs.writeFileSync(path.join(local, 'frozen-worker.py'), workerBytes);
      for (const [name, checksum] of Object.entries(config.candidate_sources_sha256 ?? {})) {
        const bytes = fs.readFileSync(path.join(here, name));
        if (sha256(bytes) !== checksum) throw new Error(`Candidate source hash differs: ${name}`);
        fs.writeFileSync(path.join(local, `frozen-candidate-${name}`), bytes);
      }
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
