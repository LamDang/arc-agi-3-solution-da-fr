/** Bounded forward-only dtype trace via the private Kaggle Jupyter API. */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
const here=path.dirname(fileURLToPath(import.meta.url));
const root=path.resolve(here,'../../..');
const args=process.argv.slice(2);
const configPath=path.resolve(args[args.indexOf('--config')+1]);
const urlFile=args.includes('--url-file')?args[args.indexOf('--url-file')+1]:'/tmp/kaggle_probe_url';
const base=fs.readFileSync(urlFile,'utf8').trim().replace(/\/$/,'');
const sha=data=>crypto.createHash('sha256').update(data).digest('hex');
async function request(endpoint,options={}) {
  let response;
  try {response=await fetch(base+endpoint,{signal:AbortSignal.timeout(60000),...options});}
  catch {throw new Error('Jupyter connection failed; private URL omitted');}
  if(!response.ok)throw new Error(`Jupyter HTTP ${response.status}; private URL omitted`);
  return response;
}
async function execute(code) {
  const kernels=await(await request('/api/kernels')).json();
  if(kernels.length!==1)throw new Error('Expected one kernel');
  const id=crypto.randomUUID();
  return await new Promise((resolve,reject)=>{
    const ws=new WebSocket(base.replace(/^https:/,'wss:')+`/api/kernels/${kernels[0].id}/channels`);
    let output='',done=false;
    const timer=setTimeout(()=>{done=true;ws.close();reject(new Error('Jupyter execution timeout'));},60000);
    ws.addEventListener('open',()=>ws.send(JSON.stringify({header:{msg_id:id,username:'codex',session:crypto.randomUUID(),date:new Date().toISOString(),msg_type:'execute_request',version:'5.3'},parent_header:{},metadata:{},channel:'shell',content:{code,silent:false,store_history:false,user_expressions:{},allow_stdin:false,stop_on_error:true}})));
    ws.addEventListener('message',event=>{
      const msg=JSON.parse(event.data);if(msg.parent_header?.msg_id!==id)return;
      if(msg.msg_type==='stream')output+=msg.content.text;
      if(msg.msg_type==='error')output+=`${msg.content.ename}: ${msg.content.evalue}`;
      if(msg.msg_type==='execute_reply'){done=true;clearTimeout(timer);ws.close();msg.content.status==='ok'?resolve(output.trim()):reject(new Error(output.slice(-1500)));}
    });
    ws.addEventListener('error',()=>{if(!done){done=true;clearTimeout(timer);reject(new Error('Jupyter WebSocket failed; private URL omitted'));}});
  });
}
async function upload(remote,bytes) {
  const relative=remote.replace(/^\/kaggle\/working\//,'');
  if(relative===remote)throw new Error('Unsafe upload path');
  await request('/api/contents/'+relative,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({type:'file',format:'base64',content:bytes.toString('base64')})});
}
const template=JSON.parse(fs.readFileSync(configPath));
if(template.purpose!=='forward-only-layer-dtype-trace'||template.tokens!==32||template.layers!==4)throw new Error('Unbounded or unknown diagnostic');
const local=path.join(root,template.local_output);fs.mkdirSync(local,{recursive:true});
const resolved=path.join(local,'resolved-config.json');
const remoteSource='/kaggle/working/training-gradient-audit/exp/sft-flash-next/train';
let c;
if(fs.existsSync(resolved)) {
  c=JSON.parse(fs.readFileSync(resolved));
  if(c.template_sha256!==sha(fs.readFileSync(configPath)))throw new Error('Existing diagnostic config differs');
} else {
  const attempt=new Date().toISOString().replace(/[^0-9]/g,'').slice(0,14)+'-'+crypto.randomBytes(4).toString('hex');
  c={...template,attempt_id:attempt,remote_output:template.remote_output+'-attempt-'+attempt,
     git_commit:execFileSync('git',['rev-parse','HEAD'],{cwd:root,encoding:'utf8'}).trim(),
     template_sha256:sha(fs.readFileSync(configPath)),sources_sha256:{}};
  for(const name of template.sources)c.sources_sha256[name]=sha(fs.readFileSync(path.join(here,name)));
  c.remote_config=c.remote_output+'-config.json';
  await execute(`import subprocess\nr=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True)\nassert r.returncode==0 and not r.stdout.strip(),'GPU busy; no diagnostic launched'\nprint('GPU_IDLE')`);
  for(const name of template.sources)await upload(remoteSource+'/'+name,fs.readFileSync(path.join(here,name)));
  await upload(c.remote_config,Buffer.from(JSON.stringify(c,null,2)+'\n'));
  const bootstrap=`import sys,runpy\nsys.path=[p for p in sys.path if p!='/usr/local/lib/python3.13/dist-packages']\nsys.path[:0]=['/tmp/peft-autoround-compat/package','/tmp/peft-autoround-compat/site-without-torchao',${JSON.stringify(remoteSource)}]\nsys.argv=['trace_layer_dtypes.py','--config',${JSON.stringify(c.remote_config)}]\nrunpy.run_path(${JSON.stringify(remoteSource+'/trace_layer_dtypes.py')},run_name='__main__')\n`;
  c.remote_bootstrap=c.remote_output+'-bootstrap.py';
  await upload(c.remote_bootstrap,Buffer.from(bootstrap));
  const launch=await execute(`import subprocess,json,os\nfrom pathlib import Path\nconfig=json.loads(Path(${JSON.stringify(c.remote_config)}).read_text())\nlog=open(config['remote_output']+'-process.log','w')\nenv=dict(os.environ,**config['environment'])\np=subprocess.Popen(['/usr/bin/python3',${JSON.stringify(c.remote_bootstrap)}],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)\nprint(json.dumps({'pid':p.pid}))`);
  c.pid=JSON.parse(launch).pid;
  fs.writeFileSync(resolved,JSON.stringify(c,null,2)+'\n');
  fs.writeFileSync(path.join(local,'config-template.json'),fs.readFileSync(configPath));
  fs.writeFileSync(path.join(local,'runner.mjs'),fs.readFileSync(fileURLToPath(import.meta.url)));
  for(const name of template.sources)fs.writeFileSync(path.join(local,'frozen-'+name),fs.readFileSync(path.join(here,name)));
  console.log(JSON.stringify({launched:c.attempt_id,pid:c.pid,commit:c.git_commit}));
}
const deadline=Date.now()+template.timeout_seconds*1000;
let status;
while(Date.now()<deadline) {
  status=JSON.parse(await execute(`import json,os\nfrom pathlib import Path\nout=Path(${JSON.stringify(c.remote_output)})\ntry:\n os.kill(${c.pid},0);alive=Path('/proc/${c.pid}/stat').read_text().split()[2]!='Z'\nexcept (OSError,FileNotFoundError):alive=False\nlog=Path(str(out)+'-process.log')\nprint(json.dumps({'complete':(out/'file-hashes.json').exists(),'alive':alive,'tail':log.read_text()[-1800:] if log.exists() else ''}))`));
  if(status.complete||!status.alive)break;
  console.log(JSON.stringify({running:c.attempt_id,tail:status.tail.slice(-550)}));
  await new Promise(resolve=>setTimeout(resolve,15000));
}
const log=await(await request('/files/'+(c.remote_output+'-process.log').replace('/kaggle/working/',''))).arrayBuffer();
fs.writeFileSync(path.join(local,'process.log'),Buffer.from(log));
if(!status?.complete)throw new Error('Diagnostic not complete; retained process log. Resume collector without relaunching.');
const hashes=await(await request('/files/'+c.remote_output.replace('/kaggle/working/','')+'/file-hashes.json')).json();
for(const [name,checksum] of Object.entries(hashes)) {
  if(name.includes('..')||path.isAbsolute(name))throw new Error('Unsafe artifact filename');
  const bytes=Buffer.from(await(await request('/files/'+c.remote_output.replace('/kaggle/working/','')+'/'+name)).arrayBuffer());
  if(sha(bytes)!==checksum)throw new Error('Artifact checksum mismatch: '+name);
  const dest=path.join(local,name);fs.mkdirSync(path.dirname(dest),{recursive:true});fs.writeFileSync(dest,bytes);
}
fs.writeFileSync(path.join(local,'file-hashes.json'),JSON.stringify(hashes,null,2)+'\n');
const report=JSON.parse(fs.readFileSync(path.join(local,'report.json')));
if(!report.completed||report.full_forward||report.backward||report.optimizer_updates!==0||report.layers.length!==4)throw new Error('Unexpected diagnostic scope');
fs.writeFileSync(path.join(here,'metrics',template.run_id+'.json'),JSON.stringify({attempt_id:c.attempt_id,commit:c.git_commit,tokens:report.tokens,layers:report.layers,operation_count:report.operation_count,introduces_fp32_count:report.introduces_fp32_count,embedding_dtype:report.embedding_dtype,adapter_storage_dtype_counts:report.adapter_storage_dtype_counts,traced_forward_seconds:report.traced_forward_seconds,gpu_peak_allocated_gib:report.gpu_peak_allocated_gib,backward:false,optimizer_updates:0},null,2)+'\n');
console.log(JSON.stringify({collected:c.attempt_id,operations:report.operation_count,introduces_fp32:report.introduces_fp32_count}));
