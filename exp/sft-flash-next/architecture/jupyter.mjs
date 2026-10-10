/** Upload a frozen attempt and execute via Kaggle Jupyter HTTP/WebSocket only. */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
const root = path.dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(process.argv.slice(2).reduce((rows,value,index,all)=>{
  if(value.startsWith('--')) rows.push([value.slice(2),all[index+1]]);return rows;
},[]));
const base = fs.readFileSync(args['url-file'] ?? '/tmp/kaggle_probe_url','utf8').trim().replace(/\/$/,'');
const sha = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
async function request(relative, options={}) {
  try {
    const response = await fetch(base+relative,{...options,signal:AbortSignal.timeout(30000)});
    if(!response.ok) throw new Error('HTTP '+response.status);
    return response;
  } catch { throw new Error('Jupyter request failed; private connection URL omitted'); }
}
async function execute(code) {
  const kernels = await (await request('/api/kernels')).json();
  if(kernels.length !== 1) throw new Error('Expected one Jupyter control kernel');
  return await new Promise((resolve,reject)=>{
    const ws = new WebSocket(base.replace(/^https:/,'wss:')+'/api/kernels/'+kernels[0].id+'/channels');
    const id = crypto.randomUUID();let output='';
    const timer=setTimeout(()=>{ws.close();reject(new Error('Jupyter control execution timed out'));},60000);
    ws.addEventListener('open',()=>ws.send(JSON.stringify({header:{msg_id:id,username:'codex',session:crypto.randomUUID(),date:new Date().toISOString(),msg_type:'execute_request',version:'5.3'},parent_header:{},metadata:{},channel:'shell',content:{code,silent:false,store_history:false,user_expressions:{},allow_stdin:false,stop_on_error:true}})));
    ws.addEventListener('message',event=>{
      let message;try{message=JSON.parse(event.data);}catch{return;}
      if(message.parent_header?.msg_id!==id)return;
      if(message.msg_type==='stream')output+=message.content.text;
      if(message.msg_type==='error')output+=message.content.ename+': '+message.content.evalue;
      if(message.msg_type==='execute_reply'){
        clearTimeout(timer);ws.close();
        if(message.content.status==='ok')resolve(output.trim());else reject(new Error('Remote control failed: '+output.slice(-1500)));
      }
    });
    ws.addEventListener('error',()=>{clearTimeout(timer);reject(new Error('Jupyter WebSocket failed; private URL omitted'));});
  });
}
function files(directory,prefix=''){
  return fs.readdirSync(directory,{withFileTypes:true}).flatMap(entry=>{
    if(['results','reports','__pycache__','.git'].includes(entry.name))return [];
    const relative=path.posix.join(prefix,entry.name);const full=path.join(directory,entry.name);
    if(entry.isDirectory())return files(full,relative);
    return /\.(py|mjs|json|md)$/.test(entry.name)?[relative]:[];
  });
}
async function upload(remote,bytes){
  await request('/api/contents/'+remote.replace(/^\/kaggle\/working\//,''),{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({type:'file',format:'base64',content:bytes.toString('base64')})});
}
async function json(remote){
  const response = await fetch(base+'/files/'+remote.replace(/^\/kaggle\/working\//,'')+'?t='+Date.now(),{signal:AbortSignal.timeout(15000)}).catch(()=>null);
  if(!response || response.status===404)return null;
  if(!response.ok)throw new Error('Jupyter status unavailable; private URL omitted');
  return response.json();
}
async function collect(remote,local){
  fs.mkdirSync(local,{recursive:true});
  const item=await (await request('/api/contents/'+remote.replace(/^\/kaggle\/working\//,'')+'?content=1')).json();
  for(const entry of item.content){
    if(path.basename(entry.name)!==entry.name)throw new Error('Unsafe artifact name');
    if(entry.name==='model-view')continue; // Derived links to full model weights.
    if(entry.type==='directory'){await collect(remote+'/'+entry.name,path.join(local,entry.name));continue;}
    const response=await request('/files/'+(remote+'/'+entry.name).replace(/^\/kaggle\/working\//,'')+'?t='+Date.now());
    const dest=path.join(local,entry.name);await pipeline(Readable.fromWeb(response.body),fs.createWriteStream(dest+'.part'));fs.renameSync(dest+'.part',dest);
  }
}
async function main(){
  const action=args.action ?? 'run';
  const timeoutSeconds=Number(args['timeout-seconds'] ?? 1200);
  if(!Number.isSafeInteger(timeoutSeconds)||timeoutSeconds<1)throw new Error('Timeout must be a positive integer');
  let job,config,attempt;
  if(action==='run'){
    config=JSON.parse(fs.readFileSync(path.resolve(args.config),'utf8'));
    const mode=args.mode ?? 'test';
    if(!['test','train','benchmark'].includes(mode))throw new Error('Unknown mode');
    attempt=new Date().toISOString().replace(/[-:.TZ]/g,'')+'-'+crypto.randomBytes(4).toString('hex');
    job='/kaggle/working/architecture-runs/'+attempt;
    const large=config.optimizations?.lora_routed_experts||config.architecture==='reference';
    config.output=large?'/tmp/flash-next-architecture-artifacts/'+attempt+'/output':job+'/output';
    config.dispatch_commit=execFileSync('git',['rev-parse','HEAD'],{cwd:root,encoding:'utf8'}).trim();
    const sources=files(root),directories=new Set([job,job+'/source',job+'/dependencies']);
    for(const name of sources)directories.add(path.posix.dirname(job+'/source/'+name));
    const code='from pathlib import Path\n'+[...directories].map(d=>'Path('+JSON.stringify(d)+').mkdir(parents=True,exist_ok=True)').join('\n');
    await execute(code);
    const hashes={};for(const name of sources){const bytes=fs.readFileSync(path.join(root,name));hashes[name]=sha(bytes);await upload(job+'/source/'+name,bytes);}
    const opt=config.optimizations ?? {},registry=JSON.parse(fs.readFileSync(path.join(root,'configs/dependencies.json'),'utf8'));
    const needed=[];
    if(opt.head==='cce_exact')needed.push('cce');
    if(opt.head==='liger_flce'||opt.liger_rmsnorm||opt.liger_swiglu)needed.push('liger');
    for(const name of needed){
      const dep=registry[name],response=await fetch(dep.url,{signal:AbortSignal.timeout(60000)});
      if(!response.ok)throw new Error('Pinned dependency download failed: '+name);
      const bytes=Buffer.from(await response.arrayBuffer());
      if(sha(bytes)!==dep.sha256)throw new Error('Dependency hash mismatch: '+name);
      await upload(job+'/dependencies/'+dep.filename,bytes);
    }
    await upload(job+'/config.json',Buffer.from(JSON.stringify(config,null,2)+'\n'));
    await upload(job+'/source-hashes.json',Buffer.from(JSON.stringify(hashes,null,2)+'\n'));
    const entrypoint=args.entrypoint;
    if(entrypoint && (mode!=='test'||!['diagnostics/gdn_backward.py','diagnostics/expert_replay.py','diagnostics/expert_head_replay.py'].includes(entrypoint)))throw new Error('Unknown diagnostic entrypoint');
    const extra=entrypoint?',"--entrypoint",'+JSON.stringify(entrypoint):'';
    console.log(await execute('import subprocess,json\nfrom pathlib import Path\np=subprocess.Popen(["/usr/bin/python3",'+JSON.stringify(job+'/source/runtime/worker.py')+',"--config",'+JSON.stringify(job+'/config.json')+',"--mode",'+JSON.stringify(mode)+',"--timeout",'+JSON.stringify(String(timeoutSeconds))+extra+'],stdout=open('+JSON.stringify(job+'/supervisor.log')+',"w"),stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps({"attempt":'+JSON.stringify(attempt)+',"supervisor_pid":p.pid}))'));
    fs.mkdirSync(path.join(root,'results'),{recursive:true});fs.writeFileSync(path.join(root,'results','last-attempt.json'),JSON.stringify({attempt,job,mode},null,2)+'\n');
  }else{
    attempt=args.attempt;if(!/^[0-9]+-[a-f0-9]+$/.test(attempt))throw new Error('Invalid attempt');job='/kaggle/working/architecture-runs/'+attempt;
  }
  if(action==='status'){
    const monitor=await json(job+'/monitor.json'),result=await json(job+'/output/result.json');
    console.log(JSON.stringify({attempt,monitor,result}));return;
  }
  let monitor,missing=0;
  const deadline=Date.now()+(timeoutSeconds+100)*1000;let lastLog=0;
  while(!(monitor=await json(job+'/monitor.json'))){
    if(++missing%6===0){
      try {
        const health=await request('/api/status');
        if(!health.headers.get('content-type')?.includes('application/json'))throw new Error('Invalid status');
      }catch{
        throw new Error('Jupyter connection unavailable; reconnect and collect attempt '+attempt+'. The detached job status is unknown; private URL omitted');
      }
    }
    if(Date.now()>deadline)throw new Error('Supervisor completion deadline exceeded; inspect the saved attempt before retrying');
    if(Date.now()-lastLog>60000){console.log(JSON.stringify({attempt,status:'running'}));lastLog=Date.now();}
    await new Promise(r=>setTimeout(r,10000));
  }
  const local=path.join(root,'results',attempt);
  await collect(job,local);
  execFileSync('dvc',['add',path.relative(root,local)],{cwd:root,stdio:'inherit'});
  console.log(JSON.stringify({attempt,monitor,local,dvc_cached:true}));
  if(monitor.returncode!==0)throw new Error('Architecture job failed; artifacts collected');
}
main().catch(error=>{console.error(error.message);process.exitCode=1;});
