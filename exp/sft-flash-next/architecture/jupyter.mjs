/** Upload a frozen attempt and execute via Kaggle Jupyter HTTP/WebSocket only. */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { CheckpointCollector, dvcPublisher, durableJson, hashFile, requireSpace } from './training/transfer.mjs';
const root = path.dirname(fileURLToPath(import.meta.url));
const repository = path.resolve(root,'../../..');
const args = Object.fromEntries(process.argv.slice(2).reduce((rows,value,index,all)=>{
  if(value.startsWith('--')) rows.push([value.slice(2),all[index+1]]);return rows;
},[]));
const base = fs.readFileSync(args['url-file'] ?? '/tmp/kaggle_probe_url','utf8').trim().replace(/\/$/,'');
const sha = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
async function request(relative, options={}, timeout=30000) {
  try {
    const response = await fetch(base+relative,{...options,signal:AbortSignal.timeout(timeout)});
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
  await request('/api/contents/'+remote.replace(/^\/kaggle\/working\//,''),{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({type:'file',format:'base64',content:bytes.toString('base64')})},300000);
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
    if(['model-view','transfer'].includes(entry.name))continue; // No frozen weights or transient checkpoint spool.
    if(entry.type==='directory'){await collect(remote+'/'+entry.name,path.join(local,entry.name));continue;}
    const response=await request('/files/'+(remote+'/'+entry.name).replace(/^\/kaggle\/working\//,'')+'?t='+Date.now());
    const dest=path.join(local,entry.name);await pipeline(Readable.fromWeb(response.body),fs.createWriteStream(dest+'.part'));fs.renameSync(dest+'.part',dest);
  }
}
async function main(){
  const action=args.action ?? 'run';
  const timeoutSeconds=Number(args['timeout-seconds'] ?? (args.mode==='train'?86400:1200));
  if(!Number.isSafeInteger(timeoutSeconds)||timeoutSeconds<1)throw new Error('Timeout must be a positive integer');
  let job,config,attempt,trainingRun,resume;
  if(action==='run'){
    config=JSON.parse(fs.readFileSync(path.resolve(args.config),'utf8'));
    const mode=args.mode ?? 'test';
    if(!['test','train','benchmark'].includes(mode))throw new Error('Unknown mode');
    attempt=new Date().toISOString().replace(/[-:.TZ]/g,'')+'-'+crypto.randomBytes(4).toString('hex');
    job='/kaggle/working/architecture-runs/'+attempt;
    if(config.training){
      if(mode!=='train')throw new Error('Production settings require --mode train');
      if(args.resume){
        resume=path.resolve(args.resume);
        if(fs.existsSync(path.join(resume,'latest.json'))){
          trainingRun=resume;
          const latest=JSON.parse(fs.readFileSync(path.join(resume,'latest.json'),'utf8'));
          resume=path.resolve(resume,latest.checkpoint);
        }else if(fs.existsSync(path.join(resume,'manifest.json'))){
          trainingRun=path.dirname(path.dirname(resume));
        }
      }
      trainingRun=path.resolve(args['local-run'] ?? trainingRun ?? path.join(repository,'exp/sft-flash-next/training-runs',attempt));
      if(!trainingRun.startsWith(repository+path.sep))throw new Error('Local checkpoints must be in this working repository');
      requireSpace(trainingRun,(fs.existsSync(path.join(trainingRun,'latest.json'))?23:46)*2**30);
      const ignore=path.join(trainingRun,'.gitignore');
      fs.writeFileSync(ignore,'/dvc-cache\n/incoming\n/ack.json\n/launch.json\n');
      if(args.resume){
        const manifest=JSON.parse(fs.readFileSync(path.join(resume,'manifest.json'),'utf8'));
        for(const shard of manifest.shards){
          if(path.basename(shard.path)!==shard.path||hashFile(path.join(resume,shard.path))!==shard.sha256)
            throw new Error('Local resume checkpoint is incomplete or corrupt');
        }
        config.training.resume=job+'/resume';
      }else if(config.training.resume)throw new Error('Use --resume with the local checkpoint; resume uploads are streamed');
    }
    const large=config.optimizations?.lora_routed_experts||config.architecture==='reference';
    const storage=JSON.parse(await execute('import tempfile,json\ntry:\n    with tempfile.TemporaryDirectory(dir="/tmp"): pass\n    writable=True\nexcept OSError:\n    writable=False\nprint(json.dumps({"tmp_writable":writable}))'));
    config.output=!config.training&&large&&storage.tmp_writable?'/tmp/flash-next-architecture-artifacts/'+attempt+'/output':job+'/output';
    config.dispatch_commit=execFileSync('git',['rev-parse','HEAD'],{cwd:root,encoding:'utf8'}).trim();
    const sources=files(root),directories=new Set([job,job+'/source',job+'/dependencies']);
    if(config.training){
      config.training.dataset=job+'/dataset';config.training.folds=job+'/folds.json';
      directories.add(job+'/dataset');
      if(resume)directories.add(job+'/resume');
    }
    for(const name of sources)directories.add(path.posix.dirname(job+'/source/'+name));
    const code='from pathlib import Path\n'+[...directories].map(d=>'Path('+JSON.stringify(d)+').mkdir(parents=True,exist_ok=True)').join('\n');
    await execute(code);
    if(config.training){
      const dataset=path.resolve(args.dataset ?? path.join(repository,'data/progressive-sol25-trajectories'));
      for(const name of ['trajectories.jsonl','index.json','summary.json'])await upload(job+'/dataset/'+name,fs.readFileSync(path.join(dataset,name)));
      await upload(job+'/folds.json',fs.readFileSync(path.resolve(args.folds ?? path.join(repository,'data/game_folds/folds.json'))));
      if(resume)await upload(job+'/resume/manifest.json',fs.readFileSync(path.join(resume,'manifest.json')));
      durableJson(path.join(trainingRun,'launch.json'),{attempt,job,resume,trainingRun});
    }
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
    if(entrypoint && (mode!=='test'||!['diagnostics/gdn_backward.py','diagnostics/expert_replay.py','diagnostics/expert_head_replay.py','diagnostics/hyperconnection_replay.py','diagnostics/hyperconnection_focus.py','diagnostics/host_memory.py'].includes(entrypoint)))throw new Error('Unknown diagnostic entrypoint');
    const extra=entrypoint?',"--entrypoint",'+JSON.stringify(entrypoint):'';
    console.log(await execute('import subprocess,json\nfrom pathlib import Path\np=subprocess.Popen(["/usr/bin/python3",'+JSON.stringify(job+'/source/runtime/worker.py')+',"--config",'+JSON.stringify(job+'/config.json')+',"--mode",'+JSON.stringify(mode)+',"--timeout",'+JSON.stringify(String(timeoutSeconds))+extra+'],stdout=open('+JSON.stringify(job+'/supervisor.log')+',"w"),stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps({"attempt":'+JSON.stringify(attempt)+',"supervisor_pid":p.pid}))'));
    fs.mkdirSync(path.join(root,'results'),{recursive:true});fs.writeFileSync(path.join(root,'results','last-attempt.json'),JSON.stringify({attempt,job,mode},null,2)+'\n');
  }else{
    attempt=args.attempt;if(!/^[0-9]+-[a-f0-9]+$/.test(attempt))throw new Error('Invalid attempt');job='/kaggle/working/architecture-runs/'+attempt;
    config=await json(job+'/config.json');
    if(config?.training){
      trainingRun=path.resolve(args['local-run'] ?? path.join(repository,'exp/sft-flash-next/training-runs',attempt));
      const launch=JSON.parse(fs.readFileSync(path.join(trainingRun,'launch.json'),'utf8'));resume=launch.resume;
    }
  }
  if(action==='status'){
    const monitor=await json(job+'/monitor.json'),result=await json(job+'/output/result.json');
    console.log(JSON.stringify({attempt,monitor,result}));return;
  }
  let monitor,missing=0;
  let collector;
  async function telemetry(){
    await collect(job+'/output',path.join(trainingRun,'telemetry',attempt));
    const events=path.join(trainingRun,'telemetry',attempt,'tensorboard');
    if(fs.existsSync(events)){
      const combined=path.join(trainingRun,'telemetry','tensorboard');fs.mkdirSync(combined,{recursive:true});
      for(const name of fs.readdirSync(events))fs.copyFileSync(path.join(events,name),path.join(combined,name)+'.part');
      for(const name of fs.readdirSync(combined).filter(n=>n.endsWith('.part')))fs.renameSync(path.join(combined,name),path.join(combined,name.slice(0,-5)));
    }
  }
  if(trainingRun){
    let python=args['dvc-python'];
    if(!python){
      const executable=execFileSync('which',['dvc'],{encoding:'utf8'}).trim();
      const shebang=fs.readFileSync(executable,'utf8').split('\n')[0];
      if(!shebang.startsWith('#!/')||shebang.includes(' '))throw new Error('Set --dvc-python to a Python with DVC installed');
      python=shebang.slice(2);
    }
    const publish=dvcPublisher(repository,trainingRun,python,path.join(root,'training/dvc_checkpoint.py'));
    collector=new CheckpointCollector({run:trainingRun,remote:job+'/output',json,request,upload,resume,
      publish:async manifest=>{
        await telemetry();
        publish(manifest);
      }});
  }
  const deadline=Date.now()+(timeoutSeconds+100)*1000;let lastLog=0;
  while(!(monitor=await json(job+'/monitor.json'))){
    if(collector){await collector.restore();await collector.poll();}
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
    await new Promise(r=>setTimeout(r,collector?1000:10000));
  }
  if(collector){
    await collector.poll();
    await telemetry();
    const latest=path.join(trainingRun,'latest.json');
    if(fs.existsSync(latest)){
      const current=JSON.parse(fs.readFileSync(latest,'utf8'));
      const python=args['dvc-python'] ?? fs.readFileSync(execFileSync('which',['dvc'],{encoding:'utf8'}).trim(),'utf8').split('\n')[0].slice(2);
      dvcPublisher(repository,trainingRun,python,path.join(root,'training/dvc_checkpoint.py'))(path.join(trainingRun,current.checkpoint,'manifest.json'));
    }
    console.log(JSON.stringify({attempt,monitor,local:trainingRun,dvc_cached:fs.existsSync(latest)}));
    if(monitor.returncode!==0)throw new Error('Training stopped; resume from the last complete local DVC checkpoint');
    return;
  }
  const local=path.join(root,'results',attempt);
  await collect(job,local);
  execFileSync('dvc',['add',path.relative(root,local)],{cwd:root,stdio:'inherit'});
  console.log(JSON.stringify({attempt,monitor,local,dvc_cached:true}));
  if(monitor.returncode!==0)throw new Error('Architecture job failed; artifacts collected');
}
main().catch(error=>{console.error(error.message);process.exitCode=1;});
