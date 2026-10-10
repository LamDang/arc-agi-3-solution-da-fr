/** Desktop copy-back protocol, separate from model execution and its memory. */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { execFileSync } from 'node:child_process';

export const hashFile = file => {
  const hash=crypto.createHash('sha256'),fd=fs.openSync(file,'r'),buffer=Buffer.alloc(8<<20);
  try {let n;while((n=fs.readSync(fd,buffer,0,buffer.length,null)))hash.update(buffer.subarray(0,n));}
  finally {fs.closeSync(fd);}
  return hash.digest('hex');
};
export function durableJson(file,value) {
  const temporary=file+'.part';fs.mkdirSync(path.dirname(file),{recursive:true});
  fs.writeFileSync(temporary,JSON.stringify(value,null,2)+'\n');
  const fd=fs.openSync(temporary,'r');try{fs.fsyncSync(fd);}finally{fs.closeSync(fd);}
  fs.renameSync(temporary,file);
  const directory=fs.openSync(path.dirname(file),'r');try{fs.fsyncSync(directory);}finally{fs.closeSync(directory);}
}
export function requireSpace(directory,bytes) {
  fs.mkdirSync(directory,{recursive:true});
  const stats=fs.statfsSync(directory),available=stats.bavail*stats.bsize;
  if(available<bytes)throw new Error(`Checkpoint copy-back requires ${(bytes/2**30).toFixed(1)} GiB free in ${directory}; ${(available/2**30).toFixed(1)} GiB available`);
}
export class CheckpointCollector {
  constructor({run,remote,json,request,upload,publish,resume}) {
    Object.assign(this,{run,remote,json,request,upload,publish,resume});
    fs.mkdirSync(run,{recursive:true});
  }
  async restore() {
    const item=await this.json(this.remote+'/transfer/restore-request.json');
    if(!item)return;
    if(!this.resume||!/^\d{5}\.pt$/.test(item.path)||!/^[a-f0-9]{64}$/.test(item.sha256)||
       !/^[a-f0-9]{64}$/.test(item.id))throw new Error('Invalid resume request');
    const ready=await this.json(this.remote+'/transfer/restore-ready.json');
    if(ready?.id===item.id)return;
    const file=path.join(this.resume,item.path);
    if(fs.statSync(file).size!==item.bytes||hashFile(file)!==item.sha256)throw new Error('Local resume shard is corrupt');
    await this.upload(this.remote+'/transfer/restore-spool/'+item.path,fs.readFileSync(file));
    await this.upload(this.remote+'/transfer/restore-ready.json',Buffer.from(JSON.stringify({id:item.id})));
  }
  async poll() {
    const item=await this.json(this.remote+'/transfer/request.json');
    if(!item)return false;
    if(!/^update-\d{8}$/.test(item.checkpoint)||!/^([0-9]{5}\.pt|manifest\.json)$/.test(item.path)||
       !['shard','manifest'].includes(item.kind)||!Number.isSafeInteger(item.bytes)||item.bytes<1||
       !/^[a-f0-9]{64}$/.test(item.sha256)||!/^[a-f0-9]{64}$/.test(item.id))throw new Error('Unsafe checkpoint request');
    if((item.kind==='manifest')!==(item.path==='manifest.json'))throw new Error('Checkpoint request kind/path mismatch');
    const receipt=path.join(this.run,'ack.json');
    const last=fs.existsSync(receipt)?JSON.parse(fs.readFileSync(receipt,'utf8')):null;
    if(last?.id===item.id){
      await this.upload(this.remote+'/transfer/ack.json',Buffer.from(JSON.stringify(last)));
      return true;
    }
    let directory=path.join(this.run,'incoming',item.checkpoint);
    const completed=path.join(this.run,'checkpoints',item.checkpoint);
    if(fs.existsSync(completed))directory=completed; // Recover a crash during DVC publication.
    fs.mkdirSync(directory,{recursive:true});
    const destination=path.join(directory,item.path);
    if(!fs.existsSync(destination)||fs.statSync(destination).size!==item.bytes||hashFile(destination)!==item.sha256){
      requireSpace(this.run,item.bytes+(256<<20));
      const response=await this.request('/files/'+(this.remote+'/transfer/spool/'+item.path).replace(/^\/kaggle\/working\//,'')+'?t='+Date.now(),{},900000);
      await pipeline(Readable.fromWeb(response.body),fs.createWriteStream(destination+'.part'));
      if(fs.statSync(destination+'.part').size!==item.bytes||hashFile(destination+'.part')!==item.sha256)
        throw new Error('Checkpoint download checksum mismatch; shard not acknowledged');
      const fd=fs.openSync(destination+'.part','r');try{fs.fsyncSync(fd);}finally{fs.closeSync(fd);}
      fs.renameSync(destination+'.part',destination);
      const fdDir=fs.openSync(directory,'r');try{fs.fsyncSync(fdDir);}finally{fs.closeSync(fdDir);}
    }
    if(item.kind==='manifest')await this.publish(destination);
    const ack={id:item.id,sha256:item.sha256};durableJson(receipt,ack);
    await this.upload(this.remote+'/transfer/ack.json',Buffer.from(JSON.stringify(ack)));
    return true;
  }
}

export function dvcPublisher(repository,run,python,script) {
  return manifest => {
    const result=execFileSync(python,[script,'--repository',repository,'--run',run,'--manifest',manifest],
      {encoding:'utf8',maxBuffer:8<<20});
    console.log(result.trim());
  };
}
