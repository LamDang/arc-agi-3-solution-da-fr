import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import { test } from 'node:test';
import { CheckpointCollector, hashFile, requireSpace } from '../training/transfer.mjs';

const sha=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');
function fixture(t){
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'trajectory-transfer-'));
  t.after(()=>fs.rmSync(root,{recursive:true,force:true}));
  const files=new Map(),uploads=[],published=[];
  const remote='/kaggle/working/job/output';
  let current,restore,ready;
  const collector=new CheckpointCollector({run:root,remote,
    json:async file=>file.endsWith('/request.json')?current:file.endsWith('/restore-request.json')?restore:ready,
    request:async url=>new Response(files.get(url.split('?')[0].split('/').pop())),
    upload:async (file,bytes)=>{uploads.push({file,bytes});if(file.endsWith('restore-ready.json'))ready=JSON.parse(bytes);},
    publish:async file=>{published.push(file);}
  });
  const request=(name,bytes,kind='shard')=>{
    files.set(name,bytes);
    current={checkpoint:'update-00000001',path:name,kind,bytes:bytes.length,sha256:sha(bytes),id:sha(Buffer.from(name))};
    return current;
  };
  return {root,collector,uploads,published,request,setRestore:r=>{restore=r;}};
}

test('download verifies bytes before acknowledgment and survives collector restart',async t=>{
  const f=fixture(t),bytes=Buffer.from('mock tensor shard');
  const item=f.request('00000.pt',bytes);
  await f.collector.poll();
  const local=path.join(f.root,'incoming/update-00000001/00000.pt');
  assert.equal(hashFile(local),item.sha256);
  assert.equal(JSON.parse(f.uploads[0].bytes).id,item.id);
  f.collector.request=()=>{throw Error('Should reuse durable local receipt');};
  await f.collector.poll();assert.equal(f.uploads.length,2);
});

test('corrupt download is never acknowledged',async t=>{
  const f=fixture(t);f.request('00000.pt',Buffer.from('good'));
  f.collector.request=async()=>new Response('wrong');
  await assert.rejects(f.collector.poll(),/checksum mismatch/);
  assert.equal(f.uploads.length,0);
  assert.equal(fs.existsSync(path.join(f.root,'ack.json')),false);
});

test('manifest requires successful DVC publication before final acknowledgment',async t=>{
  const f=fixture(t);f.request('manifest.json',Buffer.from('{}'),'manifest');
  f.collector.publish=async()=>{throw Error('DVC failed');};
  await assert.rejects(f.collector.poll(),/DVC failed/);
  assert.equal(f.uploads.length,0);
  f.collector.publish=async file=>f.published.push(file);
  await f.collector.poll();
  assert.equal(f.published.length,1);assert.equal(f.uploads.length,1);
});

test('resume uploads one verified shard and reuses server readiness',async t=>{
  const f=fixture(t),bytes=Buffer.from('saved optimizer state');
  f.collector.resume=path.join(f.root,'resume');fs.mkdirSync(f.collector.resume);
  const file=path.join(f.collector.resume,'00000.pt');fs.writeFileSync(file,bytes);
  f.setRestore({path:'00000.pt',bytes:bytes.length,sha256:sha(bytes),id:sha(Buffer.from('restore'))});
  await f.collector.restore();assert.equal(f.uploads.length,2);
  assert.deepEqual(f.uploads[0].bytes,bytes);
  await f.collector.restore();assert.equal(f.uploads.length,2);
});

test('unsafe paths and insufficient local storage fail closed',async t=>{
  const f=fixture(t);f.request('../00000.pt',Buffer.from('x'));
  await assert.rejects(f.collector.poll(),/Unsafe/);
  assert.throws(()=>requireSpace(f.root,Number.MAX_SAFE_INTEGER),/requires/);
  assert.equal(f.uploads.length,0);
});
