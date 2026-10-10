import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {validateConfig} from '../kaggle_overfit_run.mjs';
const c=JSON.parse(fs.readFileSync(new URL('../configs/bf16-overfit-v8.json',import.meta.url)));
test('learning criterion is bounded and replaces exact gradient matching',()=>{
  assert.equal(validateConfig(c),c);
  assert.throws(()=>validateConfig({...c,learning:{...c.learning,max_updates:200}}),/learning contract/);
  assert.throws(()=>validateConfig({...c,purpose:'production-training'}),/learning contract/);
});
test('Opt6 preserves learning contract and pins both kernel sources and wheel',()=>{
  const opt6=JSON.parse(fs.readFileSync(new URL('../configs/liger-opt6-overfit-v9.json',import.meta.url)));
  assert.equal(validateConfig(opt6),opt6);
  assert.throws(()=>validateConfig({...opt6,opt6:'blanket-liger'}),/Opt6 kernel contract/);
  assert.throws(()=>validateConfig({...opt6,liger_dependency:{...opt6.liger_dependency,sha256:'0'.repeat(64)}}),/Opt6 kernel contract/);
});
