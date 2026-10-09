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
