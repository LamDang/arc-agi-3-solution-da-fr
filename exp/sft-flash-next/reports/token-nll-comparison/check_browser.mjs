// Verify the actual rendered report via a local Chromium DevTools endpoint.
import fs from 'node:fs';
const reportUrl=process.argv[2] || 'http://127.0.0.1:8765/report.html';
const out=process.argv[3] || '.';
const debugBase=process.argv[4] || 'http://127.0.0.1:9223';
fs.mkdirSync(out,{recursive:true});
const pages=await(await fetch(debugBase+'/json/list')).json();
const page=pages.find(x=>x.type==='page');
const socket=new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve,reject)=>{socket.addEventListener('open',resolve,{once:true});socket.addEventListener('error',reject,{once:true});});
let next=0;const waiting=new Map(),errors=[],requests=[];
socket.addEventListener('message',e=>{const m=JSON.parse(e.data);if(m.id){const p=waiting.get(m.id);waiting.delete(m.id);m.error?p.reject(new Error(JSON.stringify(m.error))):p.resolve(m.result);}else if(m.method==='Runtime.exceptionThrown'){errors.push(m.params.exceptionDetails.text);}else if(m.method==='Network.requestWillBeSent'){requests.push(m.params.request.url);}});
function cdp(method,params={}){return new Promise((resolve,reject)=>{const id=++next;waiting.set(id,{resolve,reject});socket.send(JSON.stringify({id,method,params}));});}
async function evaluate(expression){const r=await cdp('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;}
const assert=(x,m)=>{if(!x)throw Error(m);};
await cdp('Runtime.enable');await cdp('Network.enable');
await cdp('Emulation.setDeviceMetricsOverride',{width:1440,height:1320,deviceScaleFactor:1,mobile:false});
await cdp('Page.navigate',{url:reportUrl});
for(let i=0;i<100;i++){if(await evaluate('!!window.NLLReport'))break;await new Promise(r=>setTimeout(r,100));}
assert(await evaluate('!!window.NLLReport'),'Report initialization failed');
const initial=await evaluate(`(()=>{const D=NLLReport.data;let mismatches=0;for(const r of D.requests){for(const p of ['source','genthink']){const card=[...document.querySelectorAll('.request')].find(c=>c.dataset.sample===r.sample_id);const pres=[...card.querySelectorAll('.token-text')].filter(n=>n.dataset.panel===p);if(pres.map(n=>n.textContent).join('')!==r[p].text)mismatches++;for(const token of pres.flatMap(n=>[...n.querySelectorAll('.tok')])){const i=+token.dataset.token;if(+token.dataset.nll!==r[p].losses['256'][i]||token.style.backgroundColor.replaceAll(" ","")!==NLLReport.color(+token.dataset.nll))mismatches++;}}}return{requests:document.querySelectorAll('.request').length,tokens:document.querySelectorAll('.tok').length,textOrLossMismatches:mismatches,experts:document.getElementById('experts').value}})()`);
assert(initial.requests===30&&initial.tokens===36466&&initial.textOrLossMismatches===0,'Token/text reconstruction failed: '+JSON.stringify(initial));
const shot=await cdp('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(out+'/preview.png',Buffer.from(shot.data,'base64'));
await evaluate(`document.getElementById('experts').value='512';document.getElementById('experts').dispatchEvent(new Event('change'))`);
assert(await evaluate(`(()=>{const D=NLLReport.data;return [...document.querySelectorAll('.tok')].every(n=>{const pre=n.closest('.token-text');const r=D.requests.find(r=>r.sample_id===pre.dataset.sample);return +n.dataset.nll===r[pre.dataset.panel].losses['512'][+n.dataset.token]})})()`),'Expert switch failed');
await evaluate(`document.getElementById('section').value='code';document.getElementById('section').dispatchEvent(new Event('change'))`);
assert(await evaluate(`document.querySelectorAll('.tok').length===16590&&[...document.querySelectorAll('.tok')].every(n=>n.dataset.category==='tool_code')`),'Code-only view failed');
assert(await evaluate(`(()=>{const t=document.querySelector('.tok');t.dispatchEvent(new MouseEvent('mouseover',{bubbles:true}));return document.querySelectorAll('.paired-token').length===2})()`),'Identical-code hover alignment failed');
await evaluate(`document.querySelector('.tok').click()`);assert(await evaluate(`!document.getElementById('inspector').hidden&&document.getElementById('inspector-content').textContent.includes('Identical code token in other panel')`),'Token inspector failed');
await evaluate(`document.getElementById('game').value=NLLReport.data.requests.find(r=>r.game.startsWith('sp80')).game;document.getElementById('game').dispatchEvent(new Event('change'))`);
assert(await evaluate(`document.querySelectorAll('.request').length===6`),'Game filter failed');
await evaluate(`document.getElementById('search').value='r11';document.getElementById('search').dispatchEvent(new Event('input'))`);await new Promise(r=>setTimeout(r,250));
assert(await evaluate(`document.querySelectorAll('.request').length===1`),'Request search failed');
await evaluate(`document.getElementById('scale').value='2';document.getElementById('scale').dispatchEvent(new Event('change'))`);
assert(await evaluate(`document.getElementById('scale-top').textContent==='≥2 · hard'&&[...document.querySelectorAll('.tok')].every(n=>n.style.backgroundColor.replaceAll(" ","")===NLLReport.color(+n.dataset.nll,2))`),'Shared scale failed');
await evaluate(`document.getElementById('collapse-all').click()`);assert(await evaluate(`[...document.querySelectorAll('.request')].every(n=>!n.open)`),'Collapse all failed');
await evaluate(`document.getElementById('expand-all').click()`);assert(await evaluate(`[...document.querySelectorAll('.request')].every(n=>n.open)`),'Expand all failed');
await cdp('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
assert(await evaluate(`document.documentElement.scrollWidth<=window.innerWidth+1`),'Mobile horizontal overflow');
const external=requests.filter(x=>/^https?:/.test(x) && x!==reportUrl && !x.endsWith('/favicon.ico'));assert(external.length===0,'Report made network requests: '+JSON.stringify(external));assert(errors.length===0,'Browser errors: '+JSON.stringify(errors));
const result={browser:'Chromium',initial,expert_switch_verified:true,code_only_tokens:16590,aligned_code_hover_verified:true,token_inspector_verified:true,game_filter_requests:6,request_search_verified:true,global_scale_verified:true,collapse_expand_verified:true,mobile_no_horizontal_overflow:true,external_network_requests:0,javascript_errors:0};
fs.writeFileSync(out+'/browser-verification.json',JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));socket.close();
