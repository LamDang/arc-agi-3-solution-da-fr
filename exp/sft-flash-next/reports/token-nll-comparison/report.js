"use strict";
(() => {
  const D = JSON.parse(document.getElementById("nll-data").textContent);
  const el = id => document.getElementById(id);
  const names = {source: "Original Sol", genthink: "Generated thinking"};
  const categories = {thinking:"Thinking",tool_code:"Python code",tool_format:"Tool formatting",turn_format:"Turn formatting",assistant_text:"Assistant text",boundary:"Boundary"};
  const anchors = [[241,248,238],[254,227,145],[242,126,120]];
  function color(value, maximum = +el("scale").value) {
    const t = Math.max(0, Math.min(1, value / maximum)) * 2;
    const i = Math.min(1, Math.floor(t)), f = t - i;
    return `rgb(${anchors[i].map((x,j)=>Math.round(x+(anchors[i+1][j]-x)*f)).join(",")})`;
  }
  function node(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  }
  const fmt = x => x === null ? "—" : x.toFixed(4);
  const pct = x => `${x > 0 ? "+" : ""}${(100*x).toFixed(2)}%`;
  D.requests.forEach(r => {
    for (const panel of ["source","genthink"]) {
      const s=r[panel];
      s.chars=Array.from(s.text);
      s.codeIndices=s.labels.map((label,i)=>label===D.labels.indexOf("tool_code")?i:-1).filter(i=>i>=0);
      s.codeOrdinals={}; s.codeIndices.forEach((i,j)=>s.codeOrdinals[i]=j);
    }
  });
  for (const game of [...new Set(D.requests.map(r=>r.game))]) el("game").add(new Option(game,game));
  function tokenInfo(r,panel,i) {
    const s=r[panel], count=el("experts").value, loss=s.losses[count][i];
    const label=D.labels[s.labels[i]];
    let text=`${names[panel]} · ${count} experts\n${r.sample_id}\nToken ${i+1}/${s.ids.length} · ID ${s.ids[i]} · ${categories[label] || label}\nNLL ${loss.toFixed(8)} nats · target probability ${Math.exp(-loss).toPrecision(5)}`;
    if (label === "tool_code") {
      const other=r[panel==="source"?"genthink":"source"];
      const j=other.codeIndices[s.codeOrdinals[i]], otherLoss=other.losses[count][j];
      text+=`\nIdentical code token in other panel: ${otherLoss.toFixed(8)} nats\nThis panel minus other: ${(loss-otherLoss).toFixed(8)} nats`;
    }
    return text;
  }
  function renderTokens(r,panel,kind) {
    const s=r[panel], count=el("experts").value, pre=node("pre","token-text");
    pre.dataset.panel=panel; pre.dataset.sample=r.sample_id; pre.dataset.kind=kind;
    const fragment=document.createDocumentFragment();
    s.ids.forEach((id,i)=>{
      const label=D.labels[s.labels[i]];
      if (kind==="thinking" && label!=="thinking") return;
      if (kind==="output" && label==="thinking") return;
      if (kind==="code" && label!=="tool_code") return;
      const [a,b]=s.offsets[i], span=node("span","tok",s.chars.slice(a,b).join(""));
      span.style.backgroundColor=color(s.losses[count][i]);
      span.dataset.token=i; span.dataset.category=label; span.dataset.nll=s.losses[count][i];
      if (label==="tool_code") span.dataset.codeOrdinal=s.codeOrdinals[i];
      span.title=tokenInfo(r,panel,i);
      span.addEventListener("click",()=>{
        const content=el("inspector-content"); content.replaceChildren();
        content.append(node("strong",null,`${names[panel]} · ${count} experts`));
        content.append(node("code",null,JSON.stringify(span.textContent)));
        const detail=node("div",null,tokenInfo(r,panel,i)); detail.style.whiteSpace="pre-line";
        content.append(detail); el("inspector").hidden=false;
      });
      fragment.append(span);
    });
    pre.append(fragment); return pre;
  }
  function showOverview() {
    const count=el("experts").value, overview=el("overview"); overview.replaceChildren();
    for (const key of ["thinking","tool_code"]) {
      const a=D.summaries.source[count].categories[key].nll, b=D.summaries.genthink[count].categories[key].nll;
      const card=node("div","metric-card"); card.append(node("h3",null,`${categories[key]} · all 30 requests`));
      const numbers=node("div","metric-numbers");
      for (const [index,value] of [a,b].entries()) {
        if (index) numbers.append(node("span","metric-arrow","→"));
        const item=node("div","metric-value"); item.append(node("strong",null,fmt(value)),node("span",null,index?"Generated thinking":"Original Sol")); numbers.append(item);
      }
      card.append(numbers,node("div","metric-change",`${pct((b-a)/a)} generated vs original · nats/token`)); overview.append(card);
    }
    const card=node("div","metric-card"); card.append(node("h3",null,"Experiment decision"),node("div","decision-title","Generated thinking · 256 experts"));
    card.append(node("div","decision-note","512 → 256: thinking −0.47%; Python code +1.82%. Both pass the +5% limit. Category scores use sampling-weighted token pooling; each reply below shows its own mean."));
    overview.append(card);
  }
  function render() {
    const count=el("experts").value, game=el("game").value, query=el("search").value.trim().toLowerCase(), mode=el("section").value;
    const rows=D.requests.filter(r=>(game==="all"||r.game===game)&&(`${r.game} ${r.sample_id}`.toLowerCase().includes(query)));
    const container=el("requests"); container.replaceChildren(); el("inspector").hidden=true;
    showOverview(); el("visible-count").textContent=`${rows.length} of 30 requests · ${count} experts`;
    el("empty").hidden=rows.length!==0;
    for (const r of rows) {
      const card=node("details","request"); card.open=true; card.dataset.sample=r.sample_id;
      const summary=node("summary"); const title=node("div","request-title",`${r.order.toString().padStart(2,"0")} / ${r.game.split("-")[0]} · request ${r.request_index}`);
      title.append(node("small",null,r.sample_id)); summary.append(title,node("span","request-meta",`${r.prompt_tokens.toLocaleString()} prompt tokens · level ${r.level ?? "unknown"}`)); card.append(summary);
      const heads=node("div","column-heads");
      for (const panel of ["source","genthink"]) {
        const s=r[panel], metric=s.metrics[count], head=node("div","column-head",names[panel]);
        head.append(node("span",null,`${s.ids.length.toLocaleString()} target tokens · thinking ${fmt(metric.thinking_nll)} · code ${fmt(metric.tool_code_nll)}`)); heads.append(head);
      }
      card.append(heads);
      const groups=mode==="full"?["thinking","output"]:[mode];
      for (const kind of groups) {
        const heading=node("div","section-heading",kind==="thinking"?"Thinking":kind==="code"?"Python code":"Python tool call & turn formatting");
        card.append(heading); const row=node("div","token-row");
        for (const panel of ["source","genthink"]) { const pane=node("div","token-pane"); pane.append(renderTokens(r,panel,kind)); row.append(pane); }
        card.append(row);
      }
      container.append(card);
    }
    el("nll-report").classList.toggle("token-boundaries",el("boundaries").checked);
  }
  function updateScale() {
    const maximum=+el("scale").value;
    el("gradient").style.background=`linear-gradient(90deg,${color(0)},${color(maximum/2)},${color(maximum)})`;
    el("scale-middle").textContent=maximum/2; el("scale-top").textContent=`≥${maximum} · hard`;
    document.querySelectorAll(".tok").forEach(span=>span.style.backgroundColor=color(+span.dataset.nll));
  }
  let debounce;
  el("search").addEventListener("input",()=>{clearTimeout(debounce);debounce=setTimeout(render,150);});
  for (const id of ["experts","game","section"]) el(id).addEventListener("change",render);
  el("scale").addEventListener("change",updateScale);
  el("boundaries").addEventListener("change",()=>el("nll-report").classList.toggle("token-boundaries",el("boundaries").checked));
  el("expand-all").addEventListener("click",()=>document.querySelectorAll(".request").forEach(x=>x.open=true));
  el("collapse-all").addEventListener("click",()=>document.querySelectorAll(".request").forEach(x=>x.open=false));
  el("close-inspector").addEventListener("click",()=>el("inspector").hidden=true);
  let paired=[];
  const clearPair=()=>{paired.forEach(x=>x.classList.remove("paired-token"));paired=[];};
  el("requests").addEventListener("mouseover",event=>{
    const token=event.target.closest(".tok"); if(!token) return; clearPair();
    if(token.dataset.codeOrdinal===undefined) return;
    paired=[...token.closest(".request").querySelectorAll(`[data-code-ordinal="${token.dataset.codeOrdinal}"]`)];
    paired.forEach(x=>x.classList.add("paired-token"));
  });
  el("requests").addEventListener("mouseout",clearPair);
  const provenance=el("provenance");
  provenance.append(node("p",null,"All 120 evaluations and 72,932 per-token loss values were checksum-verified. All 30 contexts/images were independently reprocessed and matched. Each highlighted text span is exactly one scored token: no token losses are averaged for display. The colors clip at one shared maximum; tooltips retain exact FP32 values."));
  provenance.append(node("p",null,`Original run: ${D.identities.source}\nGenerated-thinking run: ${D.identities.genthink}\nBuilt with Quarto ${D.quarto_version}. No GPU is needed to view or rebuild this report.`));
  render(); updateScale();
  window.NLLReport={data:D,color,render};
})();
