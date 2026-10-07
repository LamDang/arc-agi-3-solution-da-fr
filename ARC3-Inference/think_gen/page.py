"""A self-contained HTML page to read generated thinking side by side.

    uv run --no-sync python -m think_gen.page --run runs/base-max-dfranzen \
        --arm sum=runs/think-calib-b1/sum --arm nosum=runs/think-calib-b1/nosum \
        -o runs/think-calib-b1/page.html

One card per teacher request: the newest input the model saw, the call, the
summary each arm used, the teacher's real thinking when it exists, and each
arm's generated thinking with its checks and judge verdict.
"""
import argparse
import html
import json
from pathlib import Path

from . import logs
from .generate import read_jsonl

CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1b;--muted:#6b6a66;--card:#fff;--line:#e3e2de;--code:#f3f2ef;
--ok:#1f7a4d;--bad:#b3261e;--accent:#3d5a98}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#161615;--fg:#e8e6e1;--muted:#9a978f;
--card:#1f1f1d;--line:#34332f;--code:#262624;--ok:#5fc08c;--bad:#f08a80;--accent:#9db4ea}}
:root[data-theme="dark"]{--bg:#161615;--fg:#e8e6e1;--muted:#9a978f;--card:#1f1f1d;--line:#34332f;--code:#262624;
--ok:#5fc08c;--bad:#f08a80;--accent:#9db4ea}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif}
main{max-width:1500px;margin:0 auto;padding:16px}h1{font-size:20px;margin:8px 0}h2{font-size:17px;margin:28px 0 8px}
.meta{color:var(--muted)}table{border-collapse:collapse;margin:8px 0;font-size:13px}
td,th{border:1px solid var(--line);padding:3px 8px;text-align:right}th:first-child,td:first-child{text-align:left}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:10px 0}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:10px;margin-top:8px}
.col{min-width:0}.col h4{margin:0 0 4px;font-size:13px}
pre{white-space:pre-wrap;word-break:break-word;background:var(--code);padding:8px;border-radius:6px;margin:0;
font:12px/1.45 ui-monospace,monospace;max-height:520px;overflow:auto}
details>summary{cursor:pointer;color:var(--accent)}.ok{color:var(--ok)}.bad{color:var(--bad)}
.tag{display:inline-block;font-size:12px;padding:0 6px;border:1px solid var(--line);border-radius:10px;margin-left:4px}
"""


def esc(s) -> str:
    return html.escape(str(s or ""))


def newest_input(rec: logs.Record, limit: int = 2500) -> str:
    """Text of the messages after the last assistant turn."""
    msgs = rec.messages
    last = max((i for i, m in enumerate(msgs) if m.get("role") == "assistant"), default=0)
    parts = []
    for m in msgs[last + 1:] or msgs[-1:]:
        c = m.get("content")
        if isinstance(c, list):
            c = "\n".join(p.get("text", "[image]") if p.get("type") == "text" else "[image]" for p in c)
        parts.append(f"[{m['role']}] {c}")
    text = "\n\n".join(parts)
    return text if len(text) <= limit else text[:limit] + f"\n… ({len(text) - limit} more characters)"


def verdict_html(v: dict | None) -> str:
    if not v or "covered" not in v:
        return ""
    c = v["covered"]
    cov = sum(1 for x in c if x) / len(c) if c else 0
    pts = "".join(f"<li class={'ok' if ok else 'bad'}>{esc(p)}</li>"
                  for p, ok in zip(v.get("real_points") or [], c))
    con = "".join(f"<li class=bad>{esc(x)}</li>" for x in v.get("contradictions") or [])
    return (f"<details><summary>judge: coverage {cov:.0%}, {len(v.get('contradictions') or [])} "
            f"contradictions, leads={v.get('leads_to_output')}, leak={v.get('leak')}</summary>"
            f"<p>{esc(v.get('notes'))}</p><ul>{pts}</ul>{'<b>Contradictions</b><ul>' + con + '</ul>' if con else ''}"
            f"</details>")


def build(run: Path, arms: dict[str, Path], title: str) -> str:
    rows = {name: {} for name in arms}
    judged = {name: {} for name in arms}
    for name, d in arms.items():
        for p in sorted(d.glob("*.jsonl")):
            for r in read_jsonl(p):
                rows[name][r["key"]] = r
            for v in read_jsonl(d / "judge" / p.name):
                if "covered" in v:
                    judged[name][v["key"]] = v
    keys = sorted({k for r in rows.values() for k in r},
                  key=lambda k: (k.split("#")[0], int(k.split("#")[1])))
    games = sorted({k.split("#")[0] for k in keys})
    recs = {}
    for p in logs.request_logs(run, [g.split("_p")[0] for g in games]):
        for rec in logs.read_log(p):
            recs[rec.key] = rec
    summaries = {}
    for name, d in arms.items():
        s = d / "judge" / "summary.json"
        if s.exists():
            summaries[name] = json.loads(s.read_text())["all"]
    out = [f"<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
           f"<title>{esc(title)}</title><style>{CSS}</style></head><body><main><h1>{esc(title)}</h1>"
           f"<p class=meta>run {esc(run)}; arms: {', '.join(f'{esc(n)} = {esc(d)}' for n, d in arms.items())}</p>"]
    if summaries:
        cols = sorted({c for s in summaries.values() for c in s})
        out.append("<table><tr><th>arm</th>" + "".join(f"<th>{esc(c)}</th>" for c in cols) + "</tr>")
        for n, s in summaries.items():
            out.append(f"<tr><td>{esc(n)}</td>" + "".join(f"<td>{esc(s.get(c))}</td>" for c in cols) + "</tr>")
        out.append("</table>")
    for g in games:
        out.append(f"<h2>{esc(g)}</h2>")
        for k in [k for k in keys if k.startswith(g + "#")]:
            rec = recs.get(k)
            any_row = next(r[k] for r in rows.values() if k in r)
            out.append(f"<div class=card><b>{esc(k)}</b> <span class=meta>step {any_row['analysis_step']}."
                       f"{any_row['request_in_turn']}, teacher reasoning tokens {any_row['teacher_reasoning_tokens']}</span>")
            if rec:
                out.append(f"<details><summary>newest input</summary><pre>{esc(newest_input(rec))}</pre></details>")
            out.append(f"<details><summary>output (call)</summary><pre>{esc(any_row['call'])}</pre></details>")
            cols = []
            if any_row.get("real_reasoning"):
                cols.append(f"<div class=col><h4>real thinking <span class=tag>{len(any_row['real_reasoning'])} chars</span></h4>"
                            f"<pre>{esc(any_row['real_reasoning'])}</pre></div>")
            for name in arms:
                r = rows[name].get(k)
                if not r:
                    continue
                chk = r.get("checks") or {}
                cls = "ok" if r.get("status") == "ok" else "bad"
                leak_tags = "".join(f"<span class='tag bad'>{esc(x)}</span>" for x in chk.get("leaks") or [])
                summ = (f"<details><summary>summary ({esc(r['summary_source'])})</summary><pre>{esc(r['summary'])}</pre></details>"
                        if r.get("summary") else "<div class=meta>no summary</div>")
                cols.append(f"<div class=col><h4>{esc(name)} <span class='tag {cls}'>{esc(r.get('status'))}</span>"
                            f"<span class=tag>{len(r.get('thinking') or '')} chars</span>"
                            f"{leak_tags}</h4>"
                            f"{summ}<pre>{esc(r.get('thinking'))}</pre>{verdict_html(judged[name].get(k))}</div>")
            out.append(f"<div class=cols>{''.join(cols)}</div></div>")
    out.append("</main></body></html>")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--arm", action="append", required=True, help="name=output_dir")
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--title", default="Generated thinking")
    args = ap.parse_args(argv)
    arms = dict(a.split("=", 1) for a in args.arm)
    args.output.write_text(build(args.run, {n: Path(d) for n, d in arms.items()}, args.title))
    print(args.output)


if __name__ == "__main__":
    main()
