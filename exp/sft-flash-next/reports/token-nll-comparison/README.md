# Quarto: paired final replies with per-token NLL

A standalone HTML report for all 30 original Sol/generated-thinking request
pairs at 512 and 256 experts. Every highlighted text span is exactly one scored
token; its background uses one shared linear NLL scale across requests, panels,
categories and expert counts. The default range is 0–6 nats. Higher values
saturate, while hover/click details retain the exact FP32 loss. The cap can be
changed globally to 2, 4, 6 or 10 nats; there is no per-panel normalization.

The default view shows all 30 expanded replies, separating thinking and tool-call
rows so Python outputs line up despite different thinking lengths. Controls
select experts, filter/search requests, show thinking or Python code alone,
mark token boundaries, and collapse/expand requests. Hovering Python code
highlights its identical token in the other panel and shows the paired loss.
Thinking text differs, so it is paired by request rather than aligned by token.

`report.qmd`, `report.css` and `report.js` are the Quarto source. `build.py`
validates all 120 result records with the existing comparison code, checks that
token offsets reconstruct every reply exactly, embeds all 72,932 losses and
renders through **Quarto 1.10.19** with `embed-resources: true`. JSON is inert and
escapes `<`; teacher/code strings are rendered as text, never executed as HTML.
The output needs no CDN, network connection, running notebook or GPU.

## Rebuild

Install the documented CPU environment and Quarto 1.10.19, then fetch/extract
[the source](../../../../data/sol-nll-fold0-30-results-20261008/README.md) and
[generated-thinking](../../../../data/sol-nll-fold0-30-genthink-results-20261008/README.md)
result archives. From the repository root:

```bash
python exp/sft-flash-next/reports/token-nll-comparison/build.py \
  --source /tmp/sol-nll-source \
  --variant /tmp/sol-nll-genthink \
  --out /tmp/sol-nll-quarto-report \
  --quarto /path/to/quarto
```

`report.html` is the complete deliverable. The generated `report.qmd`, CSS, JS
and `payload.html` can also be edited/rerendered with:

```bash
quarto render /tmp/sol-nll-quarto-report/report.qmd --to html
```

The builder keeps Quarto/Deno caches under its output directory when no custom
cache location is configured, so a read-only home is supported.

## Storage and verification

The [HTML report dataset](../../../../data/sol-nll-fold0-30-token-report-20261008/README.md)
tracks the teacher-containing HTML in private DVC. Git tracks the source,
checksums and browser verification; generated payloads and screenshots remain
outside Git. `browser-verification.json` records real Chromium checks for
complete token/text reconstruction, saved losses, expert switching, shared
colors, filters, code-token alignment, token inspection, expand/collapse,
mobile layout and zero external network requests. No Kaggle session is started.


To repeat browser verification, serve the generated output locally and launch
Chromium with a local DevTools port, then run:

```bash
node exp/sft-flash-next/reports/token-nll-comparison/check_browser.mjs \
  http://127.0.0.1:8765/report.html /tmp/sol-nll-browser-check \
  http://127.0.0.1:9223
```

The checker uses Node's built-in WebSocket and filesystem APIs (tested on Node
24), writes a viewport preview and verification JSON, and checks every token
against the embedded data before exercising controls. These generated files
stay outside Git because screenshots contain teacher text.
