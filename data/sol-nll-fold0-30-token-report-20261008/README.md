# Quarto HTML: 30 paired replies colored by token NLL

`report.html` is a self-contained Quarto report showing all 30 original and
revised generated-thinking replies side by side. The expert selector switches
between 256 (default) and 512. Every token uses the same color scale across both
panels and all requests: **0–6 nats by default**, with globally adjustable caps.
Hover/click shows the exact token loss; identical Python-code tokens highlight
on both sides. Thinking/tool rows align separately; game/request filters and
thinking/code-only views make the longer replies easier to compare.

All text, 72,932 FP32 loss values, styles and scripts are embedded. Download
and open the HTML in a browser; no GPU, notebook, CDN or network is required.
The HTML contains teacher replies and is stored in the existing private DVC
remote. Git stores its pointer, checksum/provenance and browser verification.

```bash
dvc pull data/sol-nll-fold0-30-token-report-20261008/report.html.dvc
```

The report uses the verified
[original](../sol-nll-fold0-30-results-20261008/README.md) and
[generated-thinking](../sol-nll-fold0-30-genthink-results-20261008/README.md)
result datasets. The raw NLL values are unchanged and each highlighted span
reconstructs exactly one target token. Colors clip above the shared maximum;
exact losses remain available in token details.

[Quarto source and build commands](../../exp/sft-flash-next/reports/token-nll-comparison/README.md)
are committed. `provenance.json` records Quarto version, source hashes, run
identities and the HTML SHA256. `browser-verification.json` records functional
and complete-content checks in Chromium.
