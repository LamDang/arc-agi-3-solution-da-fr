# Experiment evidence storage

Code, run configurations and concise Markdown summaries belong in Git.
Per-tensor statistics, comparison arrays, memory observations and detailed JSON
reports for the final architecture belong in DVC.

`evidence-data.dvc` indexes 43 architecture reports and two train-only expert-map
metadata files. Its SHA256 inventory preserves original relative paths and exact
bytes. Architecture run archives, raw reference tensors, benchmark inputs and
restoration/diagnostic proof retain their separate DVC pointers under
`architecture/results/`; this report bundle does not duplicate them.

Retired experiments keep source/configs/tests only under
`experiments/legacy-training/`. Their artifacts, old data pointers and reports
were removed. Historical removal inventories in architecture reports describe
past operations; they do not imply those files remain available.

## Restore and verify

The retained data is currently local-only at user direction. The AWS session
expired and no further upload is requested. A fresh clone cannot pull the new
bundle from the remote yet: transfer the local bundle, or publish it when
explicitly authorized. In this workspace:

```sh
python3 exp/sft-flash-next/manage_evidence.py restore
python3 exp/sft-flash-next/manage_evidence.py verify
```

This restores ignored originals under `architecture/reports/` and
`train/artifacts/`. Existing reviewers and the `v0.md` table generator continue
reading their original report paths. Raw attempt files are separate inputs to
those reviewers; they remain under architecture results in this workspace.

Restoration validates the whole bundle before writing and refuses to overwrite
a differing working file. `verify` checks bundle bytes and any existing copies.

Once remote publication is authorized and completed, a new checkout can use:

```sh
dvc pull exp/sft-flash-next/evidence-data.dvc
python3 exp/sft-flash-next/manage_evidence.py restore
```

## Archive new reports

After producing or reviewing final-architecture reports:

```sh
python3 exp/sft-flash-next/manage_evidence.py archive
dvc add exp/sft-flash-next/evidence-data
```

Archive refreshes existing inventory entries and includes new detailed data
from `architecture/reports/`. `--paths-from paths.json` adds other evidence paths
using a JSON list relative to this folder. Commit the `.dvc` pointer and concise
Markdown conclusions; data stays ignored. Do not reintroduce retired experiment
artifacts into this bundle. When an upload is authorized:

```sh
dvc push exp/sft-flash-next/evidence-data.dvc
```

The local cache is disposable; working-copy evidence and remote availability are
separate. The old cache was cleared and the final run data remains locally.
