# Experiment evidence storage

Code, run configurations and concise Markdown summaries belong in Git.
Per-tensor statistics, comparison arrays, memory observations, detailed JSON
reports, model-selection data and historical metrics belong in DVC.

`evidence-data.dvc` indexes a bundle of those detailed files. Its SHA256
inventory preserves each original relative path and exact bytes. Raw run
archives, reference tensors and benchmark inputs retain their separate existing
DVC pointers; this bundle does not duplicate those archives.

## Restore detailed reports

From the repository root:

```sh
dvc pull exp/sft-flash-next/evidence-data.dvc
python3 exp/sft-flash-next/manage_evidence.py restore
```

This restores the original ignored paths under `architecture/reports/`,
`train/metrics/`, `train/reviews/`, historical `train/gradient-results/`, and
`train/artifacts/`. Existing review scripts and the `v0.md` table generator
continue reading their original paths. Pull the separately referenced raw run
archives when a verifier also needs them. The legacy model-selection input must
be restored before using training commands that read `train/artifacts/`.

Restoration validates every bundle file before writing anything, and refuses
to overwrite a differing working file. `verify` validates the bundle and any
existing original copies without restoring missing files:

```sh
python3 exp/sft-flash-next/manage_evidence.py verify
```

## Publish new detailed results

After producing or reviewing reports:

```sh
python3 exp/sft-flash-next/manage_evidence.py archive
dvc add exp/sft-flash-next/evidence-data
dvc push exp/sft-flash-next/evidence-data.dvc
```

Archive includes new report/metric data from the three report directories and
refreshes the files already in the inventory. `--paths-from paths.json` can add
other evidence paths using a JSON list relative to `exp/sft-flash-next/`.
Commit the updated `.dvc` pointer and concise Markdown conclusions; detailed
files remain ignored. Historical frozen pipeline metrics are restored at their
original paths but their complete data is stored in this bundle.

The local DVC cache can be cleared after a successful push; it is disposable.
Working-copy evidence is independent of remote availability. Do not claim old
raw run archives are available remotely just because this report bundle is.
