# Archive

A snapshot (2026-10-12) of the offline analysis behind the emotion model and its calibration:
the scripts, the reports and the training notebooks. It is kept for the record; nothing in the
game or the library depends on it, and it is not maintained.

| Folder | What it holds |
|---|---|
| `Analysis/` | The analysis scripts (`*.py`), their tests, and the reports (`*.md`). `exploratory/` holds the experiments (volume and jitter augmentation, error analysis, movement calibration, smoothing, ...). |
| `notebooks/` | The Colab training notebooks, **code only**: every output was cleared. |
| `Docs/` | Reports written during the work: volume augmentation, jitter, test error analysis. |
| `VREmotionDetection-Docs/` | Reports that used to live in the library: its validation report and the Anger logit-adjustment report. |

## What is not here, on purpose

- **No data.** No recordings, no headset session logs, no training arrays (`.npz`), no
  spreadsheets, no model checkpoints, no result tables. The scripts read their data from local
  paths (for example `Analysis/data/`) that exist only on the researchers' machines.
- **No participant codes.** Participants appear as `P01`, `P02`, ... The table that maps these
  back to recordings is kept only on the researchers' machines, never in this repository.

## Running the scripts

They expect the original layout: the game repository root, with the data under
`Analysis/data/` and the notebooks' arrays under `notebooks/`. Copy `Archive/Analysis` back to
`Analysis` on a machine that has the data. Results in the reports were produced on that data
and are internal until they are reproduced on new participants.
