# fpcb-microscope-to-SLRL

Standalone Windows GUI for generating structured outputs (heatmap, segment polylines JSON, summary) from microscopic FPCB crack images, with **GrabCut-assisted labeling**, **CLI training/evaluation**, and optional **nested project outputs**.

## Why a separate repo

Keep datasets, runs, and builds off crowded disks: clone to `C:\` (or another drive), keep `<project>/` data **outside** git (see `.gitignore`).

## Clone on `C:` (recommended)

```powershell
git clone https://github.com/mjk93447-cpu/fpcb-microscope-to-SLRL.git C:\dev\fpcb-microscope-to-SLRL
cd C:\dev\fpcb-microscope-to-SLRL\fpcb_heatmap_app
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Use a **virtual environment** on `C:` so wheels and caches do not fill the system drive root.

## Subproject

- [`fpcb_heatmap_app/`](fpcb_heatmap_app/) — PyQt5 app, labeling dialog, `train.py` / `eval.py`, tests, CI workflow.

## Project folder layout (runtime)

Create via **Labeling → Create project folders** or manually:

```text
<ProjectRoot>/
  images/              # source microscopy images
  labels/masks/        # crack PNG masks (0/255) + *_draft.png autosaves
  labels/meta/         # JSON sidecar per image
  outputs/             # optional nested batch outputs (see app checkbox)
  checkpoints/         # training exports
  runs/                # reserved for future logs
```

## Iteration loop (quality + stability)

1. **Label** a small, hard subset first; verify masks on a few known OK/NG cases.
2. **Train** (`train.py`), then **eval** (`eval.py`); load `checkpoints/crack_deeplab_best.pt` in the main GUI.
3. **Batch infer** with nested outputs if you want `outputs/overlays|heatmaps|segments/` + `summary.csv`.
4. Re-open ambiguous images in labeling; drafts auto-save every ~15s while editing.

## CI

Push to `main` runs tests and uploads a Windows EXE artifact (see `fpcb_heatmap_app/README.md`).
