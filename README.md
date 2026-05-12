# fpcb-microscope-to-SLRL

Standalone Windows GUI for generating structured outputs (heatmap, segment polylines JSON, summary) from microscopic FPCB crack images, with **GrabCut-assisted labeling**, **public lead pre-training**, **on-site lead calibration**, **3-class joint training**, and optional **nested project outputs**.

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
  images/                      # source microscopy images
  labels/masks/                # crack: <stem>.png ; lead: <stem>_lead.png ; drafts *_draft / *_lead_draft
  labels/meta/                 # JSON per mask (<stem>.json crack, <stem>_lead.json lead)
  outputs/                     # optional nested batch outputs (see app checkbox)
  checkpoints/                 # training exports (2-class crack, 3-class joint, lead calibration)
  runs/                        # reserved for future logs
```

Public bundles for **lead pre-training** (keep outside git):

```text
<PublicBundle>/
  images/
  labels/lead_masks/           # PNG masks, same stem as images/
  checkpoints/                 # created by pretrain_lead_public.py (default output location)
```

## Iteration loop (quality + stability)

1. **Pre-train lead** on a public `images/` + `labels/lead_masks/` bundle (`pretrain_lead_public.py`); ship `lead_public_3c_best.pt` with the tool or line image server.
2. **Label** cracks (`labels/masks/<stem>.png`) and, where needed, **lead** (`<stem>_lead.png`) for reflectance calibration; verify on known OK/NG cases.
3. **Calibrate** lead on-site: `train.py --mode calibrate-lead --init-checkpoint <public.pt>` (short epochs, backbone frozen).
4. **Joint train** when crack masks exist: `train.py --mode joint` → `segmentation_deeplab_3c_best.pt`; **eval** with `eval.py --mode joint`.
5. **Batch infer** with nested outputs; load the 3-class checkpoint in the GUI for DL-assisted lead/crack maps.
6. Re-open ambiguous images in labeling; drafts auto-save every ~15s while editing.

## CI

Push to `main` runs tests and uploads a Windows EXE artifact (see `fpcb_heatmap_app/README.md`).
