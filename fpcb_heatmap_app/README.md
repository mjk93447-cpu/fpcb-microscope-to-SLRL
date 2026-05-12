# FPCB Microscopic Heatmap App (Standalone)

PyQt5 desktop app for generating structured outputs from microscopic FPCB images.

## Outputs (Structured latent representation candidates)

For each input image `<name>.*`:

- `<name>_overlay.png`
- `<name>_heatmap.png`
- `<name>_segments.json`

For the run:

- `summary.csv` (appended in the output folder)

## Run

```bash
pip install -r requirements.txt
python main.py
```

## Build (Windows)

```bash
pip install pyinstaller
pyinstaller --onefile --windowed main.py --name fpcb-heatmap-app
```

Optional nested outputs:

- Check **Nested outputs** and set **Output Folder** to `<project_root>/outputs` so files go to `outputs/overlays/`, `outputs/heatmaps/`, `outputs/segments/`, and `outputs/summary.csv`.
- Check **Copy labeled crack mask…** to also write `*_gt_mask.png` into `outputs/overlays/` when a label exists (nested mode only).

## Labeling + training

See in-app **Help** sections 7–8, or run:

```bash
python train.py --project-root <path-to-project>
python eval.py --project-root <path> --checkpoint checkpoints/crack_deeplab_best.pt
```

## GitHub Actions EXE Artifact

Workflow file:

- `.github/workflows/fpcb-heatmap-build.yml`

After push to `main` (or manual `workflow_dispatch`), download:

- Actions → **Build FPCB Heatmap EXE** → latest run → artifact `fpcb-heatmap-app-exe`

