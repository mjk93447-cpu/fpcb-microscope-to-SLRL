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

## GitHub Actions EXE Artifact

Workflow file:
- `.github/workflows/fpcb-heatmap-build.yml`

After push to `main` (or manual `workflow_dispatch`), download:
- Actions -> `Build FPCB Heatmap EXE` -> latest run -> artifact `fpcb-heatmap-app-exe`

