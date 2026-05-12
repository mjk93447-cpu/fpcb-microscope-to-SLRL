from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class HelpSection:
    title: str
    body: str


def get_help_sections() -> List[HelpSection]:
    # Intentionally plain text (not HTML/Markdown rendering) for maximum copy/paste reliability.
    return [
        HelpSection(
            title="1) What this tool does",
            body=(
                "This app generates outputs from microscopic FPCB images:\n"
                "- Crack/lead segment overlays\n"
                "- A heatmap image\n"
                "- A structured segments JSON (for downstream use)\n"
                "- A run summary CSV (appended)\n\n"
                "You can run a single image or a whole folder (batch).\n"
            ),
        ),
        HelpSection(
            title="2) Before you start",
            body=(
                "Input images\n"
                "- Supported formats: .png, .jpg/.jpeg, .bmp, .tif/.tiff\n"
                "- Recommendation: keep one lot/product in one folder.\n\n"
                "Output folder\n"
                "- For a clean run, choose an empty output folder.\n"
                "- NOTE: summary.csv is appended across runs in the same output folder.\n"
            ),
        ),
        HelpSection(
            title="3) Step-by-step SOP",
            body=(
                "Step 1) Input\n"
                "- Click Browse next to 'Input Image/Folder'.\n"
                "- Select ONE image file.\n"
                "  - If you cancel the file picker, you will be asked to select a folder instead.\n\n"
                "Step 2) Output\n"
                "- Click Browse next to 'Output Folder' and select a folder.\n"
                "- Optional: click 'Open Output Folder' to confirm you selected the right folder.\n\n"
                "Step 3) (Optional) Model checkpoint\n"
                "- If you were provided a model file, set 'Model Checkpoint (optional)' to a .pt/.pth file.\n"
                "- If you are not sure, leave it empty.\n\n"
                "Step 4) Parameters\n"
                "- Keep defaults unless you were instructed to change them.\n\n"
                "Step 5) Run\n"
                "- Click 'Start Processing'.\n"
                "- Wait until the log shows: 'Processing finished.'\n\n"
                "Step 6) Verify outputs\n"
                "- Click 'Open Output Folder'.\n"
                "- Check the outputs checklist section.\n"
            ),
        ),
        HelpSection(
            title="4) Outputs checklist",
            body=(
                "For each input image '<name>.*', the output folder must contain:\n"
                "- <name>_overlay.png\n"
                "- <name>_heatmap.png\n"
                "- <name>_segments.json\n\n"
                "Additionally, the output folder contains:\n"
                "- summary.csv\n\n"
                "summary.csv notes\n"
                "- It is appended each run. If you want a clean report, delete/move it before running.\n"
            ),
        ),
        HelpSection(
            title="5) Troubleshooting (FAQ)",
            body=(
                "No images found\n"
                "- If you selected a folder: confirm it contains supported image formats.\n"
                "- If you selected a file: confirm it is a valid image and not corrupted.\n\n"
                "Checkpoint path error\n"
                "- Clear the checkpoint field or select an existing .pt/.pth file.\n\n"
                "Results look empty (0 segments)\n"
                "- First rerun with defaults.\n"
                "- If still empty, lower Confidence Threshold slightly (example: 0.30 -> 0.25).\n"
                "- If there are too many noisy segments, increase Min Segment Length (example: 18 -> 22).\n\n"
                "App is slow\n"
                "- Batch runs depend on image size and PC performance.\n"
                "- Keep the window open and wait for 'Processing finished.'\n"
            ),
        ),
        HelpSection(
            title="6) Operational safety notes",
            body=(
                "- Do not set the output folder inside the input folder.\n"
                "- If you re-use the same output folder, previous outputs may remain.\n"
                "- For a clean run, create a new output folder or clear the old one.\n"
            ),
        ),
        HelpSection(
            title="7) Labeling (GrabCut) SOP",
            body=(
                "Open 'Open labeling…' and select your project ROOT folder (not outputs/).\n"
                "Click 'Create project folders' once.\n"
                "Put raw microscopy files under: <project>/images/\n"
                "Mask type (dropdown):\n"
                "- Crack → saves labels/masks/<name>.png + labels/meta/<name>.json\n"
                "- Lead (copper wire) → saves labels/masks/<name>_lead.png + labels/meta/<name>_lead.json\n"
                "Per image workflow:\n"
                "- Select 'ROI rect', drag a rectangle around the region of interest.\n"
                "- Click 'Init GrabCut'.\n"
                "- Use FG brush on foreground pixels, BG brush on clearly background pixels.\n"
                "- Click 'Refine' as needed.\n"
                "- Click 'Save label'.\n"
                "Draft masks auto-save every ~15s:\n"
                "- Crack: labels/masks/<name>_draft.png\n"
                "- Lead: labels/masks/<name>_lead_draft.png\n"
            ),
        ),
        HelpSection(
            title="8) Training / evaluation (CLI)",
            body=(
                "Crack-only (legacy 2-class):\n"
                "  python train.py --project-root <path> --mode crack\n"
                "  → checkpoints/crack_deeplab_best.pt\n"
                "Joint 3-class (background / lead / crack) — needs crack mask per image; optional *_lead.png:\n"
                "  python train.py --project-root <path> --mode joint\n"
                "  → checkpoints/segmentation_deeplab_3c_best.pt\n"
                "On-site lead calibration (short fine-tune, backbone frozen):\n"
                "  python train.py --project-root <path> --mode calibrate-lead --init-checkpoint <public_pretrain.pt>\n"
                "  → checkpoints/lead_calibrated_3c_best.pt\n"
                "Evaluate:\n"
                "  python eval.py --project-root <path> --checkpoint <path-to-pt> --mode joint\n"
                "  (use --mode crack for old 2-class checkpoints)\n"
                "Use the checkpoint in the main window 'Model Checkpoint (optional)' field.\n"
                "Optional: enable 'Nested outputs' and set Output Folder to <project>/outputs\n"
                "  so overlays/heatmaps/segments/ and summary.csv are organized under outputs/.\n"
            ),
        ),
        HelpSection(
            title="9) Public pre-training for lead (recommended)",
            body=(
                "Lead (copper) has strong reflectance and straight edges; learn it once on public imagery, then\n"
                "only calibrate lightly on each line-side PC.\n\n"
                "Prepare a folder (NOT inside git) like:\n"
                "  <bundle>/images/<name>.png\n"
                "  <bundle>/labels/lead_masks/<name>.png   (255 = lead)\n"
                "Run:\n"
                "  python pretrain_lead_public.py --public-root <bundle>\n"
                "Output defaults to: <bundle>/checkpoints/lead_public_3c_best.pt\n\n"
                "Then on the factory project:\n"
                "  1) Label a small set of *_lead.png for local reflectance.\n"
                "  2) python train.py --mode calibrate-lead --init-checkpoint <bundle>/checkpoints/lead_public_3c_best.pt ...\n"
                "  3) Optionally train joint 3-class with crack masks: train.py --mode joint\n"
            ),
        ),
    ]

