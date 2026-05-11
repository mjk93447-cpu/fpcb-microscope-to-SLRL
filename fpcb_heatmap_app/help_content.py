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
    ]

