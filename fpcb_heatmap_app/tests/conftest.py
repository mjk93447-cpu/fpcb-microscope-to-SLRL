from __future__ import annotations

import sys
from pathlib import Path


# Allow `from config import AppConfig` style imports (the app assumes cwd=project folder).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

