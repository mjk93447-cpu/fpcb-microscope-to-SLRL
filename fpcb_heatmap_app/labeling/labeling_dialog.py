from __future__ import annotations

from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from io_utils import IMAGE_EXTENSIONS
from labeling.grabcut_engine import GrabCutState, apply_scribbles, gc_mask_to_binary, init_grabcut, refine_grabcut
from labeling.label_io import draft_mask_path, load_mask_if_any, save_draft_mask, save_label
from labeling.overlay_render import blend_mask_bgr
from project_layout import LabelMeta, build_project_paths, ensure_project_dirs, label_meta_path


class Tool(Enum):
    RECT_INIT = auto()
    FG_BRUSH = auto()
    BG_BRUSH = auto()
    ERASER = auto()


class LabelImageWidget(QWidget):
    """Shows BGR image with optional mask overlay; converts mouse coords to image space."""

    def __init__(self, controller: "LabelingDialog"):
        super().__init__()
        self._c = controller
        self.setMinimumSize(640, 480)
        self.setMouseTracking(True)
        self._pixmap: QPixmap | None = None
        self._scale = 1.0
        self._off_x = 0
        self._off_y = 0
        self._img_w = 0
        self._img_h = 0

    def refresh(self) -> None:
        self._rebuild_pixmap()
        self.update()

    def _rebuild_pixmap(self) -> None:
        img = self._c._image_bgr
        if img is None:
            self._pixmap = None
            return
        mask = self._c._mask_preview
        vis = blend_mask_bgr(img, mask) if mask is not None else img
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        cw, ch2 = self.width(), self.height()
        if cw < 8 or ch2 < 8:
            self._pixmap = QPixmap.fromImage(qimg)
            return
        scale = min(cw / w, ch2 / h)
        dw, dh = int(w * scale), int(h * scale)
        self._scale = scale
        self._off_x = (cw - dw) // 2
        self._off_y = (ch2 - dh) // 2
        self._img_w = w
        self._img_h = h
        self._pixmap = QPixmap.fromImage(qimg).scaled(dw, dh, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

    def resizeEvent(self, event) -> None:  # noqa: N802
        self._rebuild_pixmap()
        super().resizeEvent(event)

    def widget_to_image(self, wx: int, wy: int) -> tuple[int, int] | None:
        if self._c._image_bgr is None or self._pixmap is None:
            return None
        x = int((wx - self._off_x) / self._scale)
        y = int((wy - self._off_y) / self._scale)
        if x < 0 or y < 0 or x >= self._img_w or y >= self._img_h:
            return None
        return x, y

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.black)
        if self._pixmap:
            p.drawPixmap(self._off_x, self._off_y, self._pixmap)
        p.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self._c.on_canvas_press(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        self._c.on_canvas_move(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._c.on_canvas_release(event)


class LabelingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Contour labeling (GrabCut)")
        self.resize(1100, 720)

        self._project_root = ""
        self._images: list[str] = []
        self._index = 0
        self._image_bgr: np.ndarray | None = None
        self._gc_state: GrabCutState | None = None
        self._mask_preview: np.ndarray | None = None
        self._dirty = False
        self._tool = Tool.RECT_INIT
        self._rect_start: tuple[int, int] | None = None
        self._pending_rect: tuple[int, int, int, int] | None = None
        self._brush_radius = 8
        self._drawing = False

        root = QVBoxLayout()
        self.setLayout(root)

        top = QHBoxLayout()
        self._project_lbl = QLabel("(no project folder)")
        btn_proj = QPushButton("Choose project folder…")
        btn_proj.clicked.connect(self._pick_project)
        btn_init = QPushButton("Create project folders")
        btn_init.clicked.connect(self._ensure_dirs)
        top.addWidget(QLabel("Project:"))
        top.addWidget(self._project_lbl, 1)
        top.addWidget(btn_proj)
        top.addWidget(btn_init)
        root.addLayout(top)

        mid = QHBoxLayout()
        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._list.currentRowChanged.connect(self._on_row_changed)
        mid.addWidget(self._list, 1)

        right = QVBoxLayout()
        self._canvas = LabelImageWidget(self)
        right.addWidget(self._canvas, 1)

        tools = QHBoxLayout()
        self._btn_rect = QPushButton("ROI rect")
        self._btn_fg = QPushButton("FG brush")
        self._btn_bg = QPushButton("BG brush")
        self._btn_er = QPushButton("Eraser")
        self._btn_init = QPushButton("Init GrabCut")
        self._btn_ref = QPushButton("Refine")
        self._btn_save = QPushButton("Save label")
        self._btn_prev = QPushButton("Prev")
        self._btn_next = QPushButton("Next")
        for b in (
            self._btn_rect,
            self._btn_fg,
            self._btn_bg,
            self._btn_er,
            self._btn_init,
            self._btn_ref,
            self._btn_save,
            self._btn_prev,
            self._btn_next,
        ):
            tools.addWidget(b)
        self._spin_r = QSpinBox()
        self._spin_r.setRange(1, 64)
        self._spin_r.setValue(self._brush_radius)
        self._spin_r.valueChanged.connect(lambda v: setattr(self, "_brush_radius", int(v)))
        tools.addWidget(QLabel("r"))
        tools.addWidget(self._spin_r)
        right.addLayout(tools)

        self._btn_rect.clicked.connect(lambda: self._set_tool(Tool.RECT_INIT))
        self._btn_fg.clicked.connect(lambda: self._set_tool(Tool.FG_BRUSH))
        self._btn_bg.clicked.connect(lambda: self._set_tool(Tool.BG_BRUSH))
        self._btn_er.clicked.connect(lambda: self._set_tool(Tool.ERASER))
        self._btn_init.clicked.connect(self._init_grabcut)
        self._btn_ref.clicked.connect(self._refine)
        self._btn_save.clicked.connect(self._save_label)
        self._btn_prev.clicked.connect(self._prev)
        self._btn_next.clicked.connect(self._next)

        mid.addLayout(right, 3)
        root.addLayout(mid, 1)

        self._hint = QLabel(
            "Choose project → Create folders → add images under images/ → ROI rect → Init GrabCut → "
            "FG/BG brushes → Refine → Save. Draft auto-saves every 15s while editing."
        )
        self._hint.setWordWrap(True)
        root.addWidget(self._hint)

        self._timer = QTimer(self)
        self._timer.setInterval(15000)
        self._timer.timeout.connect(self._autosave_draft)
        self._timer.start()

    def _set_tool(self, t: Tool) -> None:
        self._tool = t
        self._hint.setText(f"Tool: {t.name}")

    def _pick_project(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select project root folder")
        if not d:
            return
        self._project_root = d
        self._project_lbl.setText(d)
        self._reload_image_list()

    def _ensure_dirs(self) -> None:
        if not self._project_root:
            QMessageBox.warning(self, "Project", "Choose a project folder first.")
            return
        p = build_project_paths(Path(self._project_root))
        ensure_project_dirs(p)
        QMessageBox.information(self, "Project", "Folders created: images/, labels/, outputs/, checkpoints/, runs/.")
        self._reload_image_list()

    def _reload_image_list(self) -> None:
        self._list.clear()
        self._images = []
        if not self._project_root:
            return
        img_dir = Path(self._project_root) / "images"
        if not img_dir.is_dir():
            return
        for f in sorted(img_dir.iterdir()):
            if f.suffix.lower() in IMAGE_EXTENSIONS:
                self._images.append(str(f))
                QListWidgetItem(f.name, self._list)
        if self._images:
            self._list.setCurrentRow(0)

    def _on_row_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._images):
            return
        self._autosave_draft()
        self._index = row
        self._load_current_image()

    def _load_current_image(self) -> None:
        path = self._images[self._index]
        img = cv2.imread(path)
        if img is None:
            QMessageBox.warning(self, "Image", f"Could not read: {path}")
            return
        self._image_bgr = img
        self._gc_state = None
        pr = Path(self._project_root)
        self._mask_preview = load_mask_if_any(pr, Path(path))
        self._canvas.refresh()

    def _current_image_path(self) -> Path:
        return Path(self._images[self._index])

    def on_canvas_press(self, event) -> None:
        if self._image_bgr is None:
            return
        pos = self._canvas.widget_to_image(event.x(), event.y())
        if pos is None:
            return
        if event.button() == Qt.LeftButton:
            self._drawing = True
            if self._tool == Tool.RECT_INIT:
                self._rect_start = pos
            elif self._tool in (Tool.FG_BRUSH, Tool.BG_BRUSH, Tool.ERASER):
                self._apply_brush(pos[0], pos[1])

    def on_canvas_move(self, event) -> None:
        if not self._drawing or self._image_bgr is None:
            return
        pos = self._canvas.widget_to_image(event.x(), event.y())
        if pos is None:
            return
        if self._tool in (Tool.FG_BRUSH, Tool.BG_BRUSH, Tool.ERASER) and event.buttons() & Qt.LeftButton:
            self._apply_brush(pos[0], pos[1])

    def on_canvas_release(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        self._drawing = False
        if self._image_bgr is None:
            return
        pos = self._canvas.widget_to_image(event.x(), event.y())
        if pos is None:
            return
        if self._tool == Tool.RECT_INIT and self._rect_start:
            x0, y0 = self._rect_start
            x1, y1 = pos
            self._rect_start = None
            x, y = min(x0, x1), min(y0, y1)
            w, h = abs(x1 - x0), abs(y1 - y0)
            if w < 5 or h < 5:
                self._hint.setText("ROI too small.")
                return
            self._pending_rect = (x, y, w, h)
            self._hint.setText(f"ROI ({x},{y},{w},{h}). Click Init GrabCut.")

    def _apply_brush(self, cx: int, cy: int) -> None:
        if not self._gc_state or self._gc_state.mask_gc is None or self._image_bgr is None:
            self._hint.setText("Init GrabCut before brushing.")
            return
        circ = np.zeros(self._image_bgr.shape[:2], dtype=np.uint8)
        cv2.circle(circ, (cx, cy), self._brush_radius, 255, thickness=-1)
        if self._tool == Tool.FG_BRUSH:
            apply_scribbles(self._gc_state, fg_scribble=circ)
        elif self._tool == Tool.BG_BRUSH:
            apply_scribbles(self._gc_state, bg_scribble=circ)
        else:
            self._gc_state.mask_gc[circ > 0] = cv2.GC_PR_BGD
        refine_grabcut(self._image_bgr, self._gc_state, iters=1)
        self._mask_preview = gc_mask_to_binary(self._gc_state.mask_gc)
        self._canvas.refresh()
        self._dirty = True

    def _init_grabcut(self) -> None:
        if self._image_bgr is None:
            return
        if not self._pending_rect:
            QMessageBox.information(self, "GrabCut", "Draw a ROI rectangle first (ROI rect tool, drag on image).")
            return
        self._gc_state = init_grabcut(self._image_bgr, self._pending_rect, iters=3)
        self._mask_preview = gc_mask_to_binary(self._gc_state.mask_gc)
        self._canvas.refresh()
        self._dirty = True

    def _refine(self) -> None:
        if self._image_bgr is None or not self._gc_state:
            return
        refine_grabcut(self._image_bgr, self._gc_state, iters=3)
        self._mask_preview = gc_mask_to_binary(self._gc_state.mask_gc)
        self._canvas.refresh()
        self._dirty = True

    def _save_label(self) -> None:
        if not self._project_root or not self._images:
            return
        if self._mask_preview is None:
            QMessageBox.warning(self, "Save", "No mask to save.")
            return
        pr = Path(self._project_root)
        ip = self._current_image_path()
        mask_path = pr / "labels" / "masks" / f"{ip.stem}.png"
        meta = LabelMeta.new(
            image_path=ip,
            project_root=pr,
            mask_png_path=mask_path,
            classes=["crack"],
            tool={"name": "fpcb-microscope-to-SLRL", "version": "0.2"},
            grabcut={"roi": self._pending_rect},
        )
        save_label(pr, ip, self._mask_preview, meta)
        dp = draft_mask_path(pr, ip)
        if dp.exists():
            try:
                dp.unlink()
            except OSError:
                pass
        self._dirty = False
        QMessageBox.information(self, "Saved", f"{mask_path}\n{label_meta_path(pr, ip)}")

    def _autosave_draft(self) -> None:
        if not self._dirty or not self._project_root or not self._images:
            return
        if self._mask_preview is None:
            return
        try:
            save_draft_mask(Path(self._project_root), self._current_image_path(), self._mask_preview)
        except OSError:
            pass

    def _prev(self) -> None:
        if self._index > 0:
            self._list.setCurrentRow(self._index - 1)

    def _next(self) -> None:
        if self._index < len(self._images) - 1:
            self._list.setCurrentRow(self._index + 1)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._autosave_draft()
        super().closeEvent(event)
