import os
import subprocess
import sys
from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config import AppConfig
from help_dialog import HelpDialog
from io_utils import ensure_dir, list_images
from labeling.labeling_dialog import LabelingDialog
from pipeline import FpcbProcessor


class ProcessingThread(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)
    done_signal = pyqtSignal()

    def __init__(self, input_path: str, output_dir: str, checkpoint_path: str, cfg: AppConfig):
        super().__init__()
        self.input_path = input_path
        self.output_dir = Path(output_dir)
        self.checkpoint_path = checkpoint_path.strip() or None
        self.cfg = cfg
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    def run(self):
        images = list_images(self.input_path)
        if not images:
            self.log_signal.emit("No images found in selected input.")
            self.done_signal.emit()
            return

        ensure_dir(self.output_dir)
        processor = FpcbProcessor(self.cfg, checkpoint_path=self.checkpoint_path)
        self.log_signal.emit(f"Inference engine: {processor.infer.info()}")

        total = len(images)
        for idx, image_path in enumerate(images, start=1):
            if self._stop_flag:
                self.log_signal.emit("Processing stopped by user.")
                break
            try:
                result = processor.process_image(image_path, self.output_dir)
                self.log_signal.emit(
                    f"[{idx}/{total}] {result.image_name} -> segments: {result.total_segments} "
                    f"(lead={result.lead_segments}, crack={result.crack_segments})"
                )
            except Exception as exc:
                self.log_signal.emit(f"[{idx}/{total}] Failed: {image_path.name} ({exc})")
            self.progress_signal.emit(idx, total)

        self.done_signal.emit()


class FpcbHeatmapGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.thread = None
        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle("FPCB Microscopic Heatmap App")
        self.setGeometry(120, 120, 920, 620)

        root = QVBoxLayout()

        self.nested_chk = QCheckBox("Nested outputs (select <project>/outputs folder)")
        self.export_gt_chk = QCheckBox("Copy labeled crack mask to outputs as *_gt_mask.png (nested only)")
        root.addWidget(self.nested_chk)
        root.addWidget(self.export_gt_chk)

        self.label_btn = QPushButton("Open labeling…")
        self.label_btn.clicked.connect(self.open_labeling)
        row_top = QHBoxLayout()
        row_top.addWidget(self.label_btn)
        row_top.addStretch()
        root.addLayout(row_top)

        root.addLayout(self._path_row("Input Image/Folder:", "input_edit", self.pick_input))
        root.addLayout(self._path_row("Output Folder:", "output_edit", self.pick_output))
        root.addLayout(self._path_row("Model Checkpoint (optional):", "ckpt_edit", self.pick_ckpt, is_file=True))

        grid = QGridLayout()
        self.conf_edit = QLineEdit("0.30")
        self.min_len_edit = QLineEdit("18")
        self.min_area_edit = QLineEdit("8")
        self.blur_edit = QLineEdit("21")
        self.crack_weight_edit = QLineEdit("2.4")
        self.lead_weight_edit = QLineEdit("1.0")

        grid.addWidget(QLabel("Confidence Threshold"), 0, 0)
        grid.addWidget(self.conf_edit, 0, 1)
        grid.addWidget(QLabel("Min Segment Length"), 0, 2)
        grid.addWidget(self.min_len_edit, 0, 3)

        grid.addWidget(QLabel("Min Component Area"), 1, 0)
        grid.addWidget(self.min_area_edit, 1, 1)
        grid.addWidget(QLabel("Heatmap Blur Kernel"), 1, 2)
        grid.addWidget(self.blur_edit, 1, 3)

        grid.addWidget(QLabel("Crack Weight"), 2, 0)
        grid.addWidget(self.crack_weight_edit, 2, 1)
        grid.addWidget(QLabel("Lead Weight"), 2, 2)
        grid.addWidget(self.lead_weight_edit, 2, 3)
        root.addLayout(grid)

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Processing")
        self.stop_btn = QPushButton("Stop")
        self.open_out_btn = QPushButton("Open Output Folder")
        self.help_btn = QPushButton("Help")

        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_processing)
        self.stop_btn.clicked.connect(self.stop_processing)
        self.open_out_btn.clicked.connect(self.open_output_folder)
        self.help_btn.clicked.connect(self.open_help)

        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.open_out_btn)
        btn_layout.addWidget(self.help_btn)
        root.addLayout(btn_layout)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        root.addWidget(self.log_edit)

        self.setLayout(root)

    def _path_row(self, label, edit_name, callback, is_file=False):
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label))
        edit = QLineEdit()
        setattr(self, edit_name, edit)
        layout.addWidget(edit)
        btn = QPushButton("Browse")
        btn.clicked.connect(callback)
        layout.addWidget(btn)
        if is_file:
            edit.setPlaceholderText("Optional .pt/.pth")
        return layout

    def pick_input(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image (or Cancel for Folder)", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if file_path:
            self.input_edit.setText(file_path)
            return
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            self.input_edit.setText(folder)

    def pick_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_edit.setText(folder)

    def pick_ckpt(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Checkpoint", "", "Torch Model (*.pt *.pth)")
        if file_path:
            self.ckpt_edit.setText(file_path)

    def _build_config(self):
        output_str = self.output_edit.text().strip()
        if not output_str:
            raise ValueError("Output Folder is empty")

        errors = []

        def _get_float(label: str, field: QLineEdit) -> float:
            try:
                return float(field.text().strip())
            except Exception:
                errors.append(f"- {label}: must be a number (got: {field.text()!r})")
                return 0.0

        def _get_int(label: str, field: QLineEdit) -> int:
            try:
                return int(field.text().strip())
            except Exception:
                errors.append(f"- {label}: must be an integer (got: {field.text()!r})")
                return 0

        confidence_threshold = _get_float("Confidence Threshold", self.conf_edit)
        min_segment_length = _get_float("Min Segment Length", self.min_len_edit)
        min_component_area = _get_int("Min Component Area", self.min_area_edit)
        heatmap_blur_kernel = _get_int("Heatmap Blur Kernel", self.blur_edit)
        crack_weight = _get_float("Crack Weight", self.crack_weight_edit)
        lead_weight = _get_float("Lead Weight", self.lead_weight_edit)

        if errors:
            raise ValueError("\n".join(errors))

        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError("- Confidence Threshold: must be between 0.0 and 1.0")
        if min_segment_length <= 0:
            raise ValueError("- Min Segment Length: must be > 0")
        if min_component_area <= 0:
            raise ValueError("- Min Component Area: must be > 0")
        if heatmap_blur_kernel <= 0:
            raise ValueError("- Heatmap Blur Kernel: must be > 0")
        if crack_weight < 0:
            raise ValueError("- Crack Weight: must be >= 0")
        if lead_weight < 0:
            raise ValueError("- Lead Weight: must be >= 0")

        return AppConfig(
            confidence_threshold=confidence_threshold,
            min_segment_length=min_segment_length,
            min_component_area=min_component_area,
            heatmap_blur_kernel=heatmap_blur_kernel,
            crack_weight=crack_weight,
            lead_weight=lead_weight,
            output_dir=Path(output_str),
            nested_project_layout=self.nested_chk.isChecked(),
            export_gt_mask_when_labeled=self.export_gt_chk.isChecked(),
        )

    def open_labeling(self):
        dlg = LabelingDialog(self)
        dlg.exec_()

    def open_help(self):
        dlg = HelpDialog(self)
        dlg.exec_()

    def open_output_folder(self):
        out_dir = self.output_edit.text().strip()
        if not out_dir:
            QMessageBox.information(self, "Output Folder", "Please click Browse and choose an output folder first.")
            return
        if not os.path.exists(out_dir):
            QMessageBox.warning(self, "Output Folder", "Selected output folder does not exist.")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(out_dir)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.run(["open", out_dir], check=False)
            else:
                subprocess.run(["xdg-open", out_dir], check=False)
        except Exception as exc:
            QMessageBox.warning(self, "Output Folder", f"Failed to open output folder: {exc}")

    def start_processing(self):
        input_path = self.input_edit.text().strip()
        output_dir = self.output_edit.text().strip()
        ckpt = self.ckpt_edit.text().strip()

        if not input_path or not output_dir:
            QMessageBox.warning(
                self,
                "Missing Input",
                "Please click Browse to choose:\n- Input Image/Folder\n- Output Folder",
            )
            return
        if not os.path.exists(input_path):
            QMessageBox.warning(self, "Invalid Input", "Selected input path does not exist.")
            return
        if ckpt and not os.path.exists(ckpt):
            QMessageBox.warning(self, "Invalid Checkpoint", "Checkpoint file does not exist.")
            return

        try:
            cfg = self._build_config()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Options", f"Invalid options:\n{exc}")
            return

        self.thread = ProcessingThread(input_path, output_dir, ckpt, cfg)
        self.thread.log_signal.connect(self.log)
        self.thread.progress_signal.connect(self.update_progress)
        self.thread.done_signal.connect(self.finish_processing)
        self.thread.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.log("Processing started.")

    def stop_processing(self):
        if self.thread is not None:
            self.thread.stop()
            self.log("Stopping...")

    def finish_processing(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log("Processing finished.")
        self.thread = None

    def update_progress(self, current: int, total: int):
        self.log(f"Progress: {current}/{total}")

    def log(self, message: str):
        self.log_edit.append(message)

