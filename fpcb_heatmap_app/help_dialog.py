from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QHBoxLayout, QListWidget, QListWidgetItem, QTextEdit

from help_content import get_help_sections


class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Help — SOP (Field Engineer)")
        self.setMinimumSize(820, 560)

        root = QHBoxLayout()
        self.setLayout(root)

        self.nav = QListWidget()
        self.nav.setMaximumWidth(280)
        self.nav.setFocusPolicy(Qt.StrongFocus)
        root.addWidget(self.nav)

        self.viewer = QTextEdit()
        self.viewer.setReadOnly(True)
        self.viewer.setLineWrapMode(QTextEdit.WidgetWidth)
        root.addWidget(self.viewer, 1)

        self.sections = get_help_sections()
        for s in self.sections:
            QListWidgetItem(s.title, self.nav)

        self.nav.currentRowChanged.connect(self._show_section)
        self.nav.setCurrentRow(0)

    def _show_section(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.sections):
            self.viewer.setPlainText("")
            return
        self.viewer.setPlainText(self.sections[idx].body)
        cur = self.viewer.textCursor()
        cur.movePosition(cur.Start)
        self.viewer.setTextCursor(cur)

