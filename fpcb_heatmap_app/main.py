#!/usr/bin/env python3

import sys

from PyQt5.QtWidgets import QApplication

from gui import FpcbHeatmapGUI


def main():
    app = QApplication(sys.argv)
    window = FpcbHeatmapGUI()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())

