"""LightBrain Hybrid BCI — component-based desktop interface.

All EOG/SSVEP feature extraction, classification, and original command gates live
in eog_original.py, ssvep_original.py, hybrid_engine.py and are left unchanged.
"""
import argparse
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from ui.main_window import HybridMainWindow


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='LightBrain Hybrid BCI • EOG + SSVEP')
    parser.add_argument('--mode', choices=['demo','native','sender','dongle'], default=None,
                        help='Choose and automatically connect to a source. Omit for welcome screen.')
    parser.add_argument('--port', default='COM3')
    parser.add_argument('--fw2', action='store_true')
    parser.add_argument('--board-name', default='')
    parser.add_argument('--eog-ch', type=int, default=1)
    parser.add_argument('--ssvep-ch', type=int, default=3)
    parser.add_argument('--legacy-ssvep-row2', action='store_true',
                        help='Use source SSVEP raw data row 2, not physical CH3')
    parser.add_argument('--keyboard', action='store_true',
                        help='Explicit Windows STIMULATOR keyboard opt-in, never physical motors')
    args = parser.parse_args(argv)
    if args.eog_ch == args.ssvep_ch and not args.legacy_ssvep_row2:
        parser.error('EOG and SSVEP channels must be different')
    return args


def main():
    args = parse_args()
    # Qt attributes must be set before QApplication is instantiated.
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = HybridMainWindow(args)
    window.show()
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
