"""Central design tokens and QSS. Change styling here without touching BCI logic."""
BG='#070E1A'
PANEL='#101D2C'
PANEL_2='#132438'
STROKE='#263C53'
CYAN='#42D9E8'
MINT='#5DE0B3'
LIME='#B9F86C'
PURPLE='#A6A6FF'
ORANGE='#FFBC78'
RED='#FF7182'
TEXT='#EEF5FC'
MUTED='#91A9BD'
DARK='#0A1422'

QSS = '''
* { font-family: 'Segoe UI', 'Inter', 'Arial'; font-size: 12px; outline: none; }
QMainWindow, QWidget#appRoot { background: #070E1A; color: #EEF5FC; }
QWidget { color: #EEF5FC; }
QScrollArea { border: none; background: transparent; }
QScrollArea > QWidget > QWidget { background: transparent; }
QFrame#sideBar { background: #0A1422; border-right: 1px solid #1D3348; }
QFrame#panel { background: #101D2C; border: 1px solid #263C53; border-radius: 16px; }
QFrame#panelAlt { background: #132438; border: 1px solid #263C53; border-radius: 16px; }
QFrame#hero { background: #102235; border: 1px solid #2C526B; border-radius: 20px; }
QLabel#brand { color: #EAFBFF; font-size: 21px; font-weight: 800; }
QLabel#smallBrand { color: #5DE0B3; letter-spacing: 1px; font-size: 10px; font-weight: 800; }
QLabel#pageTitle { color: #F1F8FF; font-size: 25px; font-weight: 800; }
QLabel#largeTitle { color: #F1F8FF; font-size: 31px; font-weight: 800; }
QLabel#sectionTitle { color: #EFF8FD; font-size: 16px; font-weight: 750; }
QLabel#muted { color: #91A9BD; font-size: 12px; }
QLabel#eyebrow { color: #5DE0B3; font-size: 11px; font-weight: 800; }
QLabel#statValue { color: #F2FCFF; font-size: 24px; font-weight: 800; }
QLabel#commandValue { color: #EEF5FC; font-size: 30px; font-weight: 800; }
QLabel#stepNumber { color: #5DE0B3; font-size: 28px; font-weight: 850; }
QLabel#tag { border: 1px solid #31586B; background: #163144; border-radius: 10px; padding: 5px 10px; color: #68E8D4; font-size: 11px; font-weight: 700; }
QLabel#tagWarn { border: 1px solid #6F4B4B; background: #33222B; border-radius: 10px; padding: 5px 10px; color: #FF9AA7; font-size: 11px; font-weight: 700; }
QPushButton { background: #1A3148; color: #EAF7FF; border: 1px solid #35536C; border-radius: 10px; padding: 11px 14px; font-weight: 650; }
QPushButton:hover { background: #25465F; border-color: #5E88A1; }
QPushButton:pressed { background: #162838; }
QPushButton:disabled { color: #668196; background: #111E2D; border-color: #263647; }
QPushButton#primary { background: #4ADBC1; color: #062022; border-color: #4ADBC1; font-weight: 800; }
QPushButton#primary:hover { background: #7AEBD3; }
QPushButton#danger { background: #5B2736; color: #FFCAD1; border: 1px solid #B44A66; font-size: 14px; font-weight: 800; }
QPushButton#danger:hover { background: #773044; }
QPushButton#ghost { background: #122435; border: 1px solid #294158; }
QPushButton#nav { text-align: left; padding: 14px 17px; border: 1px solid transparent; background: transparent; color: #89A2B8; border-radius: 12px; font-size: 13px; }
QPushButton#nav:hover { background: #14283B; color: #E9F7FF; }
QPushButton#nav[active='true'] { background: #163444; border: 1px solid #286474; color: #78E9D4; }
QComboBox, QLineEdit { background: #0E1C2C; border: 1px solid #324D65; border-radius: 9px; padding: 9px 12px; min-height: 20px; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox QAbstractItemView { background: #102437; color: #EAF7FF; selection-background-color: #266A72; border: 1px solid #375468; }
QProgressBar { color: #EAFBFE; text-align: center; border: 1px solid #2B445A; border-radius: 8px; background: #0B1725; height: 20px; }
QProgressBar::chunk { background: #5DE0B3; border-radius: 7px; }
QCheckBox { color: #C3D8E6; spacing: 9px; }
QCheckBox::indicator { width: 16px; height: 16px; }
QPlainTextEdit { background: #091523; color: #BED7E3; border: 1px solid #284055; border-radius: 10px; padding: 9px; font-family: 'Consolas',monospace; font-size: 11px; }
QSplitter::handle { background: transparent; }
QScrollBar:vertical { width: 9px; background: transparent; }
QScrollBar::handle:vertical { background: #2A485D; border-radius: 4px; min-height: 28px; }
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical { height: 0px; }
'''
