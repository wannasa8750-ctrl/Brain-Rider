"""Session events and experimental transparency."""
from PyQt5.QtWidgets import QWidget,QVBoxLayout,QPlainTextEdit,QHBoxLayout
from ui.components.common import Panel,label,button

class LogsPage(QWidget):
    def __init__(self):
        super().__init__()
        root=QVBoxLayout(self);root.setContentsMargins(2,4,2,10);root.setSpacing(16)
        root.addWidget(label('SESSION RECORDS','eyebrow'))
        root.addWidget(label('Event & Safety Log','largeTitle'))
        root.addWidget(label('Acquisition, baseline, classification, arbitration, and STOP events are recorded locally to logs/events_*.csv','muted',True))
        panel=Panel('Live session event stream')
        self.log=QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(800)
        self.log.setPlaceholderText('No events have been recorded yet.')
        panel.layout_box.addWidget(self.log,1)
        root.addWidget(panel,1)
    def append_event(self,line):
        self.log.appendPlainText(str(line))
