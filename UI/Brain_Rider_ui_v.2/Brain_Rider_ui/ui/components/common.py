"""Reusable presentation primitives — independent of BCI algorithms."""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QFrame, QLabel, QPushButton, QVBoxLayout,
                             QHBoxLayout, QWidget, QSizePolicy)
from ui.styles.theme import MUTED, MINT


def label(text='', object_name=None, word_wrap=False):
    widget=QLabel(str(text))
    if object_name:
        widget.setObjectName(object_name)
    widget.setWordWrap(word_wrap)
    return widget


def button(text, callback=None, kind='ghost'):
    widget=QPushButton(text)
    widget.setObjectName(kind)
    if callback:
        widget.clicked.connect(callback)
    widget.setCursor(Qt.PointingHandCursor)
    return widget


class Panel(QFrame):
    def __init__(self, title=None, subtitle=None, kind='panel', padding=20, parent=None):
        super().__init__(parent)
        self.setObjectName(kind)
        self.layout_box=QVBoxLayout(self)
        self.layout_box.setContentsMargins(padding,padding,padding,padding)
        self.layout_box.setSpacing(12)
        if title:
            self.layout_box.addWidget(label(title, 'sectionTitle'))
        if subtitle:
            self.layout_box.addWidget(label(subtitle, 'muted',True))


class MetricCard(Panel):
    def __init__(self, title, value='—', suffix='', accent=MINT):
        super().__init__(padding=18)
        self.layout_box.setSpacing(8)
        self.layout_box.addWidget(label(title.upper(),'eyebrow'))
        self.value=label(value,'statValue')
        self.value.setStyleSheet('color: '+accent+';')
        self.layout_box.addWidget(self.value)
        self.detail=label(suffix,'muted')
        self.layout_box.addWidget(self.detail)
        self.setMinimumWidth(155)


class CommandTile(Panel):
    def __init__(self, symbol, title, trigger, accent):
        super().__init__(kind='panelAlt',padding=17)
        header=QHBoxLayout()
        icon=label(symbol)
        icon.setStyleSheet(f'color: {accent}; font-size: 28px; font-weight: 800;')
        header.addWidget(icon)
        header.addStretch()
        self.layout_box.addLayout(header)
        self.layout_box.addWidget(label(title,'sectionTitle'))
        self.layout_box.addWidget(label(trigger,'muted',True))
        self.setMinimumHeight(143)


class StatusPill(QLabel):
    def __init__(self, text='●  OFFLINE', parent=None):
        super().__init__(text, parent)
        self.setObjectName('tagWarn')
        self.setAlignment(Qt.AlignCenter)
    def set_status(self, text, good=False):
        # 20-Hz UI refresh: only repolish when badge appearance actually changes.
        appearance='tag' if good else 'tagWarn'
        if self.text()!=text:
            self.setText(text)
        if self.objectName()!=appearance:
            self.setObjectName(appearance)
            self.style().unpolish(self)
            self.style().polish(self)


class StepCard(Panel):
    def __init__(self, number, title, description, duration, on_record):
        super().__init__(padding=19)
        row=QHBoxLayout()
        n=label(f'{number:02d}','stepNumber');n.setFixedWidth(50)
        row.addWidget(n)
        content=QVBoxLayout();content.setSpacing(5)
        content.addWidget(label(title,'sectionTitle'))
        content.addWidget(label(description,'muted',True))
        row.addLayout(content,1)
        self.state=StatusPill('NOT STARTED')
        row.addWidget(self.state)
        self.layout_box.addLayout(row)
        below=QHBoxLayout()
        self.duration=label(duration,'muted')
        below.addWidget(self.duration);below.addStretch()
        self.record=button('RECORD BASELINE  →',on_record,'ghost')
        self.record.setMinimumWidth(175)
        below.addWidget(self.record)
        self.layout_box.addLayout(below)
    def set_state(self, status, ready=False, recording=False):
        self.state.set_status(status, good=ready or recording)
        self.record.setText('RECORDING…' if recording else 'RECORD AGAIN  ↻' if ready else 'RECORD BASELINE  →')
