"""Launch / overview page."""
from PyQt5.QtWidgets import QWidget,QVBoxLayout,QHBoxLayout,QGridLayout,QComboBox,QLineEdit
from PyQt5.QtCore import Qt
from ui.components.common import Panel,MetricCard,CommandTile,label,button,StatusPill
from ui.components.brain_visual import BrainVisual
from ui.styles.theme import MINT,CYAN,PURPLE,ORANGE


class OverviewPage(QWidget):
    def __init__(self, on_connect, on_baseline, on_monitor):
        super().__init__()
        root=QVBoxLayout(self)
        root.setContentsMargins(2,4,2,10)
        root.setSpacing(17)
        root.addWidget(label('NEURAL CONTROL PLATFORM', 'eyebrow'))
        root.addWidget(label('Hybrid BCI Workspace','largeTitle'))
        root.addWidget(label('A unified control interface for Fp1 EOG and O1 SSVEP • research-grade desktop simulator','muted',True))
        hero=Panel(kind='hero',padding=24)
        hero_row=QHBoxLayout();hero_row.setSpacing(20)
        info=QVBoxLayout();info.setSpacing(15)
        info.addWidget(label('DUAL-MODAL NEURAL INTERFACE','eyebrow'))
        title=label('Two signals.\nOne control system.','largeTitle');title.setWordWrap(True)
        info.addWidget(title)
        desc=label('Intentional blink for FORWARD. Visual stimulation at 9 Hz or 15 Hz for LEFT and RIGHT. Original extraction and command gates stay unchanged.','muted',True)
        desc.setMinimumWidth(270)
        info.addWidget(desc)
        self.hero_status=StatusPill('●  NOT CONNECTED');info.addWidget(self.hero_status,0,Qt.AlignLeft)
        acts=QHBoxLayout()
        acts.addWidget(button('GO TO CALIBRATION  →',on_baseline,'primary'))
        acts.addWidget(button('OPEN LIVE MONITOR',on_monitor,'ghost'))
        info.addLayout(acts);info.addStretch()
        hero_row.addLayout(info,4)
        art=BrainVisual();hero_row.addWidget(art,5)
        hero.layout_box.addLayout(hero_row)
        root.addWidget(hero)
        stats=QHBoxLayout();stats.setSpacing(12)
        self.metric_signal=MetricCard('Signal source','OFFLINE','Acquisition channel status',MINT)
        self.metric_fs=MetricCard('Sampling rate','—','Ganglion / demo',CYAN)
        self.metric_phase=MetricCard('Calibration','0 / 3','SSVEP + EOG',PURPLE)
        self.metric_control=MetricCard('Safety state','DISARMED','STOP enabled',ORANGE)
        for x in (self.metric_signal,self.metric_fs,self.metric_phase,self.metric_control):stats.addWidget(x,1)
        root.addLayout(stats)
        lower=QHBoxLayout();lower.setSpacing(14)
        connect=Panel('Connect acquisition','Select your signal source. Connection uses one session for both EOG and SSVEP.')
        combo_row=QHBoxLayout()
        combo_row.addWidget(label('SOURCE','eyebrow'))
        self.source=QComboBox()
        self.source.addItem('Demo — synthetic data','demo')
        self.source.addItem('Ganglion — Native BLE','native')
        self.source.addItem('Ganglion — Sender-compatible BLE','sender')
        self.source.addItem('Ganglion — BLED112 Dongle','dongle')
        combo_row.addWidget(self.source,1)
        connect.layout_box.addLayout(combo_row)
        port_row=QHBoxLayout()
        port_row.addWidget(label('DONGLE PORT','eyebrow'))
        self.port=QLineEdit('COM3');self.port.setPlaceholderText('COM3')
        port_row.addWidget(self.port,1)
        connect.layout_box.addLayout(port_row)
        self.connect_button=button('CONNECT SOURCE  →',self._connect,'primary')
        connect.layout_box.addWidget(self.connect_button)
        self.connect_info=label('Connected data is processed locally.','muted',True)
        connect.layout_box.addWidget(self.connect_info)
        lower.addWidget(connect,1)
        action=Panel('Neural command mapping','In this hybrid mode each modality retains its original command interpretation.')
        tiles=QHBoxLayout();tiles.setSpacing(8)
        for a,b,c,d in [('↑','FORWARD','Fp1 · Intentional blink',MINT),('↶','LEFT','O1 · SSVEP 9 Hz',CYAN),('↷','RIGHT','O1 · SSVEP 15 Hz',PURPLE)]:
            tiles.addWidget(CommandTile(a,b,c,d),1)
        action.layout_box.addLayout(tiles)
        lower.addWidget(action,2)
        root.addLayout(lower)
        root.addStretch()
        self.on_connect=on_connect
        self.source.currentIndexChanged.connect(self._update_port)
        self._update_port()

    def _update_port(self,*_):
        mode=self.source.currentData()
        self.port.setEnabled(mode in ('dongle','sender'))

    def _connect(self):
        self.on_connect(self.source.currentData(),self.port.text().strip() or 'COM3')

    def set_connected(self,source_status,active,calibrated,armed,fs):
        self.hero_status.set_status('●  SOURCE CONNECTED' if active else '●  SOURCE OFFLINE',active)
        self.metric_signal.value.setText('ONLINE' if active else 'OFFLINE')
        self.metric_fs.value.setText(f'{fs} Hz' if active else '—')
        self.metric_phase.value.setText(f'{calibrated} / 3')
        self.metric_control.value.setText('ARMED' if armed else 'DISARMED')
        self.connect_info.setText(source_status)
        self.connect_button.setText('RECONNECT SOURCE  ↻' if active else 'CONNECT SOURCE  →')
