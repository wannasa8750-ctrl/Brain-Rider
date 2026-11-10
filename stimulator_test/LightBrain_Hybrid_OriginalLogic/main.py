"""LightBrain Hybrid BCI · ORIGINAL LOGIC PRESERVED · EOG W + SSVEP A/D.
Research desktop STIMULATOR / in-app simulation. No physical wheelchair connection.
"""
import argparse
import sys
import time
from pathlib import Path
import numpy as np

from PyQt5.QtCore import Qt,QTimer
from PyQt5.QtGui import QFont,QKeySequence
from PyQt5.QtWidgets import (QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,
    QGridLayout,QLabel,QPushButton,QProgressBar,QComboBox,QPlainTextEdit,QScrollArea,
    QSplitter,QFrame,QCheckBox,QShortcut,QMessageBox)
import pyqtgraph as pg

from acquisition import Acquisition
from hybrid_engine import HybridEngine
from desktop_output import DesktopOutput
import eog_original as E
import ssvep_original as S

BG='#080e18'; PANEL='#111c2b'; BORDER='#24405a'; CYAN='#41d9ef'; GREEN='#38db9b'; RED='#ff6374'
YELLOW='#ffd16b'; TEXT='#edf4fa'; MUTED='#9eb1c5'

STYLE=f'''
QMainWindow,QWidget {{background:{BG};color:{TEXT};font-family:'Segoe UI';font-size:11px}}
QFrame#panel {{background:{PANEL};border:1px solid {BORDER};border-radius:10px}}
QLabel#heading {{font-size:23px;font-weight:700;color:{CYAN}}}
QLabel#section {{font-size:13px;font-weight:700;color:{CYAN};padding-top:8px}}
QPushButton {{background:#213852;color:{TEXT};border:1px solid #345776;border-radius:7px;padding:8px 10px;font-weight:600}}
QPushButton:hover {{background:#315471}}
QPushButton:disabled {{background:#142234;color:#688096;border-color:#203043}}
QPushButton#danger {{background:#632734;border:1px solid {RED};font-weight:bold;font-size:15px}}
QPushButton#good {{background:#124d42;border:1px solid {GREEN}}}
QProgressBar {{border:1px solid #2f4a61;border-radius:5px;background:#0c1724;text-align:center;height:17px}}
QProgressBar::chunk {{background:{CYAN};border-radius:4px}}
QComboBox {{background:#1c344e;padding:7px;border:1px solid #3a5776;border-radius:6px}}
QPlainTextEdit {{background:#060f18;border:1px solid #314e66;border-radius:6px;color:#b9d5e6}}
QCheckBox {{color:{YELLOW};spacing:8px}}
'''

class Window(QMainWindow):
    def __init__(self,args):
        super().__init__()
        self.args=args
        self.setWindowTitle('LightBrain · Hybrid BCI · Original EOG/SSVEP Logic')
        self.resize(1530,900)
        self.engine=HybridEngine(log_dir=Path(__file__).parent/'logs')
        self.output=DesktopOutput(enabled=False)
        self.source=Acquisition(mode=args.mode,eog_ch=args.eog_ch,ssvep_ch=args.ssvep_ch,
                                port=args.port,fw2=args.fw2,board_name=args.board_name,
                                legacy_row2=args.legacy_ssvep_row2)
        self._last_phase=None
        self._source_error_seen=None
        self._last_chart=0.0
        self._last_psd=0.0
        self._build()
        self.source.start()
        self.timer=QTimer(self)
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(S.UPDATE_MS)  # ORIGINAL SSVEP UPDATE_MS: 50ms
        QShortcut(QKeySequence(Qt.Key_Space),self,activated=self.emergency)
        if args.keyboard:
            self.keyboard_box.setChecked(True)
        self.log_line('Original EOG/SSVEP processing and original command gates loaded.')
        self.log_line('SSVEP: 5.0s epoch / 5.0s non-overlapping decision step (original).')
        self.log_line('EOG: one intentional blink → W 1s → cooldown 2s after key release (original).')

    def section(self,layout,title):
        l=QLabel(title);l.setObjectName('section');layout.addWidget(l)
        return l

    def panel(self):
        f=QFrame();f.setObjectName('panel')
        v=QVBoxLayout(f);v.setContentsMargins(14,12,14,12);v.setSpacing(9)
        return f,v

    def plot(self,title,color):
        p=pg.PlotWidget(title=title,background='#0a1320')
        p.setMinimumHeight(150)
        p.showGrid(x=True,y=True,alpha=.20)
        p.setLabel('bottom','Time','s')
        p.setLabel('left','Amplitude')
        return p,p.plot(pen=pg.mkPen(color,width=1.5))

    def _build(self):
        self.setStyleSheet(STYLE)
        central=QWidget();self.setCentralWidget(central)
        outer=QVBoxLayout(central);outer.setContentsMargins(14,12,14,12);outer.setSpacing(9)
        header=QHBoxLayout()
        heading=QLabel('◈ LIGHTBRAIN  /  HYBRID BCI');heading.setObjectName('heading')
        header.addWidget(heading)
        hdr=QLabel('EOG Fp1  ·  SSVEP O1  ·  ORIGINAL ALGORITHMS  ·  RESEARCH / STIMULATOR')
        hdr.setStyleSheet(f'color:{MUTED};font-weight:600');header.addWidget(hdr);header.addStretch()
        outer.addLayout(header)
        self.conn=QLabel('Starting source…');self.conn.setStyleSheet(f'color:{YELLOW};padding:4px')
        outer.addWidget(self.conn)
        splitter=QSplitter(Qt.Horizontal);splitter.setChildrenCollapsible(False)
        outer.addWidget(splitter,1)
        left,_=self.panel();splitter.addWidget(left)
        lay=left.layout()
        self.command=QLabel('■  STOP / DISARMED')
        self.command.setFont(QFont('Segoe UI',20,QFont.Bold))
        self.command.setStyleSheet(f'color:{RED};padding:10px;background:#211620;border-radius:8px')
        lay.addWidget(self.command)
        self.status=QLabel('EOG: —  |  SSVEP: —  |  200 Hz  |  STOP')
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f'color:{MUTED};padding:4px');lay.addWidget(self.status)
        g=QGridLayout();g.setSpacing(8)
        self.curves={}
        for idx,(key,title,color) in enumerate([
            ('eog_raw','Fp1 · RAW EOG · µV','#8799fc'),
            ('eog_filt','Fp1 · FILTERED EOG · 0.1–10 Hz',GREEN),
            ('ss_raw','O1 · RAW EEG · original scaling',YELLOW),
            ('ss_filt','O1 · FILTERED EEG · 1–40 Hz',CYAN),
        ]):
            pw,curve=self.plot(title,color)
            g.addWidget(pw,idx//2,idx%2)
            self.curves[key]=curve
        lay.addLayout(g,3)
        psd,pc=self.plot('SSVEP · WELCH PSD / PSDA · 9Hz / 15Hz',CYAN)
        psd.setLabel('bottom','Frequency','Hz')
        psd.setLabel('left','Power','a.u./Hz')
        self.curves['psd']=pc
        lay.addWidget(psd,1)
        self.score_label=QLabel('9 Hz: —    15 Hz: —    PSDA null: waiting')
        self.score_label.setStyleSheet(f'color:{CYAN};font-weight:600')
        lay.addWidget(self.score_label)
        self.virtual=QLabel('VIRTUAL FORWARD DISTANCE: 0.00 / 10.00 m')
        lay.addWidget(self.virtual)
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMinimumWidth(380);splitter.addWidget(scroll)
        side,_=self.panel();scroll.setWidget(side)
        r=side.layout()
        self.section(r,'01  /  SSVEP NULL BASELINE  —  30s')
        self.ss_btn=QPushButton('▶ Record SSVEP Null / Eyes Away (30s)')
        self.ss_btn.clicked.connect(lambda:self.start_phase('ssvep'));r.addWidget(self.ss_btn)
        self.section(r,'02  /  EOG NATURAL BLINK  —  20s')
        self.natural_btn=QPushButton('▶ Record Natural EOG Blinks (20s)')
        self.natural_btn.clicked.connect(lambda:self.start_phase('natural'));r.addWidget(self.natural_btn)
        self.section(r,'03  /  EOG INTENTIONAL BLINK  —  20s')
        self.intent_btn=QPushButton('▶ Record Intentional EOG Blinks (20s)')
        self.intent_btn.clicked.connect(lambda:self.start_phase('intentional'));r.addWidget(self.intent_btn)
        self.progress=QProgressBar();self.progress.setRange(0,100);r.addWidget(self.progress)
        self.calibration_label=QLabel('SSVEP: —  |  NATURAL: —  |  INTENTIONAL: —')
        self.calibration_label.setWordWrap(True);r.addWidget(self.calibration_label)
        opts=QHBoxLayout()
        opts.addWidget(QLabel('EOG sensitivity:'))
        self.sens=QComboBox()
        self.sens.addItem('BALANCED (original default)',3.0)
        self.sens.addItem('SENSITIVE',2.0)
        self.sens.addItem('STRICT',4.5)
        self.sens.currentIndexChanged.connect(self.change_sensitivity)
        opts.addWidget(self.sens,1);r.addLayout(opts)
        self.section(r,'04  /  VALIDATE & TEST')
        row=QHBoxLayout()
        v=QPushButton('✓ VALIDATE');v.setObjectName('good');v.clicked.connect(self.validate);row.addWidget(v)
        self.arm_btn=QPushButton('ARM');self.arm_btn.setObjectName('good');self.arm_btn.clicked.connect(self.arm);row.addWidget(self.arm_btn)
        r.addLayout(row)
        row=QHBoxLayout()
        self.rest_btn=QPushButton('REST TEST 60s');self.rest_btn.clicked.connect(self.start_rest_test);row.addWidget(self.rest_btn)
        reset=QPushButton('CLEAR ALL');reset.clicked.connect(self.clear_all);row.addWidget(reset)
        r.addLayout(row)
        self.metrics=QLabel('EOG classifier: —\nSSVEP PSDA: —\nGates: —')
        self.metrics.setWordWrap(True);self.metrics.setStyleSheet(f'color:{MUTED}')
        r.addWidget(self.metrics)
        self.section(r,'05  /  OUTPUT & STOP')
        self.keyboard_box=QCheckBox('Enable W/A/D keyboard → focused STIMULATOR (Windows only)')
        self.keyboard_box.toggled.connect(self.toggle_keyboard);r.addWidget(self.keyboard_box)
        warning=QLabel('Desktop simulator only. No physical wheelchair / actuator interface. '
                      'Keyboard mode uses Windows SendInput and must never target motor controls.')
        warning.setWordWrap(True);warning.setStyleSheet(f'color:{YELLOW}')
        r.addWidget(warning)
        stop=QPushButton('■ EMERGENCY STOP [SPACE]');stop.setObjectName('danger')
        stop.clicked.connect(self.emergency);r.addWidget(stop)
        reset_stop=QPushButton('RESET E-STOP (remains disarmed)')
        reset_stop.clicked.connect(self.reset_estop);r.addWidget(reset_stop)
        self.section(r,'06  /  DEMO SIGNAL INJECTION')
        if self.args.mode=='demo':
            row=QHBoxLayout()
            for label,hz in [('NEUTRAL',0),('9 Hz LEFT',9),('15 Hz RIGHT',15)]:
                b=QPushButton(label);b.clicked.connect(lambda checked=False,f=hz:self.select_target(f));row.addWidget(b)
            r.addLayout(row)
            row=QHBoxLayout()
            for label,kind in [('NATURAL','natural'),('INTENT','intentional'),('ARTIFACT','artifact')]:
                b=QPushButton(label);b.clicked.connect(lambda checked=False,k=kind:self.blink(k));row.addWidget(b)
            r.addLayout(row)
        else:
            r.addWidget(QLabel('Manual signal injection available in --mode demo only.'))
        self.section(r,'07  /  EVENTS & COMMANDS')
        self.log=QPlainTextEdit();self.log.setReadOnly(True)
        self.log.setMinimumHeight(175);self.log.setMaximumBlockCount(450)
        r.addWidget(self.log,2)
        splitter.setSizes([1040,460])

    def log_line(self,msg):
        self.log.appendPlainText(time.strftime('%H:%M:%S')+'  '+str(msg))

    def start_phase(self,kind):
        ok,msg=self.engine.start_calibration(kind)
        self.log_line(msg)
        if not ok:return
        if self.args.mode=='demo':
            self.source.set_target(0)
            self.source.set_training(kind if kind in ('natural','intentional') else None)
        self.output.update('stop')
        self._last_phase=kind

    def change_sensitivity(self,*_):
        self.engine.set_sensitivity(self.sens.currentData())
        self.output.update('stop')

    def validate(self):
        ok,msg=self.engine.validate()
        self.log_line(('✓ ' if ok else '✕ ')+msg)

    def arm(self):
        if self.engine.armed:
            self.engine.disarm();self.output.update('stop')
            self.log_line('MANUAL DISARM');return
        ok,msg=self.engine.arm()
        self.log_line(('✓ ' if ok else '✕ ')+msg)

    def start_rest_test(self):
        ok,msg=self.engine.start_rest_test()
        self.output.update('stop')
        self.log_line(msg)

    def clear_all(self):
        self.engine.clear_all()
        self.output.update('stop')
        if self.args.mode=='demo':self.source.set_training(None);self.source.set_target(0)
        self.log_line('ALL BASELINES RESET')

    def emergency(self):
        self.engine.stop('EMERGENCY STOP [SPACE]',emergency=True)
        self.output.update('stop')
        self.log_line('EMERGENCY STOP LATCHED; reset before re-arm')

    def reset_estop(self):
        self.engine.reset_estop()
        self.output.update('stop')
        self.log_line('ESTOP RESET; DISARMED')

    def select_target(self,hz):
        self.source.set_target(hz)
        self.log_line('DEMO TARGET: '+('NEUTRAL' if hz==0 else str(hz)+' Hz'))

    def blink(self,kind):
        self.source.inject_blink(kind)
        self.log_line('DEMO INJECT EOG: '+kind)

    def toggle_keyboard(self,value):
        if self.engine.armed:
            self.engine.disarm();self.log_line('DISARMED to change output mode')
        ok=self.output.set_enabled(value)
        if not ok:
            self.keyboard_box.blockSignals(True)
            self.keyboard_box.setChecked(False)
            self.keyboard_box.blockSignals(False)
            self.log_line('Keyboard unavailable: '+self.output.problem)
        else:self.log_line('DESKTOP STIMULATOR keyboard '+('ON' if value else 'OFF'))

    def _update_plots(self):
        engine=self.engine
        for key,buf in [('eog_raw',engine.raw_e),('eog_filt',engine.filt_e),
                        ('ss_raw',engine.raw_s),('ss_filt',engine.filt_s)]:
            arr=np.asarray(buf,float)
            if len(arr):
                arr=arr[-5*engine.fs:]
                self.curves[key].setData(np.arange(len(arr))/engine.fs,arr)
        now=time.monotonic()
        if now-self._last_psd>=.3 and len(engine.filt_s)>=engine.fs:
            self._last_psd=now
            arr=np.asarray(engine.filt_s,float)[-int(S.ANALYSIS_SEC*engine.fs):]
            f,p=engine.ss_proc.compute_psd(arr)
            if len(p):
                m=(f>=0)&(f<=40)
                self.curves['psd'].setData(f[m],p[m])

    def update_loop(self):
        try:
            self._tick()
        except Exception as exc:
            self.engine.stop('GUI PROCESSING ERROR',emergency=True)
            self.output.update('stop')
            self.log_line(f'CRITICAL ERROR: {type(exc).__name__}: {exc}')
            self.timer.stop()
            QMessageBox.critical(self,'Hybrid BCI stopped',str(exc))

    def _tick(self):
        if self.source.error and self.source.error!=self._source_error_seen:
            self._source_error_seen=self.source.error
            self.engine.stop('ACQUISITION ERROR',emergency=True)
            self.output.update('stop')
            self.log_line('ACQUISITION ERROR: '+self.source.error)
        received=self.source.pull()
        if received:
            eog,ss,dropped=received
            if self.source.fs!=self.engine.fs:
                self.engine.stop('SAMPLING RATE CHANGED',emergency=True)
                self.engine.set_fs(self.source.fs)
                self.log_line(f'Sampling rate updated to {self.source.fs} Hz; calibration cleared.')
            if dropped:
                self.engine.stop('ACQUISITION QUEUE OVERFLOW',emergency=True)
                self.output.update('stop')
                self.log_line('SIGNAL OVERFLOW · discard session and reset E-STOP')
            else:self.engine.process_block(eog,ss)
        self.engine.tick()
        if self._last_phase and self.engine.phase is None:
            self.log_line(f'{self._last_phase.upper()} baseline completed.')
            self.source.set_training(None)
            self._last_phase=None
        if not self.output.update(self.engine.motion):
            self.engine.stop('WINDOWS KEYBOARD RELEASE OR PRESS FAILURE',emergency=True)
            self.log_line('KEYBOARD FAILURE: '+self.output.problem)
            self.output.set_enabled(False)
            self.keyboard_box.blockSignals(True)
            self.keyboard_box.setChecked(False)
            self.keyboard_box.blockSignals(False)
        while self.engine.events:
            ts,k,detail=self.engine.events.popleft()
            self.log.appendPlainText(f'{ts} [{k}] {detail}')
        self.conn.setText(self.source.status+f'  |  {self.engine.total:,} samples  |  {self.engine.fs} Hz'
                          + ('  |  E-STOP LATCHED' if self.engine.estop else ''))
        self.conn.setStyleSheet(f'color:{RED if self.engine.estop or self.source.error else GREEN};padding:4px')
        motion=self.engine.motion
        self.command.setText(('■ STOP' if motion=='stop' else '▶ '+motion.upper())+
                             (' · ARMED' if self.engine.armed else ' · DISARMED'))
        self.command.setStyleSheet(f'background:{"#132821" if motion!="stop" else "#211620"};'
                                   f'color:{GREEN if motion!="stop" else RED};padding:10px;border-radius:8px')
        self.arm_btn.setText('DISARM' if self.engine.armed else 'ARM SIMULATOR')
        phases=self.engine.calibration
        self.calibration_label.setText(
            f'SSVEP: {"READY" if self.engine.ss_ready else "—"}  |  '
            f'NATURAL: {"READY" if phases["natural"] is not None else "—"}  |  '
            f'INTENT: {"READY" if phases["intentional"] is not None else "—"}\n'
            +('Recording '+self.engine.phase.upper() if self.engine.phase else 'READY TO VALIDATE' if self.engine.validated else 'CALIBRATION INCOMPLETE'))
        self.progress.setValue(int(self.engine.phase_progress()*100))
        self.natural_btn.setEnabled(self.engine.phase is None and self.engine.ss_ready)
        self.intent_btn.setEnabled(self.engine.phase is None and phases['natural'] is not None)
        self.ss_btn.setEnabled(self.engine.phase is None)
        self.sens.setEnabled(self.engine.phase is None)
        self.rest_btn.setEnabled(bool(self.engine.model and self.engine.model.ready) and not self.engine.rest_test and not self.engine.phase)
        z=self.engine.last_z
        self.score_label.setText(f'9 Hz: z={z.get(9,0):+.2f}    15 Hz: z={z.get(15,0):+.2f}    '
                                 f'PSDA: {self.engine.last_ssvep_reason}')
        e=self.engine.last_eog_features
        self.metrics.setText(f'EOG classifier: {self.engine.last_eog_class}'
                             +(f' · amp={e["amplitude"]:.0f}µV NCC={e["corr"]:.2f}' if e else '')
                             +f'\nEOG Gate: {self.engine.eog_gate.state} | SSVEP Gate: {self.engine.ss_gate.state}'
                             +f'\nSSVEP epochs: {self.engine.ss_epoch_count}, artifacts: {self.engine.ss_rejected_count}'
                             +f'\nIntentional model examples: {len(self.engine.intentional_events)}, natural: {len(self.engine.natural_events)}'
                             +(f'\nREST SCREEN: {self.engine.rest_intentional} apparent intentional / {self.engine.rest_events}' if self.engine.rest_test else ''))
        self.virtual.setText(f'VIRTUAL FORWARD DISTANCE: {self.engine.virtual.position:.2f} / {self.engine.virtual.limit:.2f} m'
                             +f' · {motion.upper()}')
        now=time.monotonic()
        if now-self._last_chart>=.10:
            self._last_chart=now
            self._update_plots()

    def closeEvent(self,event):
        self.timer.stop()
        self.engine.stop('WINDOW CLOSING',emergency=True)
        self.output.close()
        self.source.stop()
        self.engine.close()
        event.accept()

def main():
    parser=argparse.ArgumentParser(description='Original Logic Hybrid BCI · EOG W + 9Hz A + 15Hz D')
    parser.add_argument('--mode',choices=['demo','native','sender','dongle'],default='demo')
    parser.add_argument('--port',default='COM3')
    parser.add_argument('--fw2',action='store_true')
    parser.add_argument('--board-name',default='')
    parser.add_argument('--eog-ch',type=int,default=1)
    parser.add_argument('--ssvep-ch',type=int,default=3)
    parser.add_argument('--legacy-ssvep-row2',action='store_true',
                        help='Use EXACT original two-command SSVEP data row 2 instead of physical CH mapping')
    parser.add_argument('--keyboard',action='store_true',
                        help='Opt in to SendInput W/A/D to foreground Windows DESKTOP STIMULATOR ONLY')
    args=parser.parse_args()
    if args.eog_ch==args.ssvep_ch and not args.legacy_ssvep_row2:
        parser.error('EOG and SSVEP must have separate channel numbers')
    app=QApplication(sys.argv[:1]);app.setStyle('Fusion')
    win=Window(args);win.show()
    sys.exit(app.exec_())

if __name__=='__main__':main()
