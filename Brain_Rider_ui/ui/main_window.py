"""Presentation/controller bridge.

IMPORTANT: this file must NOT reimplement PSD, EOG events, PSDA, original gates,
or change the feature or calibration logic. It only calls HybridEngine.
"""
import time
from pathlib import Path
import numpy as np
from PyQt5.QtCore import Qt,QTimer
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QFrame,
                             QScrollArea,QStackedWidget,QShortcut,QMessageBox,QLabel)

import eog_original as E
import ssvep_original as S
from hybrid_engine import HybridEngine
from acquisition import Acquisition
from desktop_output import DesktopOutput
from ui.styles.theme import QSS,MINT,RED
from ui.components.common import label,button,StatusPill
from ui.pages.overview import OverviewPage
from ui.pages.calibration import CalibrationPage
from ui.pages.monitor import MonitorPage
from ui.pages.logs import LogsPage


class HybridMainWindow(QMainWindow):
    def __init__(self,args):
        super().__init__()
        self.args=args
        self.setWindowTitle('LightBrain  /  Hybrid BCI Workspace')
        self.resize(1530,940)
        self.setMinimumSize(990,690)
        self.setStyleSheet(QSS)
        self.engine=HybridEngine(log_dir=Path(__file__).resolve().parents[1]/'logs')
        self.output=DesktopOutput(enabled=False)
        self.source=None
        self._source_error_seen=None
        self._last_phase=None
        self._last_chart=0.0
        self._last_psd=0.0
        self.current_mode=args.mode or 'demo'
        self._construct_shell()
        self._log('Original processing modules and original EOG/SSVEP command gates loaded.')
        self._log('SSVEP epoch = 5 seconds; step = 5 seconds. EOG single blink = 1s hold + 2s cooldown.')
        self.timer=QTimer(self)
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(S.UPDATE_MS)
        QShortcut(QKeySequence(Qt.Key_Space),self,activated=self.emergency)
        if args.keyboard:
            self.monitor.keyboard.setChecked(True)
        if args.mode:
            self.begin_session(args.mode,args.port)

    def _construct_shell(self):
        root=QWidget()
        root.setObjectName('appRoot')
        self.setCentralWidget(root)
        frame=QHBoxLayout(root)
        frame.setContentsMargins(0,0,0,0)
        frame.setSpacing(0)
        side=QFrame();side.setObjectName('sideBar');side.setFixedWidth(232)
        side_l=QVBoxLayout(side)
        side_l.setContentsMargins(18,23,18,22);side_l.setSpacing(12)
        side_l.addWidget(label('◈  LIGHTBRAIN','brand'))
        side_l.addWidget(label('HYBRID BCI SYSTEM','smallBrand'))
        side_l.addSpacing(28)
        side_l.addWidget(label('WORKSPACE','eyebrow'))
        self.nav=[]
        items=[('◈  Overview',0),('◉  Calibration Studio',1),('≋  Live Monitor',2),('▤  Session Logs',3)]
        for title,i in items:
            b=button(title,lambda checked=False,j=i:self.navigate(j),'nav')
            b.setProperty('active','false')
            side_l.addWidget(b)
            self.nav.append(b)
        side_l.addStretch()
        divider=QFrame();divider.setFixedHeight(1);divider.setStyleSheet('background:#23394D;')
        side_l.addWidget(divider)
        side_l.addWidget(label('HARDWARE CONNECTION','eyebrow'))
        self.side_status=StatusPill('●  OFFLINE')
        side_l.addWidget(self.side_status)
        self.side_channels=label('Fp1 • CH1\nO1 • CH3\n200 Hz nominal','muted',True)
        side_l.addWidget(self.side_channels)
        side_l.addSpacing(8)
        self.sidebar_stop=button('■  EMERGENCY STOP',self.emergency,'danger')
        side_l.addWidget(self.sidebar_stop)
        side_l.addWidget(label('Space: emergency stop','muted'))
        frame.addWidget(side)

        right=QWidget();right.setObjectName('appRoot')
        right_l=QVBoxLayout(right)
        right_l.setContentsMargins(21,14,21,11);right_l.setSpacing(10)
        header=QHBoxLayout();header.setSpacing(12)
        header.addWidget(label('WORKSPACE  /  LIGHTBRAIN','eyebrow'))
        header.addStretch()
        self.connection=label('Waiting for signal source','muted')
        self.connection.setMaximumWidth(540)
        header.addWidget(self.connection)
        self.top_state=StatusPill('●  DISARMED')
        header.addWidget(self.top_state)
        right_l.addLayout(header)
        line=QFrame();line.setFixedHeight(1);line.setStyleSheet('background:#1D354A;')
        right_l.addWidget(line)
        self.stack=QStackedWidget()
        self.overview=OverviewPage(self.begin_session,lambda:self.navigate(1),lambda:self.navigate(2))
        self.calibration=CalibrationPage(self.start_phase,self.validate,self.arm,self.clear_all,self.change_sensitivity,self.start_rest_test)
        self.monitor=MonitorPage(self.emergency,self.reset_estop,self.arm,self.toggle_keyboard,self.select_target,self.blink)
        self.logs=LogsPage()
        for page in (self.overview,self.calibration,self.monitor,self.logs):
            scroll=QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            scroll.setFrameShape(QFrame.NoFrame)
            page.setMinimumWidth(880)
            scroll.setWidget(page)
            self.stack.addWidget(scroll)
        right_l.addWidget(self.stack,1)
        foot=QHBoxLayout()
        foot.addWidget(label('LIGHTBRAIN  /  RESEARCH SIMULATOR','eyebrow'))
        foot.addStretch()
        self.footer=label('STOP · NOT ARMED','muted')
        foot.addWidget(self.footer)
        right_l.addLayout(foot)
        frame.addWidget(right,1)
        self.navigate(0)

    def navigate(self,index):
        self.stack.setCurrentIndex(index)
        for i,nav in enumerate(self.nav):
            nav.setProperty('active','true' if i==index else 'false')
            nav.style().unpolish(nav);nav.style().polish(nav)

    def _log(self,message):
        self.logs.append_event(time.strftime('%H:%M:%S')+'  '+str(message))

    def begin_session(self,mode,port):
        # One acquisition connection supplies both EEG channels; no algorithm changes.
        if self.source:
            self.engine.stop('RECONNECT SOURCE')
            self.output.update('stop')
            self.source.stop()
        preserve_estop=self.engine.estop
        self.engine.clear_all()     # A different source requires fresh participant baselines.
        if preserve_estop:
            self.engine.stop('PREVIOUS EMERGENCY STOP STILL LATCHED',emergency=True)
        self.source=Acquisition(mode=mode,eog_ch=self.args.eog_ch,ssvep_ch=self.args.ssvep_ch,
                                port=port,fw2=self.args.fw2,board_name=self.args.board_name,
                                legacy_row2=self.args.legacy_ssvep_row2)
        self._last_phase=None
        self._source_error_seen=None
        self.current_mode=mode
        self.monitor.set_demo_mode(mode=='demo')
        self.overview.source.setCurrentIndex(max(0,self.overview.source.findData(mode)))
        self.overview.port.setText(port)
        self.source.start()
        self._log(f'SOURCE START: {mode.upper()}  |  EOG CH{self.args.eog_ch}, SSVEP CH{self.args.ssvep_ch}')
        self.navigate(1)

    def start_phase(self,kind):
        ok,msg=self.engine.start_calibration(kind)
        self._log(('✓ ' if ok else '✕ ')+msg)
        self.calibration.feedback.setText(msg)
        if not ok:return
        if self.source and self.current_mode=='demo':
            self.source.set_target(0)
            self.source.set_training(kind if kind in ('natural','intentional') else None)
        self.output.update('stop')
        self._last_phase=kind

    def validate(self):
        ok,msg=self.engine.validate()
        self._log(('✓ ' if ok else '✕ ')+msg)
        self.calibration.feedback.setText(msg)

    def arm(self):
        if self.engine.armed:
            self.engine.disarm()
            self.output.update('stop')
            self._log('MANUAL DISARM')
        else:
            ok,msg=self.engine.arm()
            self._log(('✓ ' if ok else '✕ ')+msg)
            self.calibration.feedback.setText(msg)
            if ok:self.navigate(2)

    def start_rest_test(self):
        ok,msg=self.engine.start_rest_test()
        self.output.update('stop')
        self._log(('✓ ' if ok else '✕ ')+msg)
        self.calibration.feedback.setText(msg)

    def clear_all(self):
        self.engine.clear_all()
        self.output.update('stop')
        if self.source and self.current_mode=='demo':
            self.source.set_training(None)
            self.source.set_target(0)
        self._last_phase=None
        self._log('BASELINES CLEARED')
        self.calibration.feedback.setText('Baselines cleared. Record all three stages again.')

    def change_sensitivity(self,value):
        self.engine.set_sensitivity(value)
        self.output.update('stop')
        self._log(f'EOG sensitivity changed to {value}')

    def emergency(self):
        self.engine.stop('EMERGENCY STOP [SPACE]',emergency=True)
        self.output.update('stop')
        self._log('EMERGENCY STOP LATCHED — output blocked until reset')

    def reset_estop(self):
        self.engine.reset_estop()
        self.output.update('stop')
        self._log('E-STOP RESET — DISARMED')

    def toggle_keyboard(self,checked):
        if self.engine.armed:
            self.engine.disarm()
            self._log('DISARMED TO CHANGE KEYBOARD OUTPUT')
        ok=self.output.set_enabled(checked)
        if not ok:
            self.monitor.keyboard.blockSignals(True)
            self.monitor.keyboard.setChecked(False)
            self.monitor.keyboard.blockSignals(False)
            self._log('KEYBOARD UNAVAILABLE: '+self.output.problem)
        else:
            self._log('DESKTOP STIMULATOR KEYBOARD '+('ON' if checked else 'OFF'))

    def select_target(self,hz):
        if self.source and self.current_mode=='demo':
            self.source.set_target(hz)
            self._log('DEMO SSVEP TARGET: '+str(hz)+' Hz')

    def blink(self,kind):
        if self.source and self.current_mode=='demo':
            self.source.inject_blink(kind)
            self._log('DEMO EOG INJECTION: '+kind)

    def update_loop(self):
        try:
            self._tick()
        except Exception as exc:
            self.engine.stop('GUI PROCESSING ERROR',emergency=True)
            self.output.update('stop')
            self._log(f'CRITICAL: {type(exc).__name__}: {exc}')
            self.timer.stop()
            QMessageBox.critical(self,'LightBrain safe stop',str(exc))

    def _tick(self):
        if self.source:
            if self.source.error and self.source.error!=self._source_error_seen:
                self._source_error_seen=self.source.error
                self.engine.stop('ACQUISITION ERROR',emergency=True)
                self.output.update('stop')
                self._log('ACQUISITION ERROR: '+self.source.error)
            received=self.source.pull()
            if received:
                eog,ss,dropped=received
                if self.source.fs!=self.engine.fs:
                    self.engine.stop('SAMPLING RATE CHANGED',emergency=True)
                    self.engine.set_fs(self.source.fs)
                    self._log(f'SAMPLE RATE UPDATED: {self.source.fs} Hz; all baselines reset.')
                if dropped:
                    self.engine.stop('ACQUISITION QUEUE OVERFLOW',emergency=True)
                    self.output.update('stop')
                    self._log('ACQUISITION QUEUE OVERFLOW: E-STOP LATCHED')
                else:
                    self.engine.process_block(eog,ss)
        self.engine.tick()
        if self._last_phase and self.engine.phase is None:
            self._log(f'{self._last_phase.upper()} BASELINE COMPLETED')
            if self.source:self.source.set_training(None)
            self._last_phase=None
        if not self.output.update(self.engine.motion):
            self.engine.stop('WINDOWS KEYBOARD FAILURE',emergency=True)
            self._log('DESKTOP OUTPUT FAILURE: '+self.output.problem)
            self.output.set_enabled(False)
            self.monitor.keyboard.blockSignals(True)
            self.monitor.keyboard.setChecked(False)
            self.monitor.keyboard.blockSignals(False)
        while self.engine.events:
            ts,kind,detail=self.engine.events.popleft()
            self.logs.append_event(f'{ts}  [{kind}]  {detail}')
        self._render_status()
        now=time.monotonic()
        index=self.stack.currentIndex()
        if index in (1,2) and now-self._last_chart>=.10:
            self._last_chart=now
            self._draw_plots(index,now)

    def _render_status(self):
        e=self.engine
        active=bool(self.source and self.source.connected and not self.source.error)
        source_status=self.source.status if self.source else 'No source selected'
        self.connection.setText(source_status[:82])
        self.side_status.set_status('●  CONNECTED' if active else '●  OFFLINE / ERROR',active)
        self.top_state.set_status('●  ARMED' if e.armed else '●  E-STOP' if e.estop else '●  DISARMED',e.armed)
        self.side_channels.setText(f'Fp1 • EEG CH{self.args.eog_ch}\nO1 • EEG CH{self.args.ssvep_ch}\n{e.fs} Hz  |  '+self.current_mode.upper())
        self.footer.setText(f'{e.total:,} samples  •  {e.fs} Hz  •  {e.motion.upper()}  •  '+('E-STOP' if e.estop else 'ARMED' if e.armed else 'DISARMED'))
        calibrated=int(e.ss_ready)+int(e.calibration['natural'] is not None)+int(e.calibration['intentional'] is not None)
        self.overview.set_connected(source_status,active,calibrated,e.armed,e.fs)
        self.calibration.render(e,active)
        self.monitor.render(e,active)

    def _draw_plots(self,index,now):
        e=self.engine
        destinations={}
        if index==1:
            destinations={'eog_filt':self.calibration.eog,'ss_filt':self.calibration.eeg}
        elif index==2:
            destinations={k:v for k,v in self.monitor.charts.items() if k!='psd'}
        buffers={'eog_raw':e.raw_e,'eog_filt':e.filt_e,'ss_raw':e.raw_s,'ss_filt':e.filt_s}
        for key,chart in destinations.items():
            a=np.asarray(buffers[key],float)[-5*e.fs:]
            if len(a):chart.update_series(a,fs=e.fs)
        if index==2 and now-self._last_psd>=.30 and len(e.filt_s)>=e.fs:
            self._last_psd=now
            data=np.asarray(e.filt_s,float)[-int(S.ANALYSIS_SEC*e.fs):]
            f,p=e.ss_proc.compute_psd(data)  # Original SignalProcessor method
            if len(p):
                mask=(f>=0)&(f<=40)
                self.monitor.charts['psd'].update_series(p[mask],x_values=f[mask])

    def closeEvent(self,event):
        self.timer.stop()
        self.engine.stop('APPLICATION CLOSING',emergency=True)
        self.output.close()
        if self.source:self.source.stop()
        self.engine.close()
        event.accept()
