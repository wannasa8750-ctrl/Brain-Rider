"""LightBrain EOG — standalone GUI proof-of-concept (NOT for physical wheelchair).
Based on the panel/plot/state-machine layout of stimulator_control_v.1.py.py.
Default connection uses the sender-compatible native BLE profile; keyboard output is disabled unless explicitly enabled.
"""
import argparse
import os
import ctypes
from ctypes import wintypes
import sys
import time
import threading
import traceback
from collections import deque

import numpy as np
from scipy.signal import butter, sosfilt, find_peaks
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QKeySequence
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton,
                             QHBoxLayout, QVBoxLayout, QGridLayout, QFrame, QProgressBar,
                             QComboBox, QShortcut, QPlainTextEdit, QCheckBox, QDoubleSpinBox)
import pyqtgraph as pg

try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    BRAINFLOW_AVAILABLE = True
except ImportError:
    BRAINFLOW_AVAILABLE = False

FS_DEMO = 200
BASELINE_SEC = 10
TRIAL_SEC = 2.4
W_HOLD_SEC = 1.0  # Keep the W key DOWN for one full second per blink.
WARMUP_SEC = 2.0
N_NATURAL = 10
N_INTENTIONAL = 15
CLUSTER_GAP = 0.65
REFRACTORY_SEC = 1.8
NO_DATA_SEC = 5.0
MAX_EVENT_SEC = 1.3
ACCENT = '#00d4ff'
BG = '#0d1117'
PANEL = '#161b22'
GREEN = '#2dde98'
RED = '#f85149'
DIM = '#8b949e'


class Acquisition(QThread):
    samples = pyqtSignal(object)
    status = pyqtSignal(str)
    problem = pyqtSignal(str)
    def __init__(self, mode='sender', port='COM3', fw2=False, board_name=''):
        super().__init__()
        self.mode, self.port, self.fw2, self.board_name = mode, port, fw2, board_name
        self._running = True
        self._demo_lock = threading.Lock()
        self._demo_blink = None
        self._demo_t = 0.0
        self._demo_pulses = []
        self.fs = FS_DEMO
    def demo_blink(self, kind):
        with self._demo_lock:
            self._demo_blink = kind
    def shutdown(self):
        self._running = False
    def run(self):
        board = None
        streaming = False
        try:
            if self.mode == 'demo':
                self.status.emit('DEMO · Synthetic EOG · 200 Hz')
                rng = np.random.default_rng(1234)
                while self._running:
                    n = 10
                    t = self._demo_t + np.arange(n)/self.fs
                    with self._demo_lock:
                        cmd = self._demo_blink
                        self._demo_blink = None
                    if cmd:
                        self._demo_pulses.append((self._demo_t + .25, cmd))
                    x = rng.normal(0, 8, n) + 11*np.sin(2*np.pi*.2*t)
                    keep = []
                    for start, kind in self._demo_pulses:
                        d = t-start
                        if kind == 'intentional':
                            amp, width = 460, .16
                        else:
                            amp, width = 230, .085
                        x += amp*np.exp(-.5*(d/width)**2) - .30*amp*np.exp(-.5*((d-.28)/(.10 if kind=='intentional' else .07))**2)
                        if t[-1] < start+1.0:
                            keep.append((start, kind))
                    self._demo_pulses = keep
                    self.samples.emit(x.astype(float))
                    self._demo_t += n/self.fs
                    self.msleep(50)
                return
            if not BRAINFLOW_AVAILABLE:
                raise RuntimeError('BrainFlow is not installed: pip install brainflow')
            # 'sender' reproduces the user's known working ganglion_sender.py
            # (GANGLION_NATIVE_BOARD + serial_port).  That board ID uses native BLE:
            # serial_port is present for sender compatibility but is ignored by native BLE.
            # 'dongle' is the *actual* legacy BLED112 serial COM profile.
            if self.mode in ('native', 'sender'):
                board_id = BoardIds.GANGLION_NATIVE_BOARD
            else:
                board_id = BoardIds.GANGLION_BOARD
            params = BrainFlowInputParams()
            if self.mode in ('sender', 'dongle'):
                params.serial_port = self.port
            if self.mode in ('native', 'sender'):
                if self.fw2:
                    params.other_info = 'fw:2'
                elif self.board_name:
                    # Kept solely for compatibility with the sender's other_info.
                    params.other_info = self.board_name
            link_name = ('Sender-compatible native BLE' if self.mode == 'sender'
                         else 'Native BLE' if self.mode == 'native'
                         else 'BLED112 dongle (' + self.port + ')')
            self.status.emit(f'Connecting · {link_name}...')
            board = BoardShim(board_id, params)
            board.prepare_session()
            self.status.emit(f'Prepared · {link_name}; starting stream...')
            board.start_stream()
            streaming = True
            self.fs = int(BoardShim.get_sampling_rate(board_id))
            channel = int(BoardShim.get_eeg_channels(board_id)[0])  # CH1, already in µV
            self.status.emit(f'CONNECTED · {link_name} · CH1 · {self.fs} Hz')
            while self._running:
                block = board.get_board_data()
                if block.shape[1]:
                    values = np.asarray(block[channel], dtype=float)
                    if not np.isfinite(values).all():
                        raise ValueError('Non-finite samples from board')
                    self.samples.emit(values)  # DO NOT apply a second scale factor
                else:
                    self.msleep(10)
        except Exception as exc:
            self.problem.emit(f'{type(exc).__name__}: {exc}')
        finally:
            if board is not None:
                if streaming:
                    try: board.stop_stream()
                    except Exception: pass
                try:
                    if board.is_prepared(): board.release_session()
                except Exception: pass
            self.status.emit('DISCONNECTED')


class StreamingFilter:
    def __init__(self, fs):
        self.fs = fs
        self.sos = butter(4, [0.1, 10], btype='bandpass', fs=fs, output='sos')
        self.zi = None
    def apply(self, x):
        x = np.asarray(x, dtype=float)
        if not len(x): return x
        if self.zi is None:
            from scipy.signal import sosfilt_zi
            self.zi = sosfilt_zi(self.sos)*float(x[0])
        y, self.zi = sosfilt(self.sos, x, zi=self.zi)
        return y


def noise_stats(x):
    x = np.asarray(x)
    mid = float(np.median(x))
    sigma = float(1.4826*np.median(np.abs(x-mid)))
    return mid, sigma


def candidates(x, fs, center, sigma, threshold_factor=0.9):
    """Detect threshold excursions; merge biphasic fragments into blink groups.

    Both calibration and LIVE use exactly this function on a 2.4 s window.
    Feature duration refers to one selected excursion; labels are experimental.
    """
    x = np.asarray(x, dtype=float)-center
    if len(x) < int(.3*fs) or not np.isfinite(x).all() or sigma <= 0:
        return []
    threshold = max(1.0, threshold_factor*sigma)
    # Max absolute deflection; minimum spacing prevents adjacent local maxima
    peaks, props = find_peaks(np.abs(x), height=threshold, distance=max(1,int(.075*fs)),
                             prominence=max(0.8, 0.65*sigma))
    if not len(peaks): return []
    groups = [[int(peaks[0])]]
    for p in peaks[1:]:
        if (int(p)-groups[-1][-1])/fs <= CLUSTER_GAP:
            groups[-1].append(int(p))
        else:
            groups.append([int(p)])
    output = []
    for g in groups:
        pk = max(g, key=lambda j:abs(x[j]))
        start, end = pk, pk
        end_thr = .45*threshold
        while start > 0 and abs(x[start-1]) > end_thr: start -= 1
        while end < len(x)-1 and abs(x[end+1]) > end_thr: end += 1
        # Event may lie at the edge of live window: reject until enough post-event samples
        if end >= len(x)-int(.07*fs): continue
        duration = (end-start+1)/fs
        if not .035 <= duration <= MAX_EVENT_SEC: continue
        segment = x[start:end+1]
        v = float(np.max(np.abs(np.diff(segment)))*fs) if len(segment)>1 else 0
        output.append(dict(amplitude=float(np.max(np.abs(segment))),
                           duration=float(duration), velocity=v,
                           time=float(pk/fs), fragments=len(g)))
    return output


def feature(e):
    return np.array([e['amplitude'], e['duration'], e['velocity']], dtype=float)


class Classifier:
    def __init__(self, natural, intentional):
        if len(natural)<N_NATURAL or len(intentional)<N_INTENTIONAL:
            raise ValueError('Not enough valid calibration trials')
        n = np.array([feature(e) for e in natural]); i = np.array([feature(e) for e in intentional])
        allx = np.vstack((n, i))
        med = np.median(allx, axis=0)
        scale = np.maximum(1.4826*np.median(np.abs(allx-med), axis=0),
                           np.maximum(.05*np.ptp(allx,axis=0), 1e-6))
        self.n, self.i, self.scale = n/scale, i/scale, scale
        # Leave-one-out: reject if any natural example is mislabeled as intentional
        ok = 0; false_int = 0
        for idx, x in enumerate(self.n):
            dn = np.min(np.linalg.norm(np.delete(self.n,idx,axis=0)-x,axis=1))
            di = np.min(np.linalg.norm(self.i-x,axis=1))
            ok += bool(dn < di)
            false_int += bool(di < dn)
        for idx, x in enumerate(self.i):
            di = np.min(np.linalg.norm(np.delete(self.i,idx,axis=0)-x,axis=1))
            dn = np.min(np.linalg.norm(self.n-x,axis=1))
            ok += bool(di < dn)
        self.accuracy = ok/(len(n)+len(i)); self.false_int = false_int
        nearest = []
        for idx, x in enumerate(self.i):
            nearest.append(float(np.min(np.linalg.norm(np.delete(self.i,idx,axis=0)-x,axis=1))))
        self.max_distance = 1.25*max(nearest)+1e-8
        if self.accuracy < .9 or false_int:
            raise ValueError(f'Overlapping classes: LOO={self.accuracy:.0%}, natural→intentional={false_int}')
    def predict(self, e):
        x = feature(e)/self.scale
        dn = float(np.min(np.linalg.norm(self.n-x,axis=1)))
        di = float(np.min(np.linalg.norm(self.i-x,axis=1)))
        return ('INTENTIONAL' if di <= self.max_distance and di*1.25 < dn else 'NATURAL / UNKNOWN',dn,di)


class Keyboard:
    """Windows SendInput scan-code W, for desktop simulation only.

    Never connect this quick-threshold prototype to physical actuators.
    """
    SCAN_W = 0x11
    KEYEVENTF_SCANCODE = 0x0008
    KEYEVENTF_KEYUP = 0x0002

    def __init__(self):
        self.enabled = False
        self.down_until = None
        self.last_error = ''
        self._available = os.name == 'nt'
        if self._available:
            ULONG_PTR = getattr(wintypes, 'ULONG_PTR', ctypes.c_size_t)

            class KEYBDINPUT(ctypes.Structure):
                _fields_ = [('wVk', wintypes.WORD), ('wScan', wintypes.WORD),
                            ('dwFlags', wintypes.DWORD), ('time', wintypes.DWORD),
                            ('dwExtraInfo', ULONG_PTR)]

            class MOUSEINPUT(ctypes.Structure):
                _fields_ = [('dx', wintypes.LONG), ('dy', wintypes.LONG),
                            ('mouseData', wintypes.DWORD), ('dwFlags', wintypes.DWORD),
                            ('time', wintypes.DWORD), ('dwExtraInfo', ULONG_PTR)]

            class HARDWAREINPUT(ctypes.Structure):
                _fields_ = [('uMsg', wintypes.DWORD), ('wParamL', wintypes.WORD),
                            ('wParamH', wintypes.WORD)]

            class INPUT_UNION(ctypes.Union):
                _fields_ = [('ki', KEYBDINPUT), ('mi', MOUSEINPUT), ('hi', HARDWAREINPUT)]

            class INPUT(ctypes.Structure):
                _anonymous_ = ('u',)
                _fields_ = [('type', wintypes.DWORD), ('u', INPUT_UNION)]

            self._KEYBDINPUT = KEYBDINPUT
            self._INPUT = INPUT
            self._sendinput = ctypes.WinDLL('user32', use_last_error=True).SendInput
            self._sendinput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
            self._sendinput.restype = wintypes.UINT

    @property
    def controller(self):
        """Compatibility with existing status label: only truthy if enabled."""
        return self if self.enabled and self._available else None

    def set_enabled(self, enabled):
        self.release()
        if enabled and not self._available:
            raise RuntimeError('Windows SendInput is only available on Windows')
        self.enabled = bool(enabled)

    def _send(self, key_up=False):
        flags = self.KEYEVENTF_SCANCODE | (self.KEYEVENTF_KEYUP if key_up else 0)
        inp = self._INPUT(type=1, ki=self._KEYBDINPUT(wVk=0, wScan=self.SCAN_W,
                                                      dwFlags=flags, time=0, dwExtraInfo=0))
        ctypes.set_last_error(0)
        n = self._sendinput(1, ctypes.byref(inp), ctypes.sizeof(inp))
        if n != 1:
            self.last_error = f'SendInput failed, Win32 error={ctypes.get_last_error()} (check app elevation/focus)'
            return False
        self.last_error = ''
        return True

    def press(self):
        if self.down_until is not None or not self.enabled:
            return False
        if not self._send(False):
            return False
        self.down_until = time.monotonic() + W_HOLD_SEC
        return True

    def tick(self):
        if self.down_until is not None and time.monotonic() >= self.down_until:
            self.release()

    def release(self):
        if self.down_until is not None and self._available:
            self._send(True)
        self.down_until = None


class MainWindow(QMainWindow):
    def __init__(self, mode, port, fw2, keyboard, board_name=''):
        super().__init__()
        self.mode, self.port, self.fw2 = mode,port,fw2
        self.fs = FS_DEMO
        self.filter = StreamingFilter(self.fs)
        self.raw = deque(maxlen=self.fs*15)
        self.filt = deque(maxlen=self.fs*15)
        self.total = 0
        self.base_samples=[]
        self.base_started=0
        self.state = 'IDLE'
        self.center = self.sigma = None
        self.natural=[]; self.intentional=[]
        self.clf=None
        self.armed=False
        self.trial_kind=None; self.trial_samples=[]; self.trial_started=0
        self.live_next_index=0
        self.last_detection=-1e9
        self.last_data=time.monotonic()
        self.connected=False
        self.warmup=0
        self.keyboard=Keyboard()
        self.build_ui()
        self.chk_keyboard.setChecked(keyboard)
        self.toggle_keyboard(keyboard)
        self.source=Acquisition(mode,port,fw2,board_name)
        self.source.samples.connect(self.on_samples)
        self.source.status.connect(self.set_connection)
        self.source.problem.connect(self.on_problem)
        self.source.start()
        self.timer=QTimer(self); self.timer.timeout.connect(self.refresh); self.timer.start(50)
        QShortcut(QKeySequence(Qt.Key_Space),self,activated=self.stop_output)
    def build_ui(self):
        self.setWindowTitle('LightBrain — EOG Blink to W | Windows SendInput')
        self.resize(1320,790)
        self.setStyleSheet(f'''QWidget {{background:{BG};color:#e6edf3;font-family:Consolas;}}
        QPushButton {{background:#1d2935;border:1px solid #0e7490;border-radius:5px;padding:9px;}}
        QPushButton:hover {{background:#23465a;}}
        QPushButton:disabled {{color:#5d6874;border-color:#303943;}}
        QFrame#panel {{background:{PANEL};border:1px solid #0e7490;border-radius:8px;}}
        QProgressBar {{background:#1c2333;border:1px solid #0e7490;border-radius:4px;text-align:center;}}
        QProgressBar::chunk {{background:{ACCENT};}}''')
        root=QVBoxLayout(); central=QWidget();central.setLayout(root);self.setCentralWidget(central)
        root.setContentsMargins(12,10,12,10)
        head=QHBoxLayout();root.addLayout(head)
        title=QLabel('◉  LightBrain  /  EOG INTENTIONAL BLINK');title.setFont(QFont('Consolas',15,QFont.Bold));title.setStyleSheet(f'color:{ACCENT}')
        head.addWidget(title);head.addStretch();self.conn=QLabel('● Starting...');head.addWidget(self.conn)
        main=QHBoxLayout();root.addLayout(main,1)
        graph=QGridLayout();main.addLayout(graph,7)
        pg.setConfigOptions(antialias=False)
        self.plots={};self.curves={}
        for k,r,c,plot_title,color in [
            ('raw',0,0,'RAW EOG · Channel 1','#8799fc'),
            ('filtered',0,1,'FILTERED EOG · 0.1–10 Hz',GREEN),
            ('event',1,0,'LATEST BLINK / TRIAL WINDOW','#ffc857'),
            ('history',1,1,'DETECTION FEATURE · AMPLITUDE',ACCENT)]:
            p=pg.PlotWidget(title=plot_title,background='#0a0e14')
            p.setLabel('bottom','Time','s')
            p.setLabel('left','Amplitude','µV')
            p.showGrid(x=True,y=True,alpha=.2)
            self.curves[k]=p.plot(pen=pg.mkPen(color,width=1.5))
            self.plots[k]=p;graph.addWidget(p,r,c)
        self.history_x=[];self.history_y=[]
        panel=QFrame();panel.setObjectName('panel');main.addWidget(panel,3)
        right=QVBoxLayout(panel);right.setContentsMargins(12,10,12,10)
        def section(s):
            l=QLabel(s);l.setStyleSheet(f'color:{ACCENT};font-weight:bold;border-bottom:1px solid #175a70;padding:3px;');right.addWidget(l)
        section('01 / BASELINE ACQUISITION')
        self.baseline_btn=QPushButton('▶  Record Baseline (10 s)');self.baseline_btn.clicked.connect(self.begin_baseline);right.addWidget(self.baseline_btn)
        self.clear_btn=QPushButton('✕ Clear / Reset Calibration');self.clear_btn.clicked.connect(self.clear_calibration);right.addWidget(self.clear_btn)
        self.base_bar=QProgressBar();self.base_bar.setValue(0);right.addWidget(self.base_bar)
        self.base_text=QLabel('Waiting for baseline');right.addWidget(self.base_text)
        section('02 / BLINK CALIBRATION')
        self.natural_btn=QPushButton('Record NATURAL blink');self.natural_btn.clicked.connect(lambda:self.begin_trial('natural'));right.addWidget(self.natural_btn)
        self.intentional_btn=QPushButton('Record INTENTIONAL blink');self.intentional_btn.clicked.connect(lambda:self.begin_trial('intentional'));right.addWidget(self.intentional_btn)
        self.trial_bar=QProgressBar();self.trial_bar.setValue(0);right.addWidget(self.trial_bar)
        self.trial_text=QLabel('Natural 0/10   |   Intentional 0/15');right.addWidget(self.trial_text)
        self.arm_btn=QPushButton('▶ ARM BLINK → W (low threshold)');self.arm_btn.clicked.connect(self.validate);right.addWidget(self.arm_btn)
        factor_row=QHBoxLayout(); factor_row.addWidget(QLabel('Threshold factor (× baseline MAD):'))
        self.factor=QDoubleSpinBox();self.factor.setRange(0.5,8.0);self.factor.setSingleStep(0.2);self.factor.setDecimals(1);self.factor.setValue(0.9)
        factor_row.addWidget(self.factor);right.addLayout(factor_row)
        tip=QLabel('LOW sensitivity threshold / experimental: natural blinks may also trigger W')
        tip.setWordWrap(True);tip.setStyleSheet('color:#ffc857');right.addWidget(tip)
        section('03 / REAL-TIME SIGNAL METRICS')
        self.metrics=QLabel('Sampling: —\nThreshold: —\nAmplitude: —\nDuration: —\nVelocity: —')
        self.metrics.setMinimumHeight(90);right.addWidget(self.metrics)
        section('04 / DETECTION & KEYBOARD')
        self.result=QLabel('NO SIGNAL / DISARMED');self.result.setFont(QFont('Consolas',14,QFont.Bold));self.result.setWordWrap(True);self.result.setStyleSheet(f'color:{RED}');right.addWidget(self.result)
        self.chk_keyboard=QCheckBox('Enable desktop W key (1 s)');self.chk_keyboard.toggled.connect(self.toggle_keyboard);right.addWidget(self.chk_keyboard)
        self.key_label=QLabel('W: RELEASED');right.addWidget(self.key_label)
        self.test_btn=QPushButton('TEST W · 3s to focus simulator');self.test_btn.clicked.connect(self.test_w);right.addWidget(self.test_btn)
        stop=QPushButton('■  EMERGENCY STOP   [SPACE]');stop.setStyleSheet(f'background:#5a2026;border:1px solid {RED};padding:12px;font-weight:bold');stop.clicked.connect(self.stop_output);right.addWidget(stop)
        self.log=QPlainTextEdit();self.log.setReadOnly(True);self.log.setMaximumBlockCount(300);self.log.setMinimumHeight(130);right.addWidget(self.log,1)
        self.log_msg('Ready. Default mode: DISARMED. No physical motor control.')
        self.log_msg('Quick mode: Baseline -> ARM; no blink training required. Low threshold may false-trigger.')
        self.log_msg(f'Connection profile: {self.mode} | COM port: {self.port}')
    def log_msg(self,s):
        self.log.appendPlainText(time.strftime('%H:%M:%S  ')+s)
    def set_connection(self,s):
        self.connected=s.startswith('CONNECTED') or s.startswith('DEMO')
        self.conn.setText(('● ' if self.connected else '○ ')+s)
        self.conn.setStyleSheet(f'color:{GREEN if self.connected else RED}')
        if s=='DISCONNECTED':
            self.disarm('Disconnected')
    def on_problem(self,s):
        self.log_msg('ACQUISITION ERROR: '+s)
        self.disarm('Connection error')
    def toggle_keyboard(self,enabled):
        try:
            self.keyboard.set_enabled(enabled)
            self.log_msg('Windows SendInput keyboard '+('ENABLED' if enabled else 'disabled'))
        except Exception as e:
            self.chk_keyboard.blockSignals(True);self.chk_keyboard.setChecked(False);self.chk_keyboard.blockSignals(False)
            self.log_msg(f'Keyboard unavailable: {e}')
    def test_w(self):
        if not self.keyboard.enabled:
            self.log_msg('TEST W: enable desktop W checkbox first')
            return
        self.test_btn.setEnabled(False)
        self.log_msg('TEST W: switch focus to simulator now; sending W in 3 seconds')
        QTimer.singleShot(3000, self._perform_test_w)

    def _perform_test_w(self):
        self.test_btn.setEnabled(True)
        if not self.keyboard.enabled:
            return
        if self.keyboard.press():
            self.log_msg('TEST W: SendInput scan-code W DOWN; W UP after 1.0s')
            self.result.setText('TEST W DOWN (1s)')
            self.result.setStyleSheet(f'color:{GREEN}')
        else:
            self.log_msg('TEST W FAILED: '+(self.keyboard.last_error or 'W already pressed'))

    def stop_output(self):
        self.disarm('Emergency STOP (Space)')
    def disarm(self,reason):
        self.keyboard.release();self.clf=None;self.armed=False
        self.state='IDLE' if self.center is None else 'CALIBRATING'
        self.result.setText('STOP / DISARMED');self.result.setStyleSheet(f'color:{RED}')
        self.log_msg('STOP · '+reason)
    def clear_calibration(self):
        self.disarm('Calibration reset')
        self.center=self.sigma=None
        self.base_samples=[];self.natural=[];self.intentional=[]
        self.trial_kind=None;self.trial_samples=[]
        self.state='IDLE';self.base_bar.setValue(0);self.trial_bar.setValue(0)
        self.base_text.setText('Waiting for baseline')
        self.trial_text.setText('Natural 0/10   |   Intentional 0/15')
    def begin_baseline(self):
        if not self.connected or self.warmup < int(WARMUP_SEC*self.fs):
            self.log_msg('Wait for connection and 2 s filter warm-up.');return
        self.disarm('Re-recording baseline')
        self.natural=[];self.intentional=[];self.trial_kind=None
        self.center=self.sigma=None;self.base_samples=[]
        self.state='BASELINE'
        self.base_started=time.monotonic()
        self.base_bar.setValue(0)
        self.log_msg('BASELINE: look straight, stay still, avoid blinking (10 s)')
    def begin_trial(self,kind):
        if self.state not in ('CALIBRATING','READY') or self.center is None:
            self.log_msg('Record baseline first.');return
        if self.trial_kind:
            self.log_msg('A trial is already running.');return
        if len(self.natural if kind=='natural' else self.intentional) >= (N_NATURAL if kind=='natural' else N_INTENTIONAL):
            self.log_msg('This class is complete. Clear to restart.');return
        self.disarm('Calibration trial')
        self.trial_kind=kind;self.trial_samples=[];self.trial_started=time.monotonic()
        self.trial_bar.setValue(0)
        self.log_msg(f'BLINK ONCE NOW: {kind.upper()} (trial window {TRIAL_SEC:.1f}s)')
        if self.mode=='demo': self.source.demo_blink(kind)
    def finish_trial(self):
        kind=self.trial_kind;self.trial_kind=None
        x=np.asarray(self.trial_samples,dtype=float)
        ev=candidates(x,self.fs,self.center,self.sigma,self.factor.value())
        self.curves['event'].setData(np.arange(len(x))/self.fs,x)
        if len(ev)==1:
            (self.natural if kind=='natural' else self.intentional).append(ev[0])
            self.log_msg(f'{kind.upper()} OK: fragments={ev[0]["fragments"]} | amp={ev[0]["amplitude"]:.1f}µV | duration={ev[0]["duration"]:.3f}s')
        else:
            self.log_msg(f'{kind.upper()} REJECT: found {len(ev)} groups. Repeat trial (session stays open).')
        self.trial_text.setText(f'Natural {len(self.natural)}/{N_NATURAL}   |   Intentional {len(self.intentional)}/{N_INTENTIONAL}')
        self.trial_bar.setValue(100)
        self.state='CALIBRATING'
    def validate(self):
        """Quick ARM: any threshold excursion may trigger desktop W; no calibrated classifier."""
        self.keyboard.release()
        if not self.connected or self.center is None or self.sigma is None or self.state=='BASELINE' or self.trial_kind:
            self.log_msg('Cannot ARM: connect, complete baseline, and finish any active trial.');return
        self.clf=None
        self.armed=True
        self.state='READY'
        self.live_next_index=self.total
        self._last_peak_sample=self.total
        self.last_detection=time.monotonic()  # suppress stale pulses after arming
        self.result.setText('ARMED · Blink -> desktop W (1 s)')
        self.result.setStyleSheet(f'color:{GREEN}')
        self.log_msg(f'QUICK ARMED · factor={self.factor.value():.1f} · threshold={max(1.0,self.factor.value()*self.sigma):.2f} µV · any blink may trigger, including natural blink')

    def on_samples(self,x):
        x=np.asarray(x,dtype=float)
        if not np.isfinite(x).all():
            self.disarm('Invalid samples');return
        y=self.filter.apply(x)
        self.raw.extend(x);self.filt.extend(y)
        self.total += len(x);self.warmup += len(x)
        self.last_data=time.monotonic()
        if self.state=='BASELINE':
            self.base_samples.extend(y.tolist())
            if len(self.base_samples)>=int(BASELINE_SEC*self.fs):
                arr=np.asarray(self.base_samples[:int(BASELINE_SEC*self.fs)])
                m,s=noise_stats(arr)
                self.center,self.sigma=(m,s) if np.isfinite(s) and s>1e-6 else (None,None)
                if self.center is None:
                    self.state='IDLE';self.log_msg('BAD BASELINE: flat signal; retry without disconnecting')
                else:
                    self.state='CALIBRATING'
                    self.log_msg(f'BASELINE READY · median={m:.1f}µV · MAD-sigma={s:.1f}µV · p2p={np.ptp(arr):.1f}µV')
                self.base_bar.setValue(100)
        elif self.trial_kind:
            self.trial_samples.extend(y.tolist())
            if len(self.trial_samples)>=int(TRIAL_SEC*self.fs):
                self.finish_trial()
        elif self.state=='READY' and self.armed:
            # Analyze sliding window with 0.2 s step, only accept NEW event peaks.
            step=max(1,int(.2*self.fs))
            if self.total-self.live_next_index>=step and len(self.filt)>=int(TRIAL_SEC*self.fs):
                self.live_next_index=self.total
                window=np.asarray(list(self.filt)[-int(TRIAL_SEC*self.fs):])
                start_sample=self.total-len(window)
                evs=candidates(window,self.fs,self.center,self.sigma,self.factor.value())
                for ev in evs:
                    abs_peak=start_sample+int(ev['time']*self.fs)
                    if abs_peak<=getattr(self,'_last_peak_sample',-1):continue
                    # The peak is sufficiently inside the window (not the last 0.07s)
                    self._last_peak_sample=abs_peak
                    if (time.monotonic()-self.last_detection)<REFRACTORY_SEC: continue
                    self.history_x.append(len(self.history_x));self.history_y.append(ev['amplitude'])
                    self.last_detection=time.monotonic()
                    sent = self.keyboard.press()
                    if not self.keyboard.enabled:
                        status = 'BLINK DETECTED (keyboard output disabled)'
                    elif sent:
                        status = f'BLINK → W KEY DOWN ({W_HOLD_SEC:.1f}s)'
                    else:
                        status = 'BLINK DETECTED / W NOT SENT'
                        self.log_msg('W output: '+(self.keyboard.last_error or 'key still held / lockout'))
                    self.result.setText(status)
                    self.result.setStyleSheet(f'color:{GREEN}')
                    self.log_msg(f'BLINK -> W HOLD {W_HOLD_SEC:.1f}s: amp={ev["amplitude"]:.1f}µV duration={ev["duration"]:.3f}s velocity={ev["velocity"]:.0f}µV/s')
                    self._last_event=ev
                    self.curves['event'].setData(np.arange(len(window))/self.fs,window)
    def refresh(self):
        self.keyboard.tick()
        self.key_label.setText('W: PRESSED' if self.keyboard.down_until is not None else 'W: RELEASED')
        if self.connected and self.warmup and time.monotonic()-self.last_data>NO_DATA_SEC:
            self.disarm('No stream data for 5 s')
            self.connected=False
        arr=np.asarray(self.raw); filt=np.asarray(self.filt)
        n=min(len(arr),int(5*self.fs))
        if n:
            t=np.arange(n)/self.fs
            self.curves['raw'].setData(t,arr[-n:]);self.curves['filtered'].setData(t,filt[-n:])
        if self.history_x:
            self.curves['history'].setData(self.history_x[-60:],self.history_y[-60:])
        if self.state=='BASELINE':
            pct=min(100,round(len(self.base_samples)/(BASELINE_SEC*self.fs)*100))
            self.base_bar.setValue(pct)
            self.base_text.setText(f'RECORDING {pct}% · {len(self.base_samples)}/{int(BASELINE_SEC*self.fs)} samples')
        elif self.center is not None:
            self.base_text.setText(f'READY · sigma {self.sigma:.1f}µV · threshold {max(1.0,self.factor.value()*self.sigma):.1f}µV')
        if self.trial_kind:
            pct=min(100,round(len(self.trial_samples)/(TRIAL_SEC*self.fs)*100))
            self.trial_bar.setValue(pct)
            self.trial_text.setText(f'{self.trial_kind.upper()} · blink ONCE · {pct}%')
        last=getattr(self,'_last_event',None)
        self.metrics.setText(f'Sampling: {self.fs} Hz / samples {self.total:,}\n'
                             f'Threshold: {max(1.0,self.factor.value()*self.sigma):.1f} µV' if self.sigma else f'Sampling: {self.fs} Hz / samples {self.total:,}\nThreshold: —')
        if last:
            self.metrics.setText(self.metrics.text()+f'\nAmplitude: {last["amplitude"]:.1f} µV\nDuration: {last["duration"]:.3f} s\nVelocity: {last["velocity"]:.0f} µV/s')
        self.natural_btn.setEnabled(self.state=='CALIBRATING' and self.trial_kind is None and len(self.natural)<N_NATURAL)
        self.intentional_btn.setEnabled(self.state=='CALIBRATING' and self.trial_kind is None and len(self.intentional)<N_INTENTIONAL)
        self.arm_btn.setEnabled(self.connected and self.center is not None and self.trial_kind is None and self.state in ('CALIBRATING','READY'))
        self.baseline_btn.setEnabled(self.connected and self.trial_kind is None and self.state!='BASELINE')
    def closeEvent(self,event):
        self.timer.stop();self.keyboard.release()
        self.source.shutdown();self.source.wait(6000)
        event.accept()


def main():
    ap=argparse.ArgumentParser(description='LightBrain EOG standalone GUI / prototype')
    ap.add_argument('--connection',choices=['demo','sender','native','dongle'],default='sender',
                    help='sender = original ganglion_sender.py compatible native BLE; dongle = legacy BLED112 on COM port')
    ap.add_argument('--port',default='COM3')
    ap.add_argument('--fw2',action='store_true')
    ap.add_argument('--board-name',default='',help='Optional sender-compatible other_info string; usually leave blank')
    ap.add_argument('--no-keyboard',action='store_true',help='Disable desktop W output (simulation only)')
    args=ap.parse_args()
    app=QApplication(sys.argv);app.setStyle('Fusion')
    win=MainWindow(args.connection,args.port,args.fw2,not args.no_keyboard,args.board_name)
    win.show();sys.exit(app.exec_())

if __name__=='__main__':main()
