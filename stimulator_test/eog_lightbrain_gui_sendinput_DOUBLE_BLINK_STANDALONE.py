"""LightBrain EOG: two 20s calibration phases, personalized blink recognition,
two accepted intentional events <=3s apart -> W held 1s -> 2s cooldown.
Desktop keyboard simulation only, NOT a wheelchair motor controller.
Based on the user-provided eog_lightbrain_gui_sendinput(2).py.
"""
import argparse
import os
import ctypes
from ctypes import wintypes
import sys
import time
import threading
import csv
from pathlib import Path
from collections import deque

import numpy as np
from scipy.signal import butter, sosfilt
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QKeySequence
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton,
                             QHBoxLayout, QVBoxLayout, QGridLayout, QFrame, QProgressBar,
                             QShortcut, QPlainTextEdit, QCheckBox, QScrollArea)
import pyqtgraph as pg
# ----- Embedded blink core (standalone version) -----
"""EOG blink recognition core: adaptive candidate extraction + personalised
natural/intentional template classifier + two-blink timing gate.

Research-inspired, experimentally tuned. No clinical/physical wheelchair safety claim.
No PyQt or BrainFlow dependencies: makes the algorithm unit testable.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.signal import find_peaks

FS_DEFAULT = 200
CALIBRATION_SECONDS = 20
W_HOLD_SECONDS = 1.0
COOLDOWN_SECONDS = 2.0         # begins AFTER W releases
SECOND_BLINK_SECONDS = 5.0
MIN_INTERBLINK_SECONDS = 0.38  # avoid a blink's biphasic tail counting twice
REFRACTORY_SECONDS = 0.38
DETECT_WINDOW_SECONDS = 2.4
NO_DATA_SECONDS = 1.2


def robust_median_mad(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError('EOG data missing/nonfinite')
    m = float(np.median(x))
    return m, 1.4826 * float(np.median(np.abs(x-m)))


def natural_reference(x, fs):
    """Natural record can contain blinks. Estimate idle noise from QUIETER 0.5s
    blocks, not global MAD inflated by deliberate/spontaneous blinks.
    Always produce a reference; this is not a baseline pass/fail decision.
    """
    a = np.asarray(x, dtype=float)
    center, full_noise = robust_median_mad(a)
    step = max(8, int(fs * 0.5))
    chunk_mads = []
    for i in range(0, len(a)-step+1, step):
        _, s = robust_median_mad(a[i:i+step])
        if np.isfinite(s):
            chunk_mads.append(s)
    if chunk_mads:
        q = np.quantile(chunk_mads, 0.5)
        idle_chunks = [s for s in chunk_mads if s <= q]
        noise = float(np.median(idle_chunks)) if idle_chunks else full_noise
    else:
        noise = full_noise
    return center, max(1.0, noise)


def waveform(x, pk, fs, before=0.22, after=0.58, points=96):
    """Resample signed pulse in a fixed peak-centred window for NCC template."""
    x = np.asarray(x, dtype=float)
    ts = np.linspace(-before, after, points)
    src = pk + ts*fs
    if src.min() < 0 or src.max() >= len(x):
        return None
    y = np.interp(src, np.arange(len(x)), x)
    y = y - np.median(y[:max(3, int(points * .15))])
    if y[np.argmin(abs(ts))] < 0:
        y = -y
    norm = np.linalg.norm(y)
    return y/norm if norm > 1e-9 else None


def extract_events(samples, fs, center, sigma, min_amplitude=None):
    """Extract candidate peaks, group biphasic peaks, measure waveform features.

    Robust local amplitude threshold is for candidate extraction only, NOT a
    baseline quality threshold. Event decisions are made by the model.
    """
    x = np.asarray(samples, dtype=float)
    if x.size < max(20, int(.85*fs)) or not np.isfinite(x).all():
        return []
    y = x - float(center)
    thresh = max(5.0*float(sigma), 10.0)
    if min_amplitude is not None:
        thresh = max(thresh, float(min_amplitude))
    peaks, props = find_peaks(np.abs(y), height=thresh,
                              prominence=max(3.0*float(sigma), 5.0),
                              distance=max(1, int(.10*fs)))
    if len(peaks) == 0:
        return []
    # A biphasic blink produces a secondary lobe: group nearby local peaks.
    groups = [[int(peaks[0])]]
    for p in peaks[1:]:
        if (int(p)-groups[-1][-1])/fs <= .36:
            groups[-1].append(int(p))
        else:
            groups.append([int(p)])
    out = []
    for group in groups:
        # The later opposite-polarity lobe can be larger than the initial blink
        # peak after high-pass filtering. Prefer the FIRST prominent lobe,
        # so template timing/polarity remains stable across repeated blinks.
        strongest = max(group, key=lambda k: abs(y[k]))
        pk = group[0] if abs(y[group[0]]) >= .5*abs(y[strongest]) else strongest
        # Reject incomplete waveform (including onset/offset at buffer edges).
        shape = waveform(y, pk, fs)
        if shape is None:
            continue
        amp = abs(y[pk])
        bound = max(.20*amp, .35*thresh)
        lo, hi = pk, pk
        while lo > 0 and abs(y[lo-1]) > bound:
            lo -= 1
        while hi+1 < len(y) and abs(y[hi+1]) > bound:
            hi += 1
        dur = (hi-lo+1)/float(fs)
        if not .045 <= dur <= 1.3:
            continue
        seg = y[max(0, pk-int(.24*fs)):min(len(y),pk+int(.55*fs))]
        # Original shape and its time derivative (velocity) are classic blink descriptors.
        velocity = float(max(np.max(np.abs(np.diff(seg)))*fs, 1e-8)) if len(seg)>1 else 0.0
        energy = float(np.mean(seg*seg))
        out.append(dict(index=int(pk), time=float(pk/fs), amplitude=float(amp),
                        duration=float(dur), velocity=velocity, energy=energy,
                        shape=shape, fragments=len(group)))
    return out


def _ln_features(e):
    # Log scale stabilizes feature magnitudes across electrode placement/gain.
    return np.log(np.maximum([e['amplitude'], e['duration'], e['velocity']], 1e-8))


def _corr(a, b):
    aa = np.asarray(a, dtype=float); bb = np.asarray(b, dtype=float)
    aa = aa-aa.mean(); bb=bb-bb.mean()
    return float(np.dot(aa,bb)/(max(np.linalg.norm(aa)*np.linalg.norm(bb), 1e-10)))


class PersonalBlinkModel:
    """Personalized nearest-template + log-feature distance classifier.

    Calibration itself is never graded pass/fail; events can be few or ambiguous.
    If there are no intentional examples, model is unavailable and UI explains it.
    A new event must resemble intentional examples AND outperform the natural class.
    Outliers/spikes become ARTIFACT/UNKNOWN and cannot count towards commands.
    """
    def __init__(self, natural, intentional, noise_sigma):
        self.natural = list(natural)
        self.intentional = list(intentional)
        self.noise_sigma = float(noise_sigma)
        self.ready = bool(self.intentional)
        self.warning = ''
        self.threshold = max(5*self.noise_sigma, 10.0)
        self.natural_count = len(self.natural)
        self.intentional_count = len(self.intentional)
        if not self.ready:
            self.warning = 'No intentional blink found in the 20-second recording. Record the 20-second phase again.'
            return
        feats_i = np.array([_ln_features(e) for e in self.intentional])
        feats_n = np.array([_ln_features(e) for e in self.natural]) if self.natural else np.empty((0,3))
        pooled = np.vstack([feats_i,feats_n])
        # Robust class variability, with a minimum log-scale floor.
        median = np.median(pooled,axis=0)
        scale = 1.4826*np.median(np.abs(pooled-median),axis=0)
        if len(feats_n):
            gap = np.abs(np.median(feats_i,axis=0)-np.median(feats_n,axis=0))
            scale=np.maximum(scale, .30*gap)
        self.scale=np.maximum(scale, np.array([.20,.20,.25]))
        self.intent_feats=feats_i/self.scale
        self.natural_feats=feats_n/self.scale
        self.intent_med=np.median(feats_i,axis=0)/self.scale
        # Templates are participant-specific; an aligned normalized pulse shape.
        self.template = np.mean([e['shape'] for e in self.intentional],axis=0)
        self.template/=max(np.linalg.norm(self.template),1e-9)
        i_corr=np.array([_corr(e['shape'],self.template) for e in self.intentional])
        self.min_correlation=max(.56, float(np.percentile(i_corr, 10))-.16)
        # Use calibrated amplitude range to avoid enormous electrode spikes.
        amp_i=np.array([e['amplitude'] for e in self.intentional])
        self.amp_floor = max(self.threshold, .45*float(np.median(amp_i)))
        self.amp_ceiling=max(3.5*float(np.percentile(amp_i,95)), self.amp_floor*2)
        # Velocity-to-amplitude rejects sharp spikes of short duration.
        sharp=np.array([e['velocity']/max(e['amplitude'],1e-8) for e in self.intentional])
        self.max_sharp = max(1.9*float(np.percentile(sharp,95)), 5.0)
        # Normal class distance radius: learned nearest-neighbour spread.
        intra=[]
        for idx, v in enumerate(self.intent_feats):
            others=np.delete(self.intent_feats,idx,axis=0)
            if len(others): intra.append(float(np.min(np.linalg.norm(others-v,axis=1))))
        self.int_radius=max(1.8, 1.7*float(np.percentile(intra,80))) if intra else 2.8
        if len(feats_n)==0:
            self.warning='Natural blinks not found; classification has less protection from false positives.'
        elif len(feats_i)<3:
            self.warning='Few intentional blinks; results may be less reliable. Consider re-recording.'

    def predict(self, e):
        """Return (label, telemetry). Only 'INTENTIONAL' is command eligible."""
        if not self.ready:
            return 'UNKNOWN', {'reason':'NO_MODEL','corr':0.0}
        amp=e['amplitude']; sharp=e['velocity']/max(amp,1e-8)
        corr=_corr(e['shape'],self.template)
        if amp < self.amp_floor:
            return 'NATURAL / UNKNOWN', {'reason':'LOW_AMP','corr':corr}
        if amp > self.amp_ceiling or sharp > self.max_sharp:
            return 'ARTIFACT', {'reason':'SPIKE_OR_OUTLIER','corr':corr}
        if corr < self.min_correlation:
            return 'ARTIFACT', {'reason':'WAVEFORM_MISMATCH','corr':corr}
        feat=_ln_features(e)/self.scale
        di=float(np.min(np.linalg.norm(self.intent_feats-feat,axis=1)))
        # Compare natural and intentional in calibrated feature space.
        dn=float(np.min(np.linalg.norm(self.natural_feats-feat,axis=1))) if len(self.natural_feats) else float('inf')
        # Distinct intentional must win over spontaneous by a small margin.
        intentional_wins=(di + .20 < dn) if np.isfinite(dn) else True
        if di <= self.int_radius and intentional_wins:
            label='INTENTIONAL'
            reason='INTENTIONAL_WINS'
        else:
            label='NATURAL / UNKNOWN' if dn <= di+.20 else 'ARTIFACT'
            reason='NATURAL_WINS' if dn <= di+.20 else 'OUT_OF_TRAINING_RANGE'
        return label, {'reason':reason, 'corr':corr, 'd_int':di, 'd_nat':dn,
                       'radius':self.int_radius}


@dataclass
class GateResult:
    state: str
    event: str
    remaining: float
    command: bool=False


class DoubleBlinkGate:
    """Two independent accepted intentional blink EVENTS within 3s.

    Success: press W 1s, then cool down for full 2s AFTER the key releases.
    Timer tests use an injectable numeric timestamp (seconds).
    """
    def __init__(self, second_window=SECOND_BLINK_SECONDS, hold=W_HOLD_SECONDS,
                 cooldown=COOLDOWN_SECONDS, min_gap=MIN_INTERBLINK_SECONDS):
        self.window=float(second_window); self.hold=float(hold)
        self.cooldown=float(cooldown); self.min_gap=float(min_gap)
        self.reset()

    def reset(self):
        self.first_at=None
        self.hold_until=None
        self.cooldown_until=None
        self.state='WAIT_FIRST'

    def tick(self, now):
        now=float(now)
        if self.state=='HOLD' and now >= self.hold_until:
            self.state='COOLDOWN'
            return GateResult(self.state, 'RELEASE_W', max(0,self.cooldown_until-now))
        if self.state=='COOLDOWN' and now>=self.cooldown_until:
            self.state='WAIT_FIRST'
            return GateResult(self.state, 'READY',0)
        if self.state=='WAIT_SECOND' and now>self.first_at+self.window:
            self.first_at=None;self.state='WAIT_FIRST'
            return GateResult(self.state,'TIMEOUT_RESET',0)
        if self.state=='WAIT_SECOND':
            return GateResult(self.state,'WAITING',max(0,self.first_at+self.window-now))
        if self.state=='COOLDOWN':
            return GateResult(self.state,'COOLDOWN',max(0,self.cooldown_until-now))
        if self.state=='HOLD':
            return GateResult(self.state,'HOLDING',max(0,self.hold_until-now))
        return GateResult(self.state,'IDLE',0)

    def accept(self, now):
        now=float(now)
        self.tick(now)
        if self.state in ('HOLD','COOLDOWN'):
            return GateResult(self.state,'IGNORED_LOCKOUT',max(0,self.cooldown_until-now))
        if self.state=='WAIT_FIRST':
            self.first_at=now;self.state='WAIT_SECOND'
            return GateResult(self.state,'FIRST_ACCEPTED',self.window)
        delta=now-self.first_at
        if delta < self.min_gap:
            return GateResult(self.state,'TOO_CLOSE',max(0,self.window-delta))
        if delta>self.window:
            self.first_at=now
            return GateResult(self.state,'FIRST_ACCEPTED',self.window)
        self.first_at=None
        self.state='HOLD'
        self.hold_until=now+self.hold
        self.cooldown_until=self.hold_until+self.cooldown
        return GateResult(self.state,'COMMAND_W',self.hold,True)

# ----- End embedded blink core -----

try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    BRAINFLOW_AVAILABLE = True
except ImportError:
    BRAINFLOW_AVAILABLE = False

FS_DEMO = 200
W_HOLD_SEC = W_HOLD_SECONDS  # legacy Keyboard.press() compatibility
WARMUP_SEC = 2.0
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
        self._demo_training = None
        self._demo_next = 0.0
        self.fs = FS_DEMO
    def demo_blink(self, kind):
        with self._demo_lock:
            self._demo_blink = kind

    def demo_training(self, kind):
        # During the 20s demo calibration synthesize spontaneous or deliberate blinks.
        with self._demo_lock:
            self._demo_training = kind
            self._demo_next = self._demo_t + (1.5 if kind else 10000)
            self._demo_pulses = []
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
                        training = self._demo_training
                        if training and self._demo_t >= self._demo_next:
                            self._demo_pulses.append((self._demo_t + .2,training))
                            self._demo_next += (3.2 if training=='natural' else 2.15)
                    if cmd:
                        self._demo_pulses.append((self._demo_t + .25, cmd))
                    x = rng.normal(0, 8, n) + 11*np.sin(2*np.pi*.2*t)
                    keep = []
                    for start, kind in self._demo_pulses:
                        d = t-start
                        if kind == 'intentional':
                            amp, width = 460, .16
                        elif kind == 'artifact':
                            amp, width = 1200, .012
                        else:
                            amp, width = 230, .085
                        x += amp*np.exp(-.5*(d/width)**2) - (.30*amp*np.exp(-.5*((d-.28)/(.10 if kind=='intentional' else .07))**2) if kind!='artifact' else 0.0)
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


# Blink extraction and classification are provided by blink_core.py.


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
    """Original four-panel layout; fixed two-stage continuous calibration."""
    def __init__(self, mode, port, fw2, keyboard=False, board_name=''):
        super().__init__()
        self.mode,self.port,self.fw2 = mode,port,fw2
        self.fs=FS_DEMO
        self.filter=StreamingFilter(self.fs)
        self.raw=deque(maxlen=self.fs*15)
        self.filt=deque(maxlen=self.fs*15)
        self.total=0
        self.connected=False
        self.warmup=0
        self.last_data=time.monotonic()
        self.phase=None
        self.phase_samples=[]
        self.records={'natural':None,'intentional':None}
        self.center=None;self.sigma=None
        self.natural_events=[];self.intent_events=[]
        self.model=None
        self.armed=False
        self.gate=DoubleBlinkGate()
        self.last_processed_peak=-100000000
        self.live_next_index=0
        self.history_x=[];self.history_y=[]
        self.last_event=None
        self.keyboard=Keyboard()
        self.last_result='WAITING'
        self.logs_dir=Path(__file__).resolve().parent/'logs'
        self.logs_dir.mkdir(parents=True,exist_ok=True)
        self.session_id=time.strftime('%Y%m%d_%H%M%S')
        self.events_file=(self.logs_dir/f'events_{self.session_id}.csv').open('w',newline='',encoding='utf-8-sig')
        self.events_csv=csv.writer(self.events_file)
        self.events_csv.writerow(['timestamp','phase','classification','peak_sample','amplitude_uv','duration_sec','velocity_uv_per_sec','ncc','d_int','d_nat','gate_event'])
        self.build_ui()
        self.chk_keyboard.setChecked(bool(keyboard))
        self.toggle_keyboard(bool(keyboard))
        self.source=Acquisition(mode,port,fw2,board_name)
        self.source.samples.connect(self.on_samples)
        self.source.status.connect(self.set_connection)
        self.source.problem.connect(self.on_problem)
        self.source.start()
        self.timer=QTimer(self);self.timer.timeout.connect(self.refresh);self.timer.start(50)
        QShortcut(QKeySequence(Qt.Key_Space),self,activated=self.stop_output)

    def build_ui(self):
        self.setWindowTitle('LightBrain EOG · Double Intentional Blink → W · 2 × 20 s')
        self.resize(1380,830)
        self.setStyleSheet(f"""QWidget {{background:{BG};color:#e6edf3;font-family:Consolas;}}
        QPushButton {{background:#1d2935;border:1px solid #0e7490;border-radius:5px;padding:9px;}}
        QPushButton:hover {{background:#23465a;}}
        QPushButton:disabled {{color:#5d6874;border-color:#303943;}}
        QFrame#panel {{background:{PANEL};border:1px solid #0e7490;border-radius:8px;}}
        QProgressBar {{background:#1c2333;border:1px solid #0e7490;border-radius:4px;text-align:center;}}
        QProgressBar::chunk {{background:{ACCENT};}}""")
        root=QVBoxLayout();central=QWidget();central.setLayout(root);self.setCentralWidget(central)
        root.setContentsMargins(10,8,10,8)
        head=QHBoxLayout();root.addLayout(head)
        title=QLabel('◉  LightBrain / TWO BLINK CONFIRMATION')
        title.setFont(QFont('Consolas',15,QFont.Bold));title.setStyleSheet(f'color:{ACCENT}')
        head.addWidget(title);head.addStretch()
        self.conn=QLabel('○ Connecting...');head.addWidget(self.conn)
        content=QHBoxLayout();root.addLayout(content,1)
        graphs=QGridLayout();content.addLayout(graphs,7)
        pg.setConfigOptions(antialias=False)
        self.curves={}
        for key,r,c,name,color in [
            ('raw',0,0,'RAW EOG · Channel 1','#8799fc'),
            ('filtered',0,1,'FILTERED EOG · 0.1–10 Hz',GREEN),
            ('event',1,0,'LAST EVENT · 2.4 s WINDOW','#ffc857'),
            ('history',1,1,'DETECTED BLINK AMPLITUDE (µV)',ACCENT)]:
            pw=pg.PlotWidget(title=name,background='#0a0e14')
            pw.showGrid(x=True,y=True,alpha=.2)
            pw.setLabel('bottom','Time','s' if key!='history' else 'event')
            pw.setLabel('left','Amplitude','µV')
            self.curves[key]=pw.plot(pen=pg.mkPen(color,width=1.5))
            graphs.addWidget(pw,r,c)
        scroller=QScrollArea();scroller.setWidgetResizable(True);scroller.setFrameShape(QFrame.NoFrame)
        scroller.setMinimumWidth(380)
        content.addWidget(scroller,3)
        panel=QFrame();panel.setObjectName('panel');scroller.setWidget(panel)
        right=QVBoxLayout(panel);right.setContentsMargins(12,10,12,10)
        def section(name):
            l=QLabel(name);l.setStyleSheet(f'color:{ACCENT};font-weight:bold;border-bottom:1px solid #175a70;padding:3px;')
            right.addWidget(l)
        section('01 / NATURAL BLINK · 20 SECONDS')
        self.natural_btn=QPushButton('▶ Record normal sitting / NATURAL (20s)')
        self.natural_btn.clicked.connect(lambda:self.start_phase('natural'));right.addWidget(self.natural_btn)
        self.natural_bar=QProgressBar();self.natural_bar.setRange(0,100);right.addWidget(self.natural_bar)
        self.natural_label=QLabel('Not recorded · blink naturally / sit normally')
        self.natural_label.setWordWrap(True);right.addWidget(self.natural_label)
        section('02 / INTENTIONAL BLINK · 20 SECONDS')
        self.intentional_btn=QPushButton('▶ Record INTENTIONAL blinking (20s)')
        self.intentional_btn.clicked.connect(lambda:self.start_phase('intentional'));right.addWidget(self.intentional_btn)
        self.intentional_bar=QProgressBar();self.intentional_bar.setRange(0,100);right.addWidget(self.intentional_bar)
        self.intentional_label=QLabel('Not recorded · intentionally blink every ~2s')
        self.intentional_label.setWordWrap(True);right.addWidget(self.intentional_label)
        btnrow=QHBoxLayout()
        self.validate_btn=QPushButton('✓ VALIDATE / ARM')
        self.validate_btn.clicked.connect(self.validate);btnrow.addWidget(self.validate_btn,2)
        self.reset_btn=QPushButton('RESET');self.reset_btn.clicked.connect(self.clear_calibration)
        btnrow.addWidget(self.reset_btn,1);right.addLayout(btnrow)
        self.calibration_label=QLabel('Collect both 20-second phases. No baseline pass/fail threshold.')
        self.calibration_label.setWordWrap(True);self.calibration_label.setStyleSheet('color:#ffc857')
        right.addWidget(self.calibration_label)
        section('03 / REAL-TIME EVENT CLASSIFICATION')
        self.result=QLabel('DISARMED · NO COMMAND')
        self.result.setFont(QFont('Consolas',14,QFont.Bold));self.result.setWordWrap(True)
        self.result.setStyleSheet(f'color:{RED}');right.addWidget(self.result)
        self.classification=QLabel('CLASS: — / ARTIFACT: —')
        self.classification.setWordWrap(True);right.addWidget(self.classification)
        self.progress_label=QLabel('CONFIRM: 0 / 2')
        self.progress_label.setFont(QFont('Consolas',12,QFont.Bold));right.addWidget(self.progress_label)
        self.confirm_bar=QProgressBar();self.confirm_bar.setRange(0,100)
        right.addWidget(self.confirm_bar)
        section('04 / COMMAND → DESKTOP W')
        self.command_label=QLabel('■  STOP / W RELEASED')
        self.command_label.setWordWrap(True);self.command_label.setFont(QFont('Consolas',14,QFont.Bold))
        self.command_label.setStyleSheet(f'color:{RED};padding:7px;border:1px solid #5a2026;')
        right.addWidget(self.command_label)
        self.key_label=QLabel('W: RELEASED · no physical wheelchair connection')
        right.addWidget(self.key_label)
        self.chk_keyboard=QCheckBox('Enable Windows W key (simulator only)')
        self.chk_keyboard.toggled.connect(self.toggle_keyboard)
        right.addWidget(self.chk_keyboard)
        self.timing=QLabel('2 intentional blinks ≤3s  →  W DOWN 1s  →  delay 2s')
        self.timing.setWordWrap(True);self.timing.setStyleSheet('color:#ffc857')
        right.addWidget(self.timing)
        if self.mode=='demo':
            section('DEMO / INJECT SINGLE EVENTS')
            demos=QHBoxLayout()
            for label,kind in [('NATURAL','natural'),('INTENT','intentional'),('ARTIFACT','artifact')]:
                btn=QPushButton(label)
                btn.clicked.connect(lambda checked=False,k=kind:self.source.demo_blink(k))
                demos.addWidget(btn)
            right.addLayout(demos)
        self.stop_btn=QPushButton('■ EMERGENCY STOP [SPACE]')
        self.stop_btn.setStyleSheet(f'background:#5a2026;border:1px solid {RED};padding:12px;font-weight:bold')
        self.stop_btn.clicked.connect(self.stop_output);right.addWidget(self.stop_btn)
        self.metrics=QLabel('Sampling: —\nCenter/noise: —\nLast feature: —')
        self.metrics.setWordWrap(True);right.addWidget(self.metrics)
        self.log=QPlainTextEdit();self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(400);self.log.setMinimumHeight(140)
        right.addWidget(self.log)
        self.log_msg('Sequence: NATURAL 20s → INTENTIONAL 20s → VALIDATE / ARM.')
        self.log_msg('Only two accepted intentional blinks in 3s trigger W. No baseline pass/fail.')
        self.log_msg('Desktop simulation only. Windows W disabled by default.')

    def log_msg(self,s):
        self.log.appendPlainText(time.strftime('%H:%M:%S  ')+str(s))

    def log_event(self,phase,label,e=None,metrics=None,gate=''):
        e=e or {};metrics=metrics or {}
        self.events_csv.writerow([time.strftime('%Y-%m-%d %H:%M:%S'),phase,label,
            e.get('absolute_index',''),e.get('amplitude',''),e.get('duration',''),
            e.get('velocity',''),metrics.get('corr',''),metrics.get('d_int',''),
            metrics.get('d_nat',''),gate])
        self.events_file.flush()

    def set_connection(self,s):
        self.connected=s.startswith('CONNECTED') or s.startswith('DEMO')
        self.conn.setText(('● ' if self.connected else '○ ')+s)
        self.conn.setStyleSheet(f'color:{GREEN if self.connected else RED}')
        if s=='DISCONNECTED':
            self.disarm('Disconnected')

    def on_problem(self,s):
        self.log_msg('ACQUISITION ERROR: '+s)
        self.disarm('Connection error')

    def toggle_keyboard(self, enabled):
        try:
            self.keyboard.set_enabled(enabled)
            self.log_msg('Windows desktop W output: '+('ENABLED' if enabled else 'DISABLED / DRY RUN'))
        except Exception as e:
            self.chk_keyboard.blockSignals(True);self.chk_keyboard.setChecked(False)
            self.chk_keyboard.blockSignals(False)
            self.log_msg('Keyboard unavailable: '+str(e))

    def stop_output(self):
        self.disarm('Emergency STOP (Space)')

    def disarm(self,reason):
        self.keyboard.release()
        self.armed=False
        self.gate.reset()
        if self.phase:
            self.phase=None;self.phase_samples=[]
            if self.mode=='demo' and getattr(self,'source',None):
                self.source.demo_training(None)
        self.result.setText('STOP / DISARMED')
        self.result.setStyleSheet(f'color:{RED}')
        self.command_label.setText('■  STOP / W RELEASED')
        self.command_label.setStyleSheet(f'color:{RED};padding:7px;border:1px solid #5a2026;')
        self.progress_label.setText('CONFIRM: 0 / 2')
        self.confirm_bar.setValue(0)
        if getattr(self,'log',None): self.log_msg('STOP · '+reason)

    def clear_calibration(self):
        self.disarm('Calibration reset')
        self.phase=None;self.phase_samples=[]
        self.records={'natural':None,'intentional':None}
        self.center=None;self.sigma=None
        self.natural_events=[];self.intent_events=[];self.model=None
        for bar in (self.natural_bar,self.intentional_bar):bar.setValue(0)
        self.natural_label.setText('Not recorded · blink naturally / sit normally')
        self.intentional_label.setText('Not recorded · intentionally blink every ~2s')
        self.calibration_label.setText('Calibration reset. Natural 20s → Intentional 20s.')
        if self.mode=='demo':self.source.demo_training(None)

    def start_phase(self,kind):
        if not self.connected:
            self.calibration_label.setText('Wait for Ganglion / demo connection, then retry.')
            self.log_msg('Calibration waiting: no EOG connection.');return
        if self.warmup < int(WARMUP_SEC*self.fs):
            self.calibration_label.setText('Waiting for 2-second filter warmup (not baseline pass/fail).')
            return
        if self.phase:
            self.log_msg('Finish current 20-second phase first.');return
        if kind=='intentional' and self.records['natural'] is None:
            self.calibration_label.setText('First record NATURAL for 20 seconds.');return
        self.disarm('Starting calibration: '+kind)
        if kind=='natural':
            self.records={'natural':None,'intentional':None}
            self.center=None;self.sigma=None;self.model=None
            self.natural_events=[];self.intent_events=[]
            self.intentional_bar.setValue(0)
            self.intentional_label.setText('Not recorded · intentionally blink every ~2s')
        else:
            self.records['intentional']=None;self.model=None;self.intent_events=[]
        self.phase=kind;self.phase_samples=[]
        (self.natural_bar if kind=='natural' else self.intentional_bar).setValue(0)
        self.calibration_label.setText('Recording '+kind.upper()+' · 20 seconds of received samples')
        self.log_msg('RECORD '+kind.upper()+' (20s) · blink naturally' if kind=='natural' else
                     'RECORD INTENTIONAL (20s) · blink on purpose every ~2 seconds')
        if self.mode=='demo':self.source.demo_training(kind)

    def finish_phase(self):
        kind=self.phase
        data=np.asarray(self.phase_samples[:int(self.fs*CALIBRATION_SECONDS)],dtype=float)
        self.records[kind]=data
        self.phase=None;self.phase_samples=[]
        if self.mode=='demo':self.source.demo_training(None)
        file=self.logs_dir/f'{kind}_{self.session_id}.csv'
        with file.open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.writer(f);w.writerow(['sample_index','seconds','filtered_eog_uv'])
            w.writerows((idx,idx/self.fs,float(val)) for idx,val in enumerate(data))
        if kind=='natural':
            # NOT pass/fail; natural record contains spontaneous blinks as requested.
            self.center,self.sigma=natural_reference(data,self.fs)
            self.natural_events=extract_events(data,self.fs,self.center,self.sigma)
            self.natural_bar.setValue(100)
            self.natural_label.setText(f'COMPLETE 20s · {len(data):,} samples · {len(self.natural_events)} natural blink candidates')
            self.calibration_label.setText('Natural saved. Next: INTENTIONAL 20 seconds.')
        else:
            self.intent_events=extract_events(data,self.fs,self.center,self.sigma)
            self.model=PersonalBlinkModel(self.natural_events,self.intent_events,self.sigma)
            self.intentional_bar.setValue(100)
            self.intentional_label.setText(f'COMPLETE 20s · {len(data):,} samples · {len(self.intent_events)} intentional blink candidates')
            self.calibration_label.setText('Both 20-second recordings saved. Press VALIDATE / ARM.')
            if self.model.warning:
                self.log_msg('MODEL NOTICE · '+self.model.warning)
        self.log_msg(f'{kind.upper()} COMPLETE · {len(data)} samples · saved {file.name}')
        self.log_event(kind,'CALIBRATION_COMPLETE',gate=f'{len(data)} samples')

    def validate(self):
        """Always clickable; explain missing step; no quality-gated baseline check."""
        if not self.connected:
            self.calibration_label.setText('VALIDATE: no EOG stream. Connect first.');return
        if self.phase:
            self.calibration_label.setText('VALIDATE: wait until active 20s recording finishes.');return
        if self.records['natural'] is None:
            self.calibration_label.setText('VALIDATE: first record NATURAL 20s.');return
        if self.records['intentional'] is None:
            self.calibration_label.setText('VALIDATE: now record INTENTIONAL 20s.');return
        if self.model is None:
            self.model=PersonalBlinkModel(self.natural_events,self.intent_events,self.sigma)
        if not self.model.ready:
            self.calibration_label.setText('VALIDATE: no intentional blink detected in the 20s recording. Repeat that phase; no baseline was rejected.')
            self.log_msg('VALIDATE: no usable intentional template; command output stays DISARMED.')
            return
        self.keyboard.release()
        self.gate.reset();self.armed=True
        self.live_next_index=self.total
        self.last_processed_peak=self.total
        self.result.setText('ARMED · waiting for 2 intentional blinks')
        self.result.setStyleSheet(f'color:{GREEN}')
        self.command_label.setText('READY · W RELEASED / DRY RUN' if not self.keyboard.enabled else 'READY · W RELEASED')
        self.calibration_label.setText('VALIDATED / ARMED · natural vs intentional template + artifact rejection.')
        if self.model.warning:self.log_msg('WARNING · '+self.model.warning)
        self.log_msg(f'ARMED · natural={len(self.natural_events)} examples; intentional={len(self.intent_events)} examples; no baseline pass/fail.')

    def _ensure_sampling_rate(self):
        newfs=int(getattr(self.source,'fs',self.fs))
        if newfs<=0 or newfs==self.fs:return
        self.disarm(f'Sampling rate changed: {self.fs}→{newfs}')
        self.fs=newfs
        self.filter=StreamingFilter(self.fs)
        self.raw=deque(maxlen=self.fs*15);self.filt=deque(maxlen=self.fs*15)
        self.total=0;self.warmup=0
        self.records={'natural':None,'intentional':None}
        self.phase=None;self.phase_samples=[]
        self.center=None;self.sigma=None;self.model=None
        self.natural_events=[];self.intent_events=[]
        self.natural_bar.setValue(0);self.intentional_bar.setValue(0)
        self.log_msg(f'Sampling rate updated to {newfs} Hz. Calibration cleared.')

    def on_samples(self,x):
        self._ensure_sampling_rate()
        x=np.asarray(x,dtype=float)
        if x.size==0:return
        if not np.isfinite(x).all():
            self.disarm('Invalid sample (NaN or infinity)');return
        y=self.filter.apply(x)
        self.raw.extend(x.tolist());self.filt.extend(y.tolist())
        self.total+=len(x);self.warmup+=len(x)
        self.last_data=time.monotonic()
        if self.phase:
            self.phase_samples.extend(y.tolist())
            if len(self.phase_samples)>=int(CALIBRATION_SECONDS*self.fs):
                self.finish_phase()
            return
        if not self.armed or not self.model or not self.model.ready:return
        step=max(1,int(.10*self.fs))
        window_size=int(DETECT_WINDOW_SECONDS*self.fs)
        if self.total-self.live_next_index<step or len(self.filt)<window_size:return
        self.live_next_index=self.total
        window=np.asarray(list(self.filt)[-window_size:])
        min_amp=max(self.model.threshold,self.model.amp_floor*.62)
        events=extract_events(window,self.fs,self.center,self.sigma,min_amp)
        start_abs=self.total-len(window)
        for e in events:
            abs_pk=start_abs+e['index']
            if abs_pk-self.last_processed_peak<int(REFRACTORY_SECONDS*self.fs):continue
            # Do not count old pulses from before Validate or last phase.
            if abs_pk<=0:continue
            self.last_processed_peak=abs_pk
            e['absolute_index']=abs_pk
            label,met=self.model.predict(e)
            self.last_event=e
            self.history_x.append(len(self.history_x));self.history_y.append(e['amplitude'])
            self.curves['event'].setData(np.arange(len(window))/self.fs,window)
            self.classification.setText(f'CLASS: {label}  |  NCC={met.get("corr",0):.2f} · {met.get("reason","")}')
            gate_event='REJECTED'
            if label=='INTENTIONAL':
                result=self.gate.accept(time.monotonic())
                gate_event=result.event
                if result.event=='FIRST_ACCEPTED':
                    self.result.setText('INTENTIONAL 1/2 · second blink within 3 seconds')
                    self.result.setStyleSheet('color:#ffc857')
                    self.log_msg('INTENTIONAL 1/2 → waiting second intentional ≤3s')
                elif result.command:
                    self.trigger_w()
                elif result.event=='IGNORED_LOCKOUT':
                    self.log_msg('Blink ignored: W hold / 2s cooldown active')
            else:
                # Crucial: artifacts/natural blinks NEVER advance the confirmation gate.
                self.log_msg(f'{label} / rejected · amplitude={e["amplitude"]:.1f}µV · {met.get("reason","")}')
            self.log_event('LIVE',label,e,met,gate_event)

    def trigger_w(self):
        output='DRY RUN (keyboard disabled)'
        if self.keyboard.enabled:
            pressed=self.keyboard.press()
            output='W KEY DOWN' if pressed else 'SENDINPUT FAILED: '+(self.keyboard.last_error or 'key already down')
            if not pressed:self.disarm('Keyboard press failed');return
        self.command_label.setText('▲  FORWARD / W · 1s  ['+output+']')
        self.command_label.setStyleSheet(f'color:{GREEN};padding:7px;border:1px solid #2e7b55;')
        self.result.setText('2/2 CONFIRMED → W 1s → cooldown 2s')
        self.result.setStyleSheet(f'color:{GREEN}')
        self.log_msg('COMMAND W · 2 intentional blinks confirmed · 1s hold · 2s delay after release · '+output)
        self.log_event('COMMAND','W_CONFIRMED',gate=output)

    def refresh(self):
        now=time.monotonic()
        self.keyboard.tick()
        if self.connected and self.warmup and now-self.last_data>NO_DATA_SECONDS:
            self.disarm('NO DATA >1.2s (software fail-stop)')
            self.connected=False
            self.conn.setText('○ Signal timeout / DISCONNECTED')
            self.conn.setStyleSheet(f'color:{RED}')
        if self.armed:
            r=self.gate.tick(now)
            if r.event=='RELEASE_W':
                self.keyboard.release()
                self.command_label.setText(f'■ W RELEASED · COOLDOWN {r.remaining:.1f}s')
                self.command_label.setStyleSheet('color:#ffc857;padding:7px;border:1px solid #81651a;')
                self.log_msg('W UP · now 2s cooldown')
            elif r.event=='TIMEOUT_RESET':
                self.result.setText('3s EXPIRED · 0/2 · first blink reset')
                self.result.setStyleSheet('color:#ffc857')
                self.log_msg('TIMEOUT: second intentional not detected ≤3s → start again at 0/2')
            elif r.event=='READY':
                self.command_label.setText('READY · W RELEASED / next command allowed')
                self.command_label.setStyleSheet(f'color:{ACCENT};padding:7px;border:1px solid #0e7490;')
                self.result.setText('ARMED · waiting for 2 intentional blinks')
                self.result.setStyleSheet(f'color:{GREEN}')
            if self.gate.state=='WAIT_SECOND':
                self.progress_label.setText(f'CONFIRM: 1 / 2 · {r.remaining:.1f}s left')
                self.confirm_bar.setValue(min(100,int(100*(1-r.remaining/SECOND_BLINK_SECONDS))))
            elif self.gate.state=='HOLD':
                self.progress_label.setText(f'W HOLD · {r.remaining:.1f}s remaining')
                self.confirm_bar.setValue(100)
            elif self.gate.state=='COOLDOWN':
                self.progress_label.setText(f'COOLDOWN · {r.remaining:.1f}s remaining')
                self.confirm_bar.setValue(min(100,int(100*(1-r.remaining/COOLDOWN_SECONDS))))
            else:
                self.progress_label.setText('CONFIRM: 0 / 2')
                self.confirm_bar.setValue(0)
        self.key_label.setText('W: PRESSED' if self.keyboard.down_until is not None else 'W: RELEASED')
        if self.phase:
            val=min(100,int(len(self.phase_samples)*100/(CALIBRATION_SECONDS*self.fs)))
            bar=self.natural_bar if self.phase=='natural' else self.intentional_bar
            lab=self.natural_label if self.phase=='natural' else self.intentional_label
            bar.setValue(val)
            lab.setText(f'RECORDING {val}% · {len(self.phase_samples):,}/{int(CALIBRATION_SECONDS*self.fs):,} samples')
        self.natural_btn.setEnabled(self.connected and self.phase is None)
        self.intentional_btn.setEnabled(self.connected and self.phase is None and self.records['natural'] is not None)
        self.validate_btn.setEnabled(True) # Always clickable and explains missing stage
        arr=np.asarray(self.raw);fil=np.asarray(self.filt)
        n=min(len(arr),int(5*self.fs))
        if n:
            t=np.arange(n)/self.fs
            self.curves['raw'].setData(t,arr[-n:])
            self.curves['filtered'].setData(t,fil[-n:])
        if self.history_x:
            self.curves['history'].setData(self.history_x[-60:],self.history_y[-60:])
        event=self.last_event or {}
        self.metrics.setText(f'Sampling: {self.fs} Hz / {self.total:,} samples\n'
            f'Natural center: {self.center:.1f} µV / quiet σMAD: {self.sigma:.2f} µV\n'
            f'Last amplitude: {event.get("amplitude",0):.1f} µV · width: {event.get("duration",0):.3f}s'
            if self.center is not None else f'Sampling: {self.fs} Hz / {self.total:,} samples\nWaiting for natural baseline')

    def closeEvent(self,event):
        self.timer.stop()
        self.disarm('Window closing')
        self.source.shutdown()
        self.source.wait(6000)
        try:self.events_file.close()
        except Exception:pass
        event.accept()


def main():
    ap=argparse.ArgumentParser(description='LightBrain / Two 20s baselines / Two intentional blinks within 3s')
    ap.add_argument('--connection',choices=['demo','sender','native','dongle'],default='sender')
    ap.add_argument('--port',default='COM3')
    ap.add_argument('--fw2',action='store_true')
    ap.add_argument('--board-name',default='')
    ap.add_argument('--keyboard',action='store_true',help='Explicitly enable desktop Windows SendInput W')
    ap.add_argument('--no-keyboard',action='store_true',help='Dry-run only; backward-compatible flag')
    args=ap.parse_args()
    app=QApplication(sys.argv);app.setStyle('Fusion')
    win=MainWindow(args.connection,args.port,args.fw2,args.keyboard and not args.no_keyboard,args.board_name)
    win.show();sys.exit(app.exec_())

if __name__=='__main__': main()
