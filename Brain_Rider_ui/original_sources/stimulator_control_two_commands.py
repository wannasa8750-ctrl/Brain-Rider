"""
LightBrain — Two-Command SSVEP Control
=============================================
Integrated SSVEP-based Wheelchair Controller

Bug Fixes & Improvements (v3 → v4):
  [BUG-1] เพิ่ม Notch Filter 50 Hz (IIR Butterworth Order 4) หลัง Bandpass
          เพื่อตัดสัญญาณรบกวนไฟบ้านกระแสสลับ (Power Line Noise) ที่อาจรั่วผ่าน
          Transition Band ของ Bandpass Filter ได้
  [BUG-2] เปลี่ยน Baseline เป็น Running Average PSD (สะสมตลอด 30 วินาที)
          แทนการดึง PSD ก้อนเดียวเมื่อสิ้นสุดช่วงเวลา ทำให้ค่า Baseline เสถียร
          และทนทานต่อ Artifact ชั่วคราว (กระพริบตา, ขยับตัว)
  [BUG-3] เพิ่ม State Machine 3 สถานะ (IDLE / RECORDING / READY) แทนการใช้
          Boolean flag เดียว ทำให้ควบคุม Logic การเก็บ Baseline ได้ถูกต้อง
          และเพิ่มปุ่ม Clear Baseline พร้อม Countdown สำหรับ Warm-up
  [BUG-4] เพิ่ม Filter Warm-up Guard: ป้องกันการเก็บ Baseline ด้วยสัญญาณ
          ที่ยังไม่เสถียร (Transient) ในช่วง 0.5 วินาทีแรกหลังเริ่มโปรแกรม
  [BUG-5] ปรับ Dynamic PSD Window: ใช้ข้อมูล 5 วินาทีล่าสุดแทน 3 วินาที
          เพื่อให้ Frequency Resolution สูงกว่า (~0.2 Hz vs ~0.33 Hz)
          ทำให้แยกแยะความถี่ 7, 8, 9 Hz ออกจากกันได้ชัดเจนขึ้น

v4.6 — PSDA (วิธีมาตรฐาน SSVEP) + null calibration:
  • คะแนนต่อเป้าหมาย = SNR แบบ PSDA (กำลังในแถบ ±0.2 Hz ÷ ค่าเฉลี่ย bin ข้างเคียง 0.6–3 Hz) รวม harmonic ถ่วงน้ำหนัก
  • ตอน baseline 30 s เก็บ "null" (ตอนไม่ได้มอง) ของแต่ละเป้าหมาย → ตัดสินด้วย z-score เทียบ null ของผู้ใช้เอง
  • ผู้ชนะต้อง z ≥ PSDA_Z_THR และนำอันดับสอง ≥ PSDA_Z_GAP ; z ≥ PSDA_Z_STRONG สั่งทันที, ปานกลางต้องยืนยันซ้ำ 2 epoch
  • เลิกใช้ PEAK_MODE เดิมที่ "สั่งตามความถี่ที่แรงสุดเสมอ" (สั่งเองทุก epoch แม้ไม่ได้มอง)
  • แก้ artifact rejection ที่ทิ้งทุก epoch เพราะค่าอ้างอิง RMS ค้างค่าเก่า

Command timing (v4.1):
  • ก่อนเก็บ/ระหว่างเก็บ Baseline แสดง No Signal ไม่ตรวจจับ ไม่สั่งรถ
  • หลังมี Baseline ใช้ข้อมูล 8 s ล่าสุด (EPOCH_SEC) ตัดสินใจทุก 3 s (DECISION_STEP_SEC) เลือกความถี่ที่พีคแรงที่สุด
  • กด Space = หยุดฉุกเฉินทันที
  • สั่งแล้วค้าง: เดินหน้า/ถอยหลัง 2 s, เลี้ยวซ้าย/ขวา 1 s (COMMAND_HOLD_SEC)
  • หลังเริ่มคำสั่ง หน่วง 2 s ก่อนรับความถี่ใหม่ (COMMAND_LOCKOUT_SEC); STOP (17 Hz) ผ่านทันที
  • DETECTION_SNR_THR 2.5 → 7.0

Combines:
  • Ganglion Sender  — EEG data acquisition via OpenBCI Ganglion (BrainFlow)
  • Signal Analyzer  — Bandpass + Notch Filter + FFT + Welch PSD + SSVEP Detection
  • Keyboard Control  — Sends WASD keystrokes + visual feedback

SSVEP Frequencies (fundamental + harmonics):
   9 Hz  → [9, 18, 27] — เลี้ยวซ้าย (Left / A)
   15 Hz → [15, 30]    — เลี้ยวขวา (Right / D)
   No detection → STOP (release all keys); Space → emergency STOP.

Usage:
  python wheelchair_control.py              # UDP mode (needs ganglion_sender)
  python wheelchair_control.py --brainflow  # Direct BrainFlow connection
  python wheelchair_control.py --demo       # Synthetic signal demo
  python wheelchair_control.py --sender     # Run as Ganglion sender only
"""

import sys
import os
import time
import socket
import threading
import argparse
import ctypes
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import signal as sig

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QProgressBar, QFrame, QSizePolicy, QShortcut,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QColor, QPainter, QBrush, QPen, QKeySequence
import pyqtgraph as pg

# ── Try optional BrainFlow ──────────────────────────────────────────
try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    HAS_BRAINFLOW = True
except ImportError:
    HAS_BRAINFLOW = False


# ╔════════════════════════════════════════════════════════════════════╗
# ║                     CONFIGURATION                                 ║
# ╚════════════════════════════════════════════════════════════════════╝

# ── UDP ──
UDP_IP          = "127.0.0.1"
UDP_PORT        = 5005

# ── Board ──
SERIAL_PORT     = "COM3"
BOARD_ID        = BoardIds.GANGLION_NATIVE_BOARD if HAS_BRAINFLOW else -1
SCALE_FACTOR    = 0.02235
SELECTED_CH     = 2          # Channel 3 (0-indexed)
BOARD_NAME      = ""         # e.g. "Ganglion-b2a4"

# ── Signal Processing ──
SAMPLE_RATE     = 200        # Hz  (Ganglion default)
FILTER_ORDER    = 16
FILTER_LOW      = 1.0        # Hz
FILTER_HIGH     = 40.0       # Hz
ELLIP_RP        = 0.5        # passband ripple dB
ELLIP_RS        = 60.0       # stopband attenuation dB

# ── [BUG-1 FIX] Notch Filter for 50 Hz Power Line Noise ──
NOTCH_FREQ      = 50.0       # Hz — AC power line (Thailand/EU standard)
NOTCH_BW        = 2.0        # Hz — notch width (48–52 Hz)
NOTCH_ORDER     = 4          # Butterworth order for flat passband

# ── FFT / PSD ──
PSD_NPERSEG     = 800        # v4.3: segment 4 s (0.25 Hz) เฉลี่ย 3 segment ใน 5 s → นิ่งกว่า periodogram เดียว แต่ยังแยกเป้าหมายห่าง 1 Hz ได้
PSD_NOVERLAP    = 700        # overlap 87.5% (5 s = 3 segments)
FFT_SIZE        = 1024       # zero-padded FFT for display

# ── Display ──
DISPLAY_SEC     = 5          # seconds shown on time-domain graphs
DISPLAY_SAMPLES = SAMPLE_RATE * DISPLAY_SEC
UPDATE_MS       = 50         # 20 FPS graph update
BUFFER_MAX      = SAMPLE_RATE * 30   # 30 s circular buffer

# ── [BUG-5 FIX] PSD Analysis Window — extended to 5s for better frequency resolution ──
ANALYSIS_SEC    = 5          # seconds of data used for PSD (was 3s)

# ── Baseline ──
BASELINE_DURATION = 30       # seconds

# ── [BUG-3 FIX] Baseline State Machine ──
class BaselineState:
    IDLE      = "IDLE"       # Not recorded yet / cleared
    RECORDING = "RECORDING"  # Currently accumulating data
    READY     = "READY"      # Baseline computed and active

# ── Detection (v4.2) ──
DETECTION_SMOOTH  = 0.25     # EMA alpha (ใช้เฉพาะโหมด smooth=True; โหมด epoch ไม่ใช้)
DETECTION_SNR_THR = 3.6      # เกณฑ์ของ "คะแนนเฉลี่ย" (เดิม 6.0 ของคะแนนดิบคนละสเกล)
                             # จำลอง: 3.6 → เดินผิด ≈0.03/นาที | 3.2 → ≈0.12/นาที แต่ไวขึ้น (ปรับซ้ำด้วยข้อมูลจริง ดู [decision] ใน console)
DETECTION_AVG_EPOCHS = 1     # v4.3: ใช้เฉพาะ epoch ล่าสุด (ไม่เฉลี่ยข้ามรอบ → ติดเร็วขึ้น)
DETECTION_MARGIN  = 1.2      # winner must exceed runner-up by this factor (ไม่ใช้เมื่อ PEAK_MODE)
# ── v4.3 PEAK MODE: เลือกความถี่ที่แรงที่สุดใน EPOCH_SEC ล่าสุดเสมอ (ติดง่าย) ──
# ── v4.6 PSDA (Power Spectral Density Analysis — วิธีมาตรฐานของ SSVEP) ──────────────
#    1) SNR(f) = กำลังเฉลี่ยในแถบรอบ f  ÷  กำลังเฉลี่ยของ bin ข้างเคียง (ข้ามช่วง guard รอบ f และรอบเป้าหมายอื่น)
#    2) รวม SNR ของ f, 2f, 3f แบบถ่วงน้ำหนัก (harmonic combination)
#    3) calibrate "null" (ตอนไม่ได้มอง) จาก baseline 30 s → ได้ค่าเฉลี่ย/SD ของคะแนนแต่ละเป้าหมาย
#       ตัดสินด้วย z-score เทียบ null  (= ตั้ง threshold จากข้อมูลผู้ใช้เอง ไม่ใช่เลขตายตัว)
#    4) ผู้ชนะต้อง: z ≥ PSDA_Z_THR และนำอันดับสองอย่างน้อย PSDA_Z_GAP
#    5) z สูงมาก (≥ PSDA_Z_STRONG) → สั่งทันที | ปานกลาง → ต้องเป็นเป้าหมายเดิมติดกัน PSDA_CONFIRM epoch
PSDA_MODE         = True     # False = กลับไปใช้ตรรกะ PEAK_MODE เดิมของ v4.3
PSDA_PEAK_HALF_BW = 0.2      # Hz: กำลังที่เป้าหมาย = ค่าเฉลี่ย PSD ในช่วง f ± ค่านี้ (จำลอง: 0.2 แยกได้ดีกว่า 0.3)
PSDA_NOISE_GAP    = 0.6      # Hz: ข้าม bin ใกล้เป้าหมาย (กัน leakage ของ peak เอง)
PSDA_NOISE_SPAN   = 3.0      # Hz: noise = ค่าเฉลี่ย bin ที่ห่าง GAP..SPAN ทั้งสองข้าง (จำลอง: 3.0 เสถียรกว่า 2.0 เพราะเฉลี่ย bin มากขึ้น)
PSDA_Z_THR        = 2.4      # z-score ขั้นต่ำเทียบ null (จำลอง สั่งรอบเดียว: 3.5→1.3 | 3.0→1.6 | 2.5→2.4 | 2.0→3.8 ครั้ง/นาทีตอนไม่ได้มอง; ลด = ติดง่ายขึ้นแต่สั่งเองมากขึ้น)
PSDA_Z_STRONG     = 6.0      # z ระดับ "ชัดเจน" → สั่งทันทีใน epoch เดียว
PSDA_Z_GAP        = 0.6      # z ของผู้ชนะต้องนำอันดับสองอย่างน้อยเท่านี้ (1.0→0.6 ติดง่ายขึ้น)
PSDA_CONFIRM      = 1        # 1 = ผ่านเกณฑ์รอบเดียวสั่งทันที | 2 = ระดับปานกลางต้องเป็นเป้าหมายเดิมติดกัน 2 epoch (สั่งเองน้อยลงมาก แต่ช้าลง)
PSDA_SD_FLOOR     = 0.10     # SD ต่ำสุดของ null ในสเกล ln(SNR) (กัน z พุ่งเมื่อ baseline นิ่งผิดปกติ)
PSDA_NULL_STEP_SEC = 1.0     # ขั้นเลื่อนของ epoch ตอน calibrate null จาก baseline
PEAK_MODE         = True     # True = ไม่ใช้ threshold/margin เลือกเป้าหมายที่คะแนน (PSD÷baseline÷noise) สูงสุด
PEAK_MIN_SCORE    = 0.0      # 0 = สั่งตามความถี่ best เสมอ (ถ้าสั่งเองตอนไม่ได้มอง ให้เพิ่ม เช่น 1.5 → 3.0)
PEAK_MARGIN       = 1.0      # 1.0 = ไม่ต้องชนะอันดับสอง (ถ้าสั่งเองบ่อย ให้เพิ่ม เช่น 1.1)
# ── v4.3 ADAPTIVE: ไม่ใช้ threshold ตายตัว แต่เทียบ best กับ "ค่ากลางของเป้าหมายที่เหลือ" (noise floor สดของรอบนั้น) ──
PEAK_ADAPTIVE     = False    # FIX: สั่งตาม best frequency ทันที ไม่บังคับ ratio/คะแนนขั้นต่ำแบบ adaptive
PEAK_RATIO_STRONG = 1.8      # best ÷ median(ที่เหลือ) ≥ ค่านี้ → สั่งทันที (ติดง่ายเมื่อเด่นชัด)
PEAK_RATIO_WEAK   = 1.4      # ≥ ค่านี้ แต่ < STRONG → สั่งเมื่อ best เป็นตัวเดิมกับรอบก่อน (รอบติดกันยืนยัน)
PEAK_NORM_HIST    = 12       # ปรับสเกลรายความถี่: หารคะแนนด้วยค่ากลางของ N รอบล่าสุดของความถี่นั้น (ความถี่ที่ติดสูงเสมอ เช่น 15 Hz จะถูกลดลง) 0 = ปิด
PEAK_NORM_MIN_N   = 4        # ต้องมีอย่างน้อยกี่รอบก่อนเริ่มปรับสเกล
# ── v4.3 ARTIFACT REJECTION (อิเล็กโทรดแห้ง 1 ตำแหน่ง: ขยับ/กะพริบ/หลุดสัมผัส ทำให้สัญญาณกระชาก) ──
ARTIFACT_REJECT   = True
ARTIFACT_RMS_X    = 4.0      # v4.6: 2.5→4.0 | RMS ของ epoch > ค่ากลางของ RMS รอบก่อน ๆ × ค่านี้ → ทิ้ง epoch (ไม่สั่งรถ)
ARTIFACT_RMS_MAX_CONSEC = 2  # v4.6: ถูกทิ้งเพราะ RMS ติดกันกี่รอบ จึงถือเป็นระดับใหม่ → ปรับค่าอ้างอิงแล้วรับ epoch
ARTIFACT_PEAK_X   = 7.0      # max|x| ÷ RMS > ค่านี้ (มี spike) → ทิ้ง epoch
ARTIFACT_HIST     = 12       # จำนวน epoch ล่าสุดที่ใช้หาค่ากลาง RMS
ARTIFACT_MIN_N    = 4        # ต้องมีประวัติอย่างน้อยกี่รอบก่อนใช้กฎ RMS
PEAK_STRONG_ABS   = 3.5      # คะแนน (หลังปรับสเกล ≈1 ตอนไม่ได้มอง) ที่สูงพอจะสั่งทันทีใน 5 วิ
PEAK_WEAK_ABS     = 2.2      # คะแนนขั้นต่ำของระดับ "ปานกลาง" (ต้องเป็นตัวเดิม 2 รอบติด = 10 วิ)
HARMONIC_BW       = 0.5      # Hz bandwidth around each harmonic for power extraction (v4.3: 0.8→0.5 ไม่ให้ 7.5 Hz เข้าแถบ 7 Hz)
PEAK_ALIGN_WIN    = 0.75     # v4.3: ดูพีคสูงสุดในช่วง ±0.75 Hz รอบความถี่เป้าหมาย
PEAK_ALIGN_TOL    = 0.3      # v4.3: พีคต้องอยู่ห่างเป้าหมายไม่เกิน 0.3 Hz ไม่งั้นไม่นับ (เช่น พีค 7.5 Hz จะไม่ถูกนับเป็น 7 หรือ 8 Hz)
NOISE_EXCLUDE_BW  = 1.0      # Hz bandwidth to exclude around targets when computing noise
LOCAL_NOISE_GAP   = 0.8      # Hz: ระยะห่างขั้นต่ำจากเป้าหมายที่เริ่มนับเป็น noise
LOCAL_NOISE_SPAN  = 2.5      # Hz: ระยะไกลสุดของ noise ข้างเคียง
LOCAL_NOISE_EXCL  = 0.5      # Hz: ตัด bin รอบความถี่เป้าหมายอื่นออกจาก noise

# ── Keyboard Control ──
KEY_HOLD_SEC      = 0.3      # (ใช้กับ send_command เดิม; โหมดปกติใช้ COMMAND_HOLD_SEC)
CONTROL_COOLDOWN  = 0.5      # minimum time between commands (seconds)

# ── Command Timing (ตัวคุมเวลาคำสั่ง — class CommandGate) ──────────────
REQUIRE_BASELINE       = True   # ต้องทำ Baseline ก่อนจึงสั่งรถเคลื่อนที่ได้ (STOP ใช้ได้เสมอ)
EPOCH_SEC              = 5.0    # หน้าต่างวิเคราะห์ (v4.3: ใช้ข้อมูลล่าสุด 5 วินาที เลือกความถี่ที่พีคแรงที่สุด)
DECISION_STEP_SEC      = 5.0    # ตัดสินใจทุก 5 วินาที (เท่ากับ EPOCH_SEC → หน้าต่างไม่ซ้อนกัน) ควรมากกว่า COMMAND_HOLD_SEC
# ค้างคำสั่งตามเวลานี้ (วินาที) ต่อทิศ — เลี้ยวซ้าย/ขวา 2 วินาที (เดิม 1) ถ้าเลี้ยวมากไปให้ลดลง
# (อยากเลี้ยวมากขึ้น: มองซ้ำหลังพ้น lockout หรือเพิ่มค่า left/right)
COMMAND_HOLD_SEC = {"left": 1.0, "right": 1.0}
COMMAND_LOCKOUT_SEC    = 2.0    # หลังเริ่มคำสั่ง รอครบเท่านี้ก่อนรับคำสั่งเคลื่อนที่ใหม่
LOCKOUT_AFTER_STOP_SEC = 0.4    # หลัง STOP รอสั้น ๆ ก่อนรับคำสั่งใหม่

# ── SSVEP Frequencies ──────────────────────────────────────────────
#    fundamental: [freq, harmonic_1, harmonic_2]
#    weights: relative importance of each harmonic
SSVEP_CONFIG = {
    9:  {"freqs": [9.0, 18.0, 27.0], "weights": [1.0, 0.5, 0.25],
         "color": "#ffc107", "label": "9 Hz  Left",
         "action": "left", "key_name": "A"},
    15: {"freqs": [15.0, 30.0], "weights": [1.0, 0.5],
         "color": "#ff5722", "label": "15 Hz Right",
         "action": "right", "key_name": "D"},
}

# ── Keyboard Scan Codes (DirectInput) ────────────────────────────────
KEY_CODES = {"A": 0x1E, "D": 0x20}

# ── UI Colors ────────────────────────────────────────────────────────
CLR_BG        = "#0d1117"
CLR_PANEL     = "#161b22"
CLR_BORDER    = "#0e7490"
CLR_ACCENT    = "#00d4ff"
CLR_TEXT      = "#e6edf3"
CLR_DIM       = "#8b949e"
CLR_GREEN     = "#2ea043"
CLR_RED       = "#f85149"
CLR_NOSIGNAL  = "#6e7681"


# ╔════════════════════════════════════════════════════════════════════╗
# ║              COMMAND GATE (ยืนยัน → ค้างตามเวลา → หน่วง)           ║
# ╚════════════════════════════════════════════════════════════════════╝
# ตรวจเจอความถี่เดิมต่อเนื่อง >= confirm → สั่งค้างตาม hold_sec ของทิศนั้น
# ระหว่างค้าง/lockout ไม่รับคำสั่งเคลื่อนที่ใหม่ (ทิศเดิมที่ยังมองต่อเนื่องจะต่อช่วงถัดไปได้เลย)
# STOP ผ่านทันทีทุกสถานะ

IDLE, ACTIVE, LOCKOUT = "IDLE", "ACTIVE", "LOCKOUT"


@dataclass
class GateResult:
    action: str                 # คำสั่งที่ต้องส่งออกตอนนี้: "stop" หรือทิศเคลื่อนที่
    state: str                  # IDLE / ACTIVE / LOCKOUT
    event: Optional[str]        # "start" | "end" | "stop" | None  (เกิดครั้งเดียวต่อเหตุการณ์)
    hold_sec: float             # event=="start": ระยะเวลาที่สั่งค้าง
    remaining: float            # ACTIVE: เวลาที่เหลือ / LOCKOUT: เวลารอก่อนรับคำสั่งใหม่
    candidate: Optional[str]    # ความถี่ที่กำลังยืนยันอยู่
    progress: float             # ความคืบหน้าการยืนยัน 0..1


class CommandGate:
    def __init__(self, hold_sec: dict, lockout_sec: float = 2.0,
                 confirm_sec: float = 0.5, stop_confirm_sec: float = 0.25,
                 stop_lockout_sec: float = 0.5, clock=time.monotonic):
        self.hold_sec = dict(hold_sec)
        self.lockout_sec = lockout_sec
        self.confirm_sec = confirm_sec
        self.stop_confirm_sec = stop_confirm_sec
        self.stop_lockout_sec = stop_lockout_sec
        self._clock = clock
        self.reset()

    # ------------------------------------------------------------
    def reset(self):
        self.state = IDLE
        self._active = None
        self._active_until = 0.0
        self._lockout_until = 0.0
        self._cand = None
        self._cand_since = 0.0

    def force_stop(self, now: float = None):
        """หยุดทันที (ไม่มีข้อมูล EEG / ปิดโปรแกรม) แล้วเข้า lockout สั้น ๆ"""
        now = self._clock() if now is None else now
        was_active = self.state == ACTIVE
        self._active = None
        self._cand = None
        self.state = LOCKOUT
        self._lockout_until = max(self._lockout_until, now + self.stop_lockout_sec)
        return was_active

    # ------------------------------------------------------------
    def update(self, detected: Optional[str], now: float = None,
               allow_move: bool = True) -> GateResult:
        """detected: "left"/"right"/"stop"/None
        allow_move=False: ไม่รับคำสั่งเคลื่อนที่ (เช่น ยังไม่ได้ทำ baseline) แต่ STOP ยังทำงาน
        """
        now = self._clock() if now is None else now
        event = None
        hold = 0.0

        if detected not in (None, "stop") and not allow_move:
            detected = None

        # 1) หมดเวลาคำสั่งที่ค้างอยู่
        if self.state == ACTIVE and now >= self._active_until:
            self.state = LOCKOUT
            self._active = None
            event = "end"
        if self.state == LOCKOUT and now >= self._lockout_until:
            self.state = IDLE

        # 2) ติดตามผู้สมัคร (ต้องคงที่ต่อเนื่อง)
        if detected is None:
            self._cand = None
        elif detected != self._cand:
            self._cand, self._cand_since = detected, now
        # ระหว่าง ACTIVE/LOCKOUT ไม่นับเวลายืนยันของคำสั่งเคลื่อนที่ใหม่
        # ยกเว้น "ทิศเดียวกับที่กำลังค้างอยู่" ที่ยังตรวจเจอต่อเนื่อง: นับต่อได้
        # เพื่อให้มองต่อเนื่องแล้วรถวิ่งต่อไม่สะดุด (ถ้าตรวจหลุด ต้องยืนยันใหม่)
        if (self.state != IDLE and self._cand not in (None, "stop")
                and not (self.state == ACTIVE and self._cand == self._active)):
            self._cand_since = now

        waited = (now - self._cand_since) if self._cand else 0.0

        # 3) STOP: ผ่านทุกสถานะ
        if self._cand == "stop" and waited >= self.stop_confirm_sec:
            if self.state == ACTIVE:
                self.state = LOCKOUT
                self._active = None
                self._lockout_until = now + self.stop_lockout_sec
                event = "stop"
        # 4) เริ่มคำสั่งเคลื่อนที่
        elif (self._cand not in (None, "stop") and self.state == IDLE
              and waited >= self.confirm_sec):
            hold = float(self.hold_sec.get(self._cand, 0.0))
            if hold > 0:
                self._active = self._cand
                self._active_until = now + hold
                self._lockout_until = now + max(self.lockout_sec, hold)
                self.state = ACTIVE
                event = "start"

        # 5) ผลลัพธ์
        if self.state == ACTIVE:
            remaining = max(0.0, self._active_until - now)
        elif self.state == LOCKOUT:
            remaining = max(0.0, self._lockout_until - now)
        else:
            remaining = 0.0
        need = self.stop_confirm_sec if self._cand == "stop" else self.confirm_sec
        if self._cand and self.state == IDLE:
            progress = 1.0 if need <= 0 else min(1.0, waited / need)
        else:
            progress = 0.0
        return GateResult(
            action=self._active if self.state == ACTIVE else "stop",
            state=self.state, event=event,
            hold_sec=hold if event == "start" else 0.0,
            remaining=remaining, candidate=self._cand, progress=progress)


# ╔════════════════════════════════════════════════════════════════════╗
# ║              KEYBOARD CONTROLLER (Windows DirectInput)            ║
# ╚════════════════════════════════════════════════════════════════════╝

class KeyboardController:
    """Sends keyboard events using Windows SendInput API (DirectInput)."""

    def __init__(self):
        self._available = (os.name == "nt")
        if self._available:
            self._SendInput = ctypes.windll.user32.SendInput
            self._PUL = ctypes.POINTER(ctypes.c_ulong)
        self._last_command_time = 0.0
        self._active_key = None
        self._hold_timer = None

    def _make_input(self, scan_code, flags):
        """Build an INPUT structure for SendInput."""
        if not self._available:
            return None

        class KeyBdInput(ctypes.Structure):
            _fields_ = [("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort),
                        ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong),
                        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

        class HardwareInput(ctypes.Structure):
            _fields_ = [("uMsg", ctypes.c_ulong), ("wParamL", ctypes.c_short),
                        ("wParamH", ctypes.c_ushort)]

        class MouseInput(ctypes.Structure):
            _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long),
                        ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
                        ("time", ctypes.c_ulong),
                        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

        class Input_I(ctypes.Union):
            _fields_ = [("ki", KeyBdInput), ("mi", MouseInput), ("hi", HardwareInput)]

        class Input(ctypes.Structure):
            _fields_ = [("type", ctypes.c_ulong), ("ii", Input_I)]

        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput(0, scan_code, flags, 0, ctypes.pointer(extra))
        return Input(ctypes.c_ulong(1), ii_)

    def press_key(self, scan_code):
        if not self._available:
            return
        x = self._make_input(scan_code, 0x0008)  # KEYEVENTF_SCANCODE
        if x:
            self._SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    def release_key(self, scan_code):
        if not self._available:
            return
        x = self._make_input(scan_code, 0x0008 | 0x0002)  # SCANCODE | KEYUP
        if x:
            self._SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    def release_all(self):
        if self._hold_timer is not None:
            self._hold_timer.cancel()
            self._hold_timer = None
        for code in KEY_CODES.values():
            self.release_key(code)
        self._active_key = None

    def hold(self, action: str, seconds: float):
        """กดปุ่มของทิศนั้นค้างไว้ `seconds` วินาที (เรียกครั้งเดียวต่อคำสั่งจาก CommandGate)"""
        key_name = {"left": "A", "right": "D"}.get(action)
        if key_name not in KEY_CODES:
            return None
        self.release_all()
        code = KEY_CODES[key_name]
        self.press_key(code)
        self._active_key = code
        t = threading.Timer(seconds, self.release_key, args=(code,))
        t.daemon = True
        t.start()
        self._hold_timer = t
        return key_name

    def send_command(self, action: str):
        """Send a wheelchair command. Returns the key name or 'STOP'."""
        now = time.time()
        if now - self._last_command_time < CONTROL_COOLDOWN:
            return None

        # Release any currently held key
        self.release_all()

        if action == "stop" or action is None:
            self._last_command_time = now
            return "STOP"

        # Map action to key
        action_to_key = {"left": "A", "right": "D"}

        key_name = action_to_key.get(action)
        if key_name and key_name in KEY_CODES:
            scan_code = KEY_CODES[key_name]
            self.press_key(scan_code)
            self._active_key = scan_code
            self._last_command_time = now

            # Schedule key release after hold duration
            def _release_later():
                time.sleep(KEY_HOLD_SEC)
                self.release_key(scan_code)

            t = threading.Thread(target=_release_later, daemon=True)
            t.start()
            return key_name

        return None


# ╔════════════════════════════════════════════════════════════════════╗
# ║                  SIGNAL PROCESSOR                                 ║
# ╚════════════════════════════════════════════════════════════════════╝

class SignalProcessor:
    """Bandpass filter + Notch filter + FFT + Welch PSD — all in one.

    Pipeline (v4):
      1. Elliptic IIR Bandpass 1–40 Hz  (removes DC drift & EMG/muscle noise)
      2. Butterworth IIR Notch 50 Hz    [BUG-1 FIX] (removes AC power line noise)
      3. Welch PSD / FFT for spectral analysis
    """

    def __init__(self, fs=SAMPLE_RATE):
        self.fs = fs
        nyq = fs / 2.0

        # ── Design Elliptic IIR Bandpass Filter ──
        # bandpass doubles the order internally, so use half
        eff_order = FILTER_ORDER // 2
        low_norm  = max(FILTER_LOW  / nyq, 0.001)
        high_norm = min(FILTER_HIGH / nyq, 0.999)
        self.sos_bp = sig.ellip(eff_order, ELLIP_RP, ELLIP_RS,
                                [low_norm, high_norm],
                                btype="bandpass", output="sos")

        # ── Notch 50 Hz (iirnotch; ซ้ำซ้อนกับ bandpass 40 Hz แต่ไม่เสียหาย) ──
        # Use iirnotch which is purpose-built for single-frequency rejection
        # Q factor controls bandwidth: Q = f0 / bw
        notch_q = NOTCH_FREQ / NOTCH_BW          # Q = 50/2 = 25
        b_notch, a_notch = sig.iirnotch(NOTCH_FREQ / nyq, notch_q)
        self.sos_notch = sig.tf2sos(b_notch, a_notch)

        # ── Initialize filter states ──
        self._zi_bp    = np.zeros_like(sig.sosfilt_zi(self.sos_bp))
        self._zi_notch = np.zeros_like(sig.sosfilt_zi(self.sos_notch))

        # ── [BUG-4 FIX] Warm-up guard: discard transient samples ──
        self._warmup = int(fs * 2.0)   # v4.2: 2 s (เดิม 0.5 s ไม่พอสำหรับ HPF 1 Hz)
        self._count  = 0

    # ── real-time filtering (block mode — more efficient than sample-by-sample) ──
    def filter_block(self, block: np.ndarray) -> np.ndarray:
        """Apply Bandpass then Notch filter to a block of samples."""
        data = block.astype(np.float64)
        if self._count == 0 and len(data) > 0:
            # v4.2: ตั้ง state เริ่มต้นตามระดับ DC ของตัวอย่างแรก ลด transient ขนาดใหญ่
            self._zi_bp = sig.sosfilt_zi(self.sos_bp) * data[0]
        # Stage 1: Bandpass (1–40 Hz)
        out, self._zi_bp = sig.sosfilt(self.sos_bp, data, zi=self._zi_bp)
        # Stage 2: Notch (50 Hz) — [BUG-1 FIX]
        out, self._zi_notch = sig.sosfilt(self.sos_notch, out, zi=self._zi_notch)
        self._count += len(block)
        return out

    # Keep sample-by-sample API for compatibility
    def filter_sample(self, x: float) -> float:
        return self.filter_block(np.array([x]))[0]

    @property
    def is_warmed_up(self) -> bool:
        """[BUG-4 FIX] True only after first 0.5s of samples have been processed."""
        return self._count >= self._warmup

    # ── Welch PSD ───────────────────────────────────────────────────
    def compute_psd(self, data: np.ndarray):
        """Compute Power Spectral Density using Welch's method.

        Uses Hann window with 75% overlap for good frequency resolution
        and variance reduction. The nfft is set to max(FFT_SIZE, nperseg)
        for zero-padded interpolation.
        """
        n = len(data)
        if n < 64:
            return np.array([]), np.array([])
        nperseg = min(PSD_NPERSEG, n)
        noverlap = min(PSD_NOVERLAP, nperseg - 1)
        f, pxx = sig.welch(data, fs=self.fs, nperseg=nperseg,
                           noverlap=noverlap, nfft=max(FFT_SIZE, nperseg),
                           window="hann", scaling="density", detrend="constant")
        return f, pxx

    # ── raw FFT magnitude ──────────────────────────────────────────
    def compute_fft(self, data: np.ndarray):
        """Compute FFT magnitude spectrum with Hanning window and zero-padding."""
        n = len(data)
        if n < 64:
            return np.array([]), np.array([])
        win = np.hanning(n)
        padded = FFT_SIZE if FFT_SIZE > n else n
        Y = np.fft.rfft(data * win, n=padded)
        mag = (2.0 / np.sum(win)) * np.abs(Y)   # v4.2: ชดเชย coherent gain ของ Hann
        freqs = np.fft.rfftfreq(padded, d=1.0 / self.fs)
        return freqs, mag


# ╔════════════════════════════════════════════════════════════════════╗
# ║                  SSVEP DETECTOR                                   ║
# ╚════════════════════════════════════════════════════════════════════╝

class SSVEPDetector:
    """Multi-harmonic, baseline-normalized SSVEP detector.

    Detection Algorithm:
    1. Compute Welch PSD of the filtered EEG signal
    2. Optionally normalize PSD by a pre-recorded baseline
    3. For each SSVEP frequency, compute weighted sum of SNR at
       fundamental + harmonics (SNR = peak_power / noise_floor)
    4. Noise floor is computed as the median PSD excluding target bands
    5. Apply temporal smoothing (EMA) to reduce false detections
    6. Winner must exceed both absolute threshold AND margin over runner-up
    """

    def __init__(self, fs=SAMPLE_RATE):
        self.fs = fs
        self.baseline_psd  = None   # stored baseline PSD array
        self.baseline_freq = None   # matching frequency axis
        self._scores_ema   = {k: 0.0 for k in SSVEP_CONFIG}

        # Collect all target frequencies for noise exclusion
        self._all_target_freqs = []
        for cfg in SSVEP_CONFIG.values():
            self._all_target_freqs.extend(cfg["freqs"])

        # ── [BUG-2 FIX] Running average accumulator state ──
        self._bl_sum   = None   # np.ndarray: cumulative PSD sum
        self._bl_freq  = None   # np.ndarray: frequency axis reference
        self._bl_count = 0      # int: number of frames accumulated
        self._bl_frames = []    # v4.3: เก็บทุกเฟรมเพื่อหา median (ทนต่อ artifact ตอนเก็บ baseline)
        self._warned_size = False
        self.null_mu = self.null_sd = None          # v4.6: null calibration (ตอนไม่ได้มอง)
        self._confirm_key, self._confirm_n = None, 0

    def set_baseline(self, freq_axis: np.ndarray, psd: np.ndarray):
        """Set a final baseline PSD (used by running-average accumulator)."""
        self.baseline_freq = freq_axis.copy()
        self.baseline_psd  = psd.copy()
        # clamp small values to avoid division-by-zero
        self.baseline_psd[self.baseline_psd < 1e-15] = 1e-15

    # ── [BUG-2 FIX] Running Average Baseline Accumulation ─────────────
    def accumulate_baseline(self, freq_axis: np.ndarray, psd: np.ndarray):
        """Add one PSD frame to the running average accumulator.

        Called every tick while RECORDING. The running mean is incrementally
        updated using Welford's online algorithm to avoid floating-point
        accumulation errors from summing many large values.
        """
        if len(psd) == 0:
            return
        self._bl_count += 1
        if self._bl_sum is None:
            self._bl_sum   = psd.copy()
            self._bl_freq  = freq_axis.copy()
            self._bl_frames = [psd.copy()]
        else:
            # Resize if frequency axis length changed (shouldn't happen, but safe)
            if len(psd) == len(self._bl_sum):
                self._bl_sum += psd
                self._bl_frames.append(psd.copy())
            # else: skip mismatched frame

    def finalize_baseline(self):
        """Compute mean PSD from accumulated frames and store as baseline.

        Returns True if successful, False if not enough data.
        """
        if self._bl_count < 1 or self._bl_sum is None:
            return False
        if len(self._bl_frames) >= 5:
            base_psd = np.median(np.vstack(self._bl_frames), axis=0)   # v4.3: median ไม่โดน spike ดึง
        else:
            base_psd = self._bl_sum / self._bl_count
        self.set_baseline(self._bl_freq, base_psd)
        # Reset accumulators
        self._bl_frames = []
        self._bl_sum   = None
        self._bl_freq  = None
        self._bl_count = 0
        return True

    def reset_accumulator(self):
        """Reset running-average accumulators (call when starting a new recording)."""
        self._bl_frames = []
        self._bl_sum   = None
        self._bl_freq  = None
        self._bl_count = 0

    def clear_baseline(self):
        self.baseline_psd  = None
        self.baseline_freq = None

    def _power_at(self, freqs, psd, target_hz, bw=HARMONIC_BW):
        """Extract peak power in a narrow band [target-bw/2, target+bw/2].

        Uses the MAXIMUM power within the band (not mean) because SSVEP
        produces sharp spectral peaks. This is more robust than averaging
        which dilutes the peak with nearby noise.
        """
        mask = (freqs >= target_hz - bw / 2) & (freqs <= target_hz + bw / 2)
        if not np.any(mask):
            # Fallback: nearest frequency bin
            idx = np.argmin(np.abs(freqs - target_hz))
            return psd[idx]
        return np.max(psd[mask])

    def _compute_noise_floor(self, freqs, psd):
        """Fallback noise floor: median ใน passband 3–38 Hz นอก bin ของเป้าหมาย
        (v4.2: เดิมใช้ 2–100 Hz ซึ่งรวม stopband 40–100 Hz ทำให้ noise ต่ำเกินจริง)"""
        m = (freqs >= 3.0) & (freqs <= FILTER_HIGH - 2.0) & ~self._target_mask(freqs, NOISE_EXCLUDE_BW / 2)
        if np.sum(m) < 5:
            return np.median(psd) + 1e-15
        return np.median(psd[m]) + 1e-15

    def _target_mask(self, freqs, half_bw):
        m = np.zeros(len(freqs), dtype=bool)
        for tf in self._all_target_freqs:
            m |= np.abs(freqs - tf) <= half_bw
        return m

    def _band_power(self, freqs, psd, target_hz, bw=HARMONIC_BW):
        """กำลังเฉลี่ยในแถบ ±bw/2 รอบเป้าหมาย"""
        mask = np.abs(freqs - target_hz) <= bw / 2
        if not np.any(mask):
            return psd[np.argmin(np.abs(freqs - target_hz))]
        return float(np.mean(psd[mask]))

    def _peak_aligned(self, freqs, psd, target_hz):
        """True ถ้าพีคสูงสุดรอบเป้าหมาย (±PEAK_ALIGN_WIN) อยู่ตรงเป้าหมายจริง
        (v4.3: กันพีคที่เยื้อง เช่น 7.5 Hz ถูกนับเป็น 7 Hz เพราะ leakage)"""
        m = np.abs(freqs - target_hz) <= PEAK_ALIGN_WIN
        if not np.any(m):
            return True
        pf = freqs[m][np.argmax(psd[m])]
        return abs(pf - target_hz) <= PEAK_ALIGN_TOL

    def _local_noise(self, freqs, psd, target_hz, excl):
        """Noise รอบเป้าหมาย: median ของ bin ที่ห่าง LOCAL_NOISE_GAP..LOCAL_NOISE_SPAN Hz
        (ไม่รวม bin ของเป้าหมายอื่น) → ปรับตาม alpha/1-f ที่เปลี่ยนไปเองในบริเวณนั้น"""
        d = np.abs(freqs - target_hz)
        nb = ((d >= LOCAL_NOISE_GAP) & (d <= LOCAL_NOISE_SPAN) & ~excl
              & (freqs >= 2.0) & (freqs <= FILTER_HIGH - 1.0))
        if np.sum(nb) >= 6:
            return np.median(psd[nb]) + 1e-15
        return self._compute_noise_floor(freqs, psd)

    # ── v4.6 PSDA ────────────────────────────────────────────────────
    def _psda_snr(self, freqs, spec, f0, excl):
        """SNR แบบ PSDA มาตรฐาน: ค่าเฉลี่ยกำลังในแถบ f0±PSDA_PEAK_HALF_BW
        ÷ ค่าเฉลี่ยของ bin ข้างเคียง (ห่าง PSDA_NOISE_GAP..PSDA_NOISE_SPAN Hz ทั้งสองข้าง
        และไม่อยู่ใกล้เป้าหมาย/ฮาร์มอนิกอื่น) — ใช้ได้ทั้ง PSD และ FFT magnitude²"""
        pm = np.abs(freqs - f0) <= PSDA_PEAK_HALF_BW
        if not np.any(pm):
            return 1.0
        d = np.abs(freqs - f0)
        nb = ((d >= PSDA_NOISE_GAP) & (d <= PSDA_NOISE_SPAN) & ~excl
              & (freqs >= 2.0) & (freqs <= FILTER_HIGH - 1.0))
        if np.sum(nb) < 6:
            return 1.0
        return float(np.mean(spec[pm]) / (np.mean(spec[nb]) + 1e-18))

    @staticmethod
    def _robust_stats(score_list):
        mu, sd = {}, {}
        for k in SSVEP_CONFIG:
            v = np.log(np.maximum(np.array([s[k] for s in score_list], dtype=float), 1e-3))
            m = float(np.median(v))
            mad = float(np.median(np.abs(v - m))) * 1.4826
            mu[k], sd[k] = m, max(mad, float(np.std(v)) * 0.5, PSDA_SD_FLOOR)
        return mu, sd

    def calibrate_null(self, score_list):
        """รับ list ของ dict คะแนนจาก epoch ตอน "ไม่ได้มอง" (baseline) → เก็บ median และ SD แบบ robust (MAD)
        ถ้า epoch น้อยกว่า 8 ถือว่าไม่พอ คืน False"""
        self.reset_null()
        if len(score_list) < 8:
            return False
        self.null_mu, self.null_sd = self._robust_stats(score_list)
        return True

    def calibrate_null_from_signal(self, proc, samples):
        """สร้าง null จากสัญญาณที่กรองแล้วช่วง baseline: ตัดเป็น epoch ยาว EPOCH_SEC เลื่อนทีละ
        PSDA_NULL_STEP_SEC ข้าม epoch ที่มี spike แล้วเรียก calibrate_null (ต้องเรียกหลัง finalize_baseline)"""
        x = np.asarray(samples, dtype=float)
        n = int(EPOCH_SEC * self.fs)
        step = max(1, int(PSDA_NULL_STEP_SEC * self.fs))
        sc_list = []
        for i in range(0, len(x) - n + 1, step):
            blk = x[i:i + n]
            xb = blk - np.mean(blk)
            rms = float(np.sqrt(np.mean(xb * xb))) + 1e-12
            if float(np.max(np.abs(xb))) / rms > ARTIFACT_PEAK_X:
                continue
            f, p = proc.compute_psd(blk)
            if len(p) == 0:
                continue
            sc_list.append(self.compute_scores(f, p, smooth=False))
        return self.calibrate_null(sc_list)

    def reset_null(self):
        self.null_mu = self.null_sd = None
        self._confirm_key, self._confirm_n = None, 0

    def zscores(self, scores: dict):
        mu = getattr(self, "null_mu", None)
        sd = getattr(self, "null_sd", None)
        if not mu:
            return None
        return {k: (float(np.log(max(scores[k], 1e-3))) - mu[k]) / sd[k] for k in scores}

    def compute_scores(self, freqs: np.ndarray, psd: np.ndarray, smooth: bool = True):
        """Return {freq_key: score}. score = ค่าเฉลี่ยถ่วงน้ำหนักของ SNR ที่ f, 2f, ...
        โดย SNR = กำลังในแถบ ÷ local noise ของ (PSD ÷ baseline)  (≈1 เมื่อไม่มีสัญญาณ)
        หารด้วยผลรวมน้ำหนักที่ใช้ เพื่อให้เป้าหมายที่มี harmonic ต่างจำนวนเทียบกันได้"""
        if len(psd) == 0:
            return {k: 0.0 for k in SSVEP_CONFIG}

        if self.baseline_psd is not None and len(self.baseline_psd) == len(psd):
            norm_psd = psd / self.baseline_psd
        else:
            norm_psd = psd.copy()
            if self.baseline_psd is not None and not self._warned_size:
                self._warned_size = True
                print("[WARN] baseline กับ epoch ขนาด PSD ไม่ตรงกัน → ข้ามการหาร baseline "
                      "(ตรวจ EPOCH_SEC ให้เท่ากับ ANALYSIS_SEC)")

        excl = self._target_mask(freqs, LOCAL_NOISE_EXCL)
        raw_scores = {}
        for key, cfg in SSVEP_CONFIG.items():
            num = 0.0
            wsum = 0.0
            for fhz, w in zip(cfg["freqs"], cfg["weights"]):
                if fhz > freqs[-1] or fhz > FILTER_HIGH:
                    continue
                if PSDA_MODE:
                    snr = self._psda_snr(freqs, norm_psd, fhz, excl)
                else:
                    p = self._band_power(freqs, norm_psd, fhz)
                    noise = self._local_noise(freqs, norm_psd, fhz, excl)
                    snr = p / noise
                if not self._peak_aligned(freqs, norm_psd, fhz):
                    snr = min(snr, 1.0)       # พีคไม่ตรงเป้าหมาย → ไม่นับว่ามีสัญญาณ
                num += w * snr
                wsum += w
            raw_scores[key] = num / wsum if wsum > 0 else 0.0

        if not smooth:
            return dict(raw_scores)
        for k in raw_scores:
            self._scores_ema[k] = (DETECTION_SMOOTH * raw_scores[k]
                                   + (1 - DETECTION_SMOOTH) * self._scores_ema[k])
        return dict(self._scores_ema)

    def _detect_psda(self, scores: dict):
        """ตัดสินแบบ PSDA: z-score เทียบ null → ผู้ชนะ + ช่องว่างจากอันดับสอง + FFT ยืนยัน + ยืนยันซ้ำ
        คืน (key, z) หรือ (None, 0) ; เหตุผลล่าสุดอยู่ใน self.last_reason (ไว้พิมพ์ log)"""
        z = self.zscores(scores)
        self.last_z = z
        if z is None:
            self.last_reason = "ยังไม่มี null calibration (ทำ baseline ใหม่)"
            return None, 0.0
        items = sorted(z.items(), key=lambda x: x[1], reverse=True)
        best, zb = items[0]
        z2 = items[1][1] if len(items) > 1 else -9.0
        self.last_best = (best, zb, z2)
        reason = None
        if zb < PSDA_Z_THR:
            reason = f"z {zb:.1f} < {PSDA_Z_THR}"
        elif zb - z2 < PSDA_Z_GAP:
            reason = f"นำอันดับสองไม่พอ ({zb:.1f} vs {z2:.1f})"
        if reason is not None:
            self._confirm_key, self._confirm_n = None, 0
            self.last_reason = reason
            return None, 0.0
        if zb >= PSDA_Z_STRONG:
            self._confirm_key, self._confirm_n = None, 0
            self.last_reason = f"ชัดเจน z={zb:.1f} → สั่งทันที"
            return best, zb
        if best == getattr(self, "_confirm_key", None):
            self._confirm_n += 1
        else:
            self._confirm_key, self._confirm_n = best, 1
        if self._confirm_n >= PSDA_CONFIRM:
            self._confirm_key, self._confirm_n = None, 0
            self.last_reason = f"ยืนยันซ้ำ {PSDA_CONFIRM} รอบ z={zb:.1f}"
            return best, zb
        self.last_reason = f"รอยืนยัน ({self._confirm_n}/{PSDA_CONFIRM}) z={zb:.1f}"
        return None, 0.0

    def detect(self, scores: dict):
        """Return (candidate_key, score) หรือ (None, 0)
        ผ่านเมื่อ: score >= DETECTION_SNR_THR และเหนือ runner-up ตาม DETECTION_MARGIN
        (ผู้สมัครนี้ยังต้องผ่าน DecisionFilter ก่อนจึงสั่งรถ)"""
        if not scores:
            return None, 0.0
        if PSDA_MODE:
            return self._detect_psda(scores)
        items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_key, best_val = items[0]
        if PEAK_MODE and PEAK_ADAPTIVE:
            if PEAK_NORM_HIST > 0:
                hist = getattr(self, "_norm_hist", None)
                if hist is None:
                    hist = self._norm_hist = {k: [] for k in scores}
                raw = dict(scores)
                scores = {}
                for k, v in raw.items():
                    h = hist.setdefault(k, [])
                    base = float(np.median(h)) if len(h) >= PEAK_NORM_MIN_N else 1.0
                    scores[k] = v / max(base, 1.0)
                    h.append(v)
                    if len(h) > PEAK_NORM_HIST:
                        h.pop(0)
                items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                best_key, best_val = items[0]
            others = [v for _, v in items[1:]]
            floor = max(float(np.median(others)), 1e-6) if others else 1.0
            ratio = best_val / floor
            self.last_ratio = ratio
            self.last_norm_best = (best_key, best_val)
            prev = getattr(self, "_prev_best", None)
            self._prev_best = best_key
            if best_val >= PEAK_STRONG_ABS and ratio >= PEAK_RATIO_STRONG:
                return best_key, best_val                      # เด่นมาก → สั่งทันที
            if best_val >= PEAK_WEAK_ABS and ratio >= PEAK_RATIO_WEAK and prev == best_key:
                return best_key, best_val                      # เด่นพอ + เป็นตัวเดิม 2 รอบติด → สั่ง
            return None, 0.0
        if PEAK_MODE:
            # v4.3: เลือกความถี่ที่แรงที่สุดเสมอ (ไม่ต้องผ่าน threshold/margin)
            if best_val <= 0 or best_val < PEAK_MIN_SCORE:
                return None, 0.0
            second = items[1][1] if len(items) > 1 else 0.0
            if second > 0 and best_val < second * PEAK_MARGIN:
                return None, 0.0
            return best_key, best_val
        runner = items[1][1] if len(items) > 1 else 0.0
        if best_val < DETECTION_SNR_THR:
            return None, 0.0
        if runner > 0 and best_val < runner * DETECTION_MARGIN:
            return None, 0.0
        return best_key, best_val


class DecisionFilter:
    """เฉลี่ยคะแนนของ DETECTION_AVG_EPOCHS epoch ล่าสุด แล้วให้ detector ตัดสิน (v4.2)
    การเฉลี่ยลดความแปรปรวนของคะแนน → ตั้ง threshold ต่ำได้โดยไม่เพิ่ม false trigger"""

    def __init__(self, n_avg=None):
        self.n_avg = DETECTION_AVG_EPOCHS if n_avg is None else n_avg
        self._hist = deque(maxlen=max(1, self.n_avg))

    def reset(self):
        self._hist.clear()

    def update(self, scores: dict, detector):
        """รับคะแนนของ epoch ล่าสุด คืน (key|None, score, avg_scores)"""
        self._hist.append(dict(scores))
        avg = {k: float(np.mean([h[k] for h in self._hist])) for k in scores}
        key, sc = detector.detect(avg)
        return key, sc, avg


# ╔════════════════════════════════════════════════════════════════════╗
# ║                  DATA SOURCE (UDP / BrainFlow / Demo)             ║
# ╚════════════════════════════════════════════════════════════════════╝

class DataSource(QObject):
    """Thread-safe data producer.  Emits nothing; main thread polls."""

    def __init__(self, mode="udp"):
        super().__init__()
        self.mode = mode
        self._buf = deque(maxlen=BUFFER_MAX)
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._board = None
        self._total_samples = 0
        self.status_text = "Initializing..."
        self.connected = False

    # ── public API ──────────────────────────────────────────────────
    def start(self):
        self._running = True
        if self.mode == "brainflow":
            self._thread = threading.Thread(target=self._run_brainflow, daemon=True)
        elif self.mode == "demo":
            self._thread = threading.Thread(target=self._run_demo, daemon=True)
        else:
            self._thread = threading.Thread(target=self._run_udp, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._board and HAS_BRAINFLOW:
            try:
                self._board.stop_stream()
                self._board.release_session()
            except Exception:
                pass
        if hasattr(self, "_sock") and self._sock:
            try: self._sock.close()
            except Exception: pass

    def pull_new(self) -> np.ndarray:
        with self._lock:
            out = np.array(list(self._buf), dtype=np.float64)
            self._buf.clear()
        return out

    @property
    def total_samples(self):
        return self._total_samples

    # ── UDP loop ────────────────────────────────────────────────────
    def _run_udp(self):
        self.status_text = f"UDP {UDP_IP}:{UDP_PORT} — waiting..."
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._sock.bind((UDP_IP, UDP_PORT))
        except OSError as e:
            self.status_text = f"UDP bind error: {e}"
            return
        self._sock.settimeout(1.0)
        self.connected = True
        self.status_text = f"UDP {UDP_IP}:{UDP_PORT} — LISTENING"
        while self._running:
            try:
                data, _ = self._sock.recvfrom(1024)
                val = float(data.decode("utf-8").strip())
                if np.isfinite(val):
                    with self._lock:
                        self._buf.append(val)
                    self._total_samples += 1
            except socket.timeout:
                continue
            except ValueError:
                continue
            except Exception:
                if self._running:
                    continue

    # ── BrainFlow loop ──────────────────────────────────────────────
    def _run_brainflow(self):
        if not HAS_BRAINFLOW:
            self.status_text = "BrainFlow not installed"
            return
        self.status_text = f"Preparing session on {SERIAL_PORT}..."
        try:
            BoardShim.enable_dev_board_logger()
            params = BrainFlowInputParams()
            params.serial_port = SERIAL_PORT
            if BOARD_NAME:
                params.other_info = BOARD_NAME
            self._board = BoardShim(BOARD_ID, params)
            self._board.prepare_session()
            self._board.start_stream()
            self.connected = True
            self.status_text = f"{SERIAL_PORT} (BrainFlow) — CONNECTED"
        except Exception as e:
            self.status_text = f"{SERIAL_PORT} — ERROR: {e}"
            self.connected = False
            return

        # Also save to data.txt for logging
        data_file = None
        try:
            data_file = open("data.txt", "w")
        except Exception:
            pass

        while self._running:
            try:
                data = self._board.get_board_data()
                if data.shape[1] > 0:
                    ch = data[SELECTED_CH]
                    vals = ch * SCALE_FACTOR
                    with self._lock:
                        self._buf.extend(vals.tolist())
                    self._total_samples += len(vals)
                    # Log to file
                    if data_file:
                        for v in vals:
                            data_file.write(f"{v:.3f}\n")
            except Exception:
                pass
            time.sleep(0.01)

        if data_file:
            data_file.close()

    # ── Demo loop (synthetic SSVEP) ─────────────────────────────────
    def _run_demo(self):
        self.connected = True
        self.status_text = "DEMO — Synthetic SSVEP"
        t = 0.0
        dt = 1.0 / SAMPLE_RATE

        # Cycle through different SSVEP frequencies for demo
        demo_freqs = [0.0, 9.0, 0.0, 15.0, 0.0]  # 0 = no SSVEP
        demo_idx = 0
        cycle_time = 0.0
        cycle_duration = 10.0  # allow complete 5-second decision epochs

        while self._running:
            # Switch target frequency periodically
            if cycle_time >= cycle_duration:
                cycle_time = 0.0
                demo_idx = (demo_idx + 1) % len(demo_freqs)

            target_f = 0.0 if t < 40.0 else demo_freqs[demo_idx]  # initial 40 s neutral for baseline
            batch = []
            for _ in range(int(SAMPLE_RATE * 0.05)):   # 50 ms batches
                val = 0.0
                if target_f > 0:
                    # Fundamental + harmonics (match SSVEP_CONFIG)
                    cfg_key = int(target_f)
                    if cfg_key in SSVEP_CONFIG:
                        freqs_list = SSVEP_CONFIG[cfg_key]["freqs"]
                        amps = [5.0, 2.5, 1.2]  # decreasing amplitudes
                        for f, a in zip(freqs_list, amps):
                            if f <= FILTER_HIGH:
                                val += a * np.sin(2 * np.pi * f * t)
                    else:
                        val += 5.0 * np.sin(2 * np.pi * target_f * t)

                # Add realistic noise components
                val += (1.0  * np.sin(2 * np.pi * 10.0 * t)      # alpha rhythm
                      + 0.4  * np.sin(2 * np.pi * 50.0 * t)      # line noise (filtered out)
                      + 0.2  * np.sin(2 * np.pi * 0.3  * t)      # drift (filtered out)
                      + np.random.randn() * 1.5)                  # gaussian noise

                batch.append(val)
                t += dt
                cycle_time += dt

            with self._lock:
                self._buf.extend(batch)
            self._total_samples += len(batch)
            time.sleep(0.05)


# ╔════════════════════════════════════════════════════════════════════╗
# ║                  CUSTOM WIDGETS                                   ║
# ╚════════════════════════════════════════════════════════════════════╝

class CircleIndicator(QWidget):
    """Colored circle indicator (connection / detection)."""
    def __init__(self, color="#2ea043", radius=8, parent=None):
        super().__init__(parent)
        self._color = QColor(color)
        self._r = radius
        self.setFixedSize(radius * 2 + 2, radius * 2 + 2)

    def set_color(self, color: str):
        self._color = QColor(color)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QBrush(self._color))
        p.setPen(QPen(self._color.darker(130), 1))
        p.drawEllipse(1, 1, self._r * 2, self._r * 2)
        p.end()


class FrequencyBar(QWidget):
    """Horizontal score bar for one SSVEP frequency."""
    def __init__(self, label, color, parent=None):
        super().__init__(parent)
        self.setFixedHeight(22)
        self._label = label
        self._color = QColor(color)
        self._value = 0.0
        self._max   = 10.0

    def set_value(self, v, mx=10.0):
        self._value = max(0, v)
        self._max = max(mx, 0.01)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # circle
        cr = h // 2 - 2
        p.setBrush(QBrush(self._color))
        p.setPen(Qt.NoPen)
        p.drawEllipse(2, 2, cr * 2, cr * 2)

        # label
        p.setPen(QColor(CLR_TEXT))
        p.setFont(QFont("Consolas", 9))
        p.drawText(cr * 2 + 8, h - 5, self._label)

        # bar background
        bar_x = 120
        bar_w = w - bar_x - 45
        p.setBrush(QBrush(QColor("#1c2333")))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(bar_x, 3, bar_w, h - 6, 3, 3)

        # bar fill
        frac = min(self._value / self._max, 1.0)
        p.setBrush(QBrush(self._color))
        p.drawRoundedRect(bar_x, 3, int(bar_w * frac), h - 6, 3, 3)

        # value text
        p.setPen(QColor(CLR_TEXT))
        p.drawText(w - 42, h - 5, f"{self._value:.2f}")
        p.end()


class DirectionPad(QWidget):
    """WASD directional pad with stop light — integrated keyboard visualizer."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(220, 140)
        self._active_key = None  # "A", "D", "STOP", or None
        self._btn_rects = {}

    def set_active(self, key_name: str, color: str = "#4CAF50"):
        """key_name: 'W', 'A', 'S', 'D', 'STOP', or None
        color: สีของปุ่มที่ active (v4.3: ใช้สีของความถี่ที่ตรวจเจอ)"""
        self._active_key = key_name
        self._active_color = color
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        btn_size = 50

        # Center coordinates
        cx = self.width() // 2
        cy = self.height() // 2

        # Button positions: (x, y, label, key_name)
        buttons = [
            (cx - 92, cy - btn_size // 2, "◄ A", "A"),
            (cx + 42, cy - btn_size // 2, "D ►", "D"),
        ]

        for bx, by, label, key_name in buttons:
            # Active = green, Inactive = dark panel
            if self._active_key == key_name:
                p.setBrush(QBrush(QColor(getattr(self, "_active_color", "#4CAF50"))))
                text_color = QColor("white")
            else:
                p.setBrush(QBrush(QColor(CLR_PANEL)))
                text_color = QColor(CLR_DIM)

            p.setPen(QPen(QColor(CLR_BORDER), 1))
            p.drawRoundedRect(bx, by, btn_size, btn_size, 5, 5)

            p.setPen(text_color)
            p.setFont(QFont("Consolas", 10, QFont.Bold))
            p.drawText(bx, by, btn_size, btn_size, Qt.AlignCenter, label)

        # Center stop light
        light_r = 18
        if self._active_key == "STOP" or self._active_key is None:
            light_color = QColor("#FF0000")
            light_outline = QColor("#8B0000")
        else:
            light_color = QColor("#3d3d3d")
            light_outline = QColor("#222222")

        p.setBrush(QBrush(light_color))
        p.setPen(QPen(light_outline, 2))
        p.drawEllipse(cx - light_r, cy - light_r, light_r * 2, light_r * 2)

        # STOP label under the light
        if self._active_key == "STOP" or self._active_key is None:
            p.setPen(QColor(CLR_RED))
            p.setFont(QFont("Consolas", 7, QFont.Bold))
            p.drawText(cx - 15, cy + light_r + 2, 30, 12, Qt.AlignCenter, "STOP")

        p.end()


# ╔════════════════════════════════════════════════════════════════════╗
# ║                  MAIN WINDOW                                      ║
# ╚════════════════════════════════════════════════════════════════════╝

class MainWindow(QMainWindow):

    def __init__(self, source: DataSource, enable_keyboard=True):
        super().__init__()
        self.source = source
        self.proc = SignalProcessor()
        self.detector = SSVEPDetector()
        self.keyboard = KeyboardController() if enable_keyboard else None
        self.decider = DecisionFilter()
        self.gate = CommandGate(COMMAND_HOLD_SEC, COMMAND_LOCKOUT_SEC,
                                0.0, 0.0,   # ตัดสินใจทีละช่วง epoch จึงไม่ต้องยืนยันซ้ำ
                                LOCKOUT_AFTER_STOP_SEC)
        self._epoch_n      = int(EPOCH_SEC * SAMPLE_RATE)
        self._step_n       = int(DECISION_STEP_SEC * SAMPLE_RATE)
        self._since_step   = 0
        self._epoch_buf    = []
        self._last_scores  = {k: 0.0 for k in SSVEP_CONFIG}
        self._last_det_key = None
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self._emergency_stop)
        self._action_key   = {c["action"]: c["key_name"] for c in SSVEP_CONFIG.values()}
        self._action_label = {c["action"]: c["label"]    for c in SSVEP_CONFIG.values()}
        self._last_data_time = time.monotonic()

        # ── data buffers ──
        self._raw_buf  = deque(maxlen=BUFFER_MAX)
        self._filt_buf = deque(maxlen=BUFFER_MAX)

        # ── [BUG-3 FIX] Baseline State Machine (IDLE / RECORDING / READY) ──
        self._bl_state   = BaselineState.IDLE
        self._bl_start   = 0.0   # timestamp when recording began
        self._bl_frames  = 0     # number of PSD frames accumulated this session
        self._bl_samples = []    # v4.6: สัญญาณที่กรองแล้วช่วงเก็บ baseline

        # ── loop counter ──
        self._loop_count = 0

        # ── last detection for action log ──
        self._last_action = None
        self._action_log = deque(maxlen=10)

        self._build_ui()
        self._start_timer()

    # ================================================================
    #  UI CONSTRUCTION
    # ================================================================
    def _build_ui(self):
        self.setWindowTitle("LightBrain — SSVEP 9 Hz Left / 15 Hz Right")
        self.resize(1280, 720)
        self.setStyleSheet(f"""
            QMainWindow {{ background: {CLR_BG}; }}
            QWidget     {{ background: {CLR_BG}; color: {CLR_TEXT}; }}
            QLabel      {{ color: {CLR_TEXT}; }}
            QPushButton {{
                background: {CLR_PANEL}; color: {CLR_TEXT};
                border: 1px solid {CLR_BORDER}; border-radius: 4px;
                padding: 6px 14px; font-family: Consolas; font-size: 11px;
            }}
            QPushButton:hover  {{ background: {CLR_BORDER}; }}
            QPushButton:disabled {{ color: {CLR_DIM}; border-color: {CLR_DIM}; }}
            QProgressBar {{
                background: #1c2333; border: 1px solid {CLR_BORDER};
                border-radius: 3px; text-align: center; color: {CLR_TEXT};
                font-family: Consolas; font-size: 10px;
            }}
            QProgressBar::chunk {{ background: {CLR_BORDER}; border-radius: 2px; }}
        """)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 4, 6, 4)
        root.setSpacing(4)

        # ── header ──
        hdr = QHBoxLayout()
        title = QLabel("  ◉  LightBrain SSVEP | 9 Hz LEFT · 15 Hz RIGHT")
        title.setFont(QFont("Consolas", 13, QFont.Bold))
        title.setStyleSheet(f"color: {CLR_ACCENT};")
        hdr.addWidget(title)
        hdr.addStretch()
        self.lbl_status_hdr = QLabel("● Initializing...")
        self.lbl_status_hdr.setFont(QFont("Consolas", 10))
        self.lbl_status_hdr.setStyleSheet(f"color: {CLR_ACCENT};")
        hdr.addWidget(self.lbl_status_hdr)
        root.addLayout(hdr)

        # ── body: graphs (left) + panel (right) ──
        body = QHBoxLayout()
        body.setSpacing(6)

        # ── graphs 2 × 2 ──
        graph_grid = QGridLayout()
        graph_grid.setSpacing(4)
        pg_cfg = {"background": "#0a0e14", "foreground": CLR_DIM}

        self.plt_psd = pg.PlotWidget(title="Power Spectral Density (PSD)", **pg_cfg)
        self.plt_fft = pg.PlotWidget(title="FFT Magnitude", **pg_cfg)
        self.plt_raw = pg.PlotWidget(title="Raw Data", **pg_cfg)
        self.plt_fil = pg.PlotWidget(title="Filtered Signal", **pg_cfg)

        for pw, xlabel, ylabel in [
            (self.plt_psd, "Frequency (Hz)", "Power (µV²/Hz)"),
            (self.plt_fft, "Frequency (Hz)", "Magnitude (µV)"),
            (self.plt_raw, "Time (s)",       "Amplitude (µV)"),
            (self.plt_fil, "Time (s)",       "Amplitude (µV)"),
        ]:
            pw.setLabel("bottom", xlabel, **{"font-size": "9pt"})
            pw.setLabel("left", ylabel,   **{"font-size": "9pt"})
            pw.showGrid(x=True, y=True, alpha=0.15)
            pw.getPlotItem().getViewBox().setBorder({"color": CLR_BORDER, "width": 1})
            pw.setMinimumSize(280, 180)

        self.curve_psd = self.plt_psd.plot(pen=pg.mkPen(CLR_ACCENT, width=1.5))
        self.curve_fft = self.plt_fft.plot(pen=pg.mkPen(CLR_ACCENT, width=1.2))
        self.curve_raw = self.plt_raw.plot(pen=pg.mkPen("#6e8efb", width=0.8))
        self.curve_fil = self.plt_fil.plot(pen=pg.mkPen("#2dde98", width=0.8))

        # vertical lines on PSD & FFT for SSVEP fundamentals + harmonics
        for cfg in SSVEP_CONFIG.values():
            for i, freq in enumerate(cfg["freqs"]):
                if freq > FILTER_HIGH:
                    continue
                style = Qt.SolidLine if i == 0 else Qt.DashLine
                width = 1.0 if i == 0 else 0.5
                for pw in (self.plt_psd, self.plt_fft):
                    line = pg.InfiniteLine(pos=freq, angle=90,
                                           pen=pg.mkPen(cfg["color"], width=width,
                                                        style=style))
                    pw.addItem(line)

        graph_grid.addWidget(self.plt_psd, 0, 0)
        graph_grid.addWidget(self.plt_fft, 0, 1)
        graph_grid.addWidget(self.plt_raw, 1, 0)
        graph_grid.addWidget(self.plt_fil, 1, 1)
        body.addLayout(graph_grid, stretch=7)

        # ── right panel ──
        panel = QVBoxLayout()
        panel.setSpacing(6)
        panel_frame = QFrame()
        panel_frame.setStyleSheet(
            f"QFrame {{ background: {CLR_PANEL}; border: 1px solid {CLR_BORDER};"
            f" border-radius: 5px; }}")
        panel_inner = QVBoxLayout(panel_frame)
        panel_inner.setContentsMargins(10, 8, 10, 8)
        panel_inner.setSpacing(6)

        # ── BASELINE ACQUISITION ──
        panel_inner.addWidget(self._section_label("BASELINE ACQUISITION"))
        # Start button row
        bl_btn_row = QHBoxLayout()
        self.btn_baseline = QPushButton("▶  Record Baseline  (30s)")
        self.btn_baseline.clicked.connect(self._on_baseline_click)
        bl_btn_row.addWidget(self.btn_baseline)
        # [BUG-3 FIX] Clear button
        self.btn_bl_clear = QPushButton("✕ Clear")
        self.btn_bl_clear.setFixedWidth(70)
        self.btn_bl_clear.setEnabled(False)
        self.btn_bl_clear.clicked.connect(self._on_baseline_clear)
        bl_btn_row.addWidget(self.btn_bl_clear)
        panel_inner.addLayout(bl_btn_row)
        self.lbl_bl_status = QLabel("Not recorded  — press ▶ to begin")
        self.lbl_bl_status.setFont(QFont("Consolas", 9))
        self.lbl_bl_status.setStyleSheet(f"color: {CLR_DIM};")
        panel_inner.addWidget(self.lbl_bl_status)
        self.bar_bl = QProgressBar()
        self.bar_bl.setRange(0, 100)
        self.bar_bl.setValue(0)
        self.bar_bl.setFixedHeight(14)
        panel_inner.addWidget(self.bar_bl)
        # [BUG-4 FIX] Warm-up status indicator
        self.lbl_warmup = QLabel("⚠ Filter warming up...")
        self.lbl_warmup.setFont(QFont("Consolas", 8))
        self.lbl_warmup.setStyleSheet(f"color: {CLR_DIM};")
        panel_inner.addWidget(self.lbl_warmup)

        # ── SIGNAL METRICS ──
        panel_inner.addWidget(self._section_label("SIGNAL METRICS"))
        met = QGridLayout()
        met.setSpacing(2)
        for r, (k, v) in enumerate([("PSD MAX", "0"), ("MAX VALUE", "0"),
                                      ("PEAK FREQ", "0"), ("LOOP COUNT", "0")]):
            lk = QLabel(k)
            lk.setFont(QFont("Consolas", 8))
            lk.setStyleSheet(f"color: {CLR_DIM};")
            lv = QLabel(v)
            lv.setFont(QFont("Consolas", 11, QFont.Bold))
            met.addWidget(lk, r // 2, (r % 2) * 2)
            met.addWidget(lv, r // 2, (r % 2) * 2 + 1)
        panel_inner.addLayout(met)
        # keep references
        self._met_labels = [met.itemAtPosition(r // 2, (r % 2) * 2 + 1).widget()
                            for r in range(4)]

        # ── FREQUENCY INDEX SCORES ──
        panel_inner.addWidget(self._section_label("FREQUENCY INDEX SCORES"))
        self.freq_bars = {}
        for key, cfg in SSVEP_CONFIG.items():
            fb = FrequencyBar(cfg["label"], cfg["color"])
            panel_inner.addWidget(fb)
            self.freq_bars[key] = fb
        # "No Signal" bar
        self.bar_nosignal = FrequencyBar("No Signal", CLR_NOSIGNAL)
        panel_inner.addWidget(self.bar_nosignal)

        # ── WHEELCHAIR CONTROL ──
        panel_inner.addWidget(self._section_label("TWO-COMMAND CONTROL | SPACE = STOP"))

        # Direction pad (replaces KeyboardVisualizer)
        ctrl_row = QHBoxLayout()
        self.direction_pad = DirectionPad()
        ctrl_row.addStretch()
        ctrl_row.addWidget(self.direction_pad)
        ctrl_row.addStretch()
        panel_inner.addLayout(ctrl_row)

        # Action label
        self.lbl_action = QLabel("Status: IDLE (STOPPED)")
        self.lbl_action.setFont(QFont("Consolas", 10, QFont.Bold))
        self.lbl_action.setStyleSheet(f"color: {CLR_RED};")
        self.lbl_action.setAlignment(Qt.AlignCenter)
        panel_inner.addWidget(self.lbl_action)

        # ── DETECTED INDEX ──
        panel_inner.addWidget(self._section_label("DETECTED INDEX"))
        det_row = QHBoxLayout()
        self.lbl_det_val = QLabel("—")
        self.lbl_det_val.setFont(QFont("Consolas", 28, QFont.Bold))
        self.lbl_det_val.setStyleSheet(f"color: {CLR_ACCENT};")
        det_row.addWidget(self.lbl_det_val)
        det_row.addStretch()
        det_sub = QVBoxLayout()
        self.det_circle = CircleIndicator(CLR_NOSIGNAL, 18)
        det_sub.addWidget(self.det_circle, alignment=Qt.AlignCenter)
        self.lbl_det_name = QLabel("No Signal")
        self.lbl_det_name.setFont(QFont("Consolas", 9))
        self.lbl_det_name.setAlignment(Qt.AlignCenter)
        det_sub.addWidget(self.lbl_det_name)
        det_row.addLayout(det_sub)
        panel_inner.addLayout(det_row)

        # ── CONNECTION ──
        panel_inner.addWidget(self._section_label("CONNECTION"))
        conn_row = QHBoxLayout()
        self.conn_circle = CircleIndicator(CLR_RED, 7)
        conn_row.addWidget(self.conn_circle)
        self.lbl_conn = QLabel(f"{SERIAL_PORT} — DISCONNECTED")
        self.lbl_conn.setFont(QFont("Consolas", 9))
        conn_row.addWidget(self.lbl_conn)
        conn_row.addStretch()
        self.lbl_samples = QLabel("Samples: 0")
        self.lbl_samples.setFont(QFont("Consolas", 9))
        self.lbl_samples.setStyleSheet(f"color: {CLR_DIM};")
        conn_row.addWidget(self.lbl_samples)
        panel_inner.addLayout(conn_row)

        panel_inner.addStretch()
        panel.addWidget(panel_frame)
        body.addLayout(panel, stretch=3)

        root.addLayout(body)

    # ── helper ──
    @staticmethod
    def _section_label(text):
        lbl = QLabel(text)
        lbl.setFont(QFont("Consolas", 9, QFont.Bold))
        lbl.setStyleSheet(
            f"color: {CLR_ACCENT}; border-bottom: 1px solid {CLR_BORDER};"
            f" padding-bottom: 2px; margin-top: 4px;"
            f" background: transparent; border-top: none;"
            f" border-left: none; border-right: none;")
        return lbl

    # ================================================================
    #  TIMER & UPDATE
    # ================================================================
    def _start_timer(self):
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(UPDATE_MS)
        self.source.start()

    def _tick(self):
        """Main update tick — pull data → filter → detect → control → update UI.

        [BUG-3 FIX] Uses BaselineState machine instead of boolean flag.
        [BUG-4 FIX] Skips baseline accumulation until filter is warmed up.
        [BUG-5 FIX] Uses ANALYSIS_SEC (5s) window for PSD instead of 3s.
        """
        self._loop_count += 1

        # ── [BUG-4 FIX] Update warm-up indicator ──
        if self.proc.is_warmed_up:
            self.lbl_warmup.setText("✔ Filter ready")
            self.lbl_warmup.setStyleSheet(f"color: {CLR_GREEN};")
        else:
            self.lbl_warmup.setText("⚠ Filter warming up...")
            self.lbl_warmup.setStyleSheet(f"color: {CLR_DIM};")

        # ── pull new samples ──
        new = self.source.pull_new()
        if len(new) == 0:
            # ไม่มีข้อมูล EEG เข้ามานาน → หยุดทันที ไม่ปล่อยให้คำสั่งค้างจนครบเวลา
            if time.monotonic() - self._last_data_time > 1.0:
                self.gate.force_stop()
                if self.keyboard:
                    self.keyboard.release_all()
                self.direction_pad.set_active("STOP")
                self.lbl_action.setText("Status: NO DATA (STOPPED)")
                self.lbl_action.setStyleSheet(f"color: {CLR_RED};")
            self._update_connection()
            return
        self._last_data_time = time.monotonic()

        # ── filter samples (block mode for efficiency) ──
        filtered = self.proc.filter_block(new)
        self._raw_buf.extend(new.tolist())
        self._filt_buf.extend(filtered.tolist())

        # ── convert to arrays ──
        raw_arr  = np.array(self._raw_buf)
        filt_arr = np.array(self._filt_buf)

        # ── time domain graphs ──
        n_disp = min(len(raw_arr), DISPLAY_SAMPLES)
        t_ax = np.arange(n_disp) / SAMPLE_RATE
        self.curve_raw.setData(t_ax, raw_arr[-n_disp:])
        self.curve_fil.setData(t_ax, filt_arr[-n_disp:])

        # ── [BUG-5 FIX] frequency domain: use ANALYSIS_SEC (5s) window ──
        analysis_n = min(len(filt_arr), SAMPLE_RATE * ANALYSIS_SEC)
        analysis = filt_arr[-analysis_n:]

        f_psd, psd = self.proc.compute_psd(analysis)
        f_fft, mag = self.proc.compute_fft(analysis)

        # mask to 0-55 Hz for display (show slightly above Notch for verification)
        if len(f_psd) > 0:
            mask_p = f_psd <= 55
            self.curve_psd.setData(f_psd[mask_p], psd[mask_p])
        if len(f_fft) > 0:
            mask_f = f_fft <= 55
            self.curve_fft.setData(f_fft[mask_f], mag[mask_f])

        # ── [BUG-2 & BUG-3 FIX] Baseline State Machine ──
        if self._bl_state == BaselineState.RECORDING:
            elapsed = time.time() - self._bl_start

            # [BUG-4 FIX] Only accumulate when filter is warmed up
            if self.proc.is_warmed_up and len(psd) > 0:
                if analysis_n >= SAMPLE_RATE * ANALYSIS_SEC:      # v4.6: เฉพาะหน้าต่างเต็ม 5 s (ขนาด PSD เท่ากับ epoch ตอนตัดสิน)
                    self.detector.accumulate_baseline(f_psd, psd)
                    self._bl_frames += 1
                self._bl_samples.extend(filtered.tolist())         # v4.6: ใช้สร้าง null (ตอนไม่ได้มอง)

            pct = min(100, int(elapsed / BASELINE_DURATION * 100))
            self.bar_bl.setValue(pct)
            self.lbl_bl_status.setText(
                f"Recording... {int(elapsed)}/{BASELINE_DURATION}s  "
                f"({self._bl_frames} frames)"
            )

            if elapsed >= BASELINE_DURATION:
                self._finish_baseline()

        # ── detection ──
        # ── วิเคราะห์ความถี่: ใช้ข้อมูล EPOCH_SEC ล่าสุด ตัดสินใจทุก DECISION_STEP_SEC (เลือกความถี่ที่พีคแรงที่สุด) ──
        recording = self._bl_state == BaselineState.RECORDING
        analysis_on = (not recording) and (
            not REQUIRE_BASELINE or self.detector.baseline_psd is not None
        )  # PSDA requires baseline/null calibration even in demo
        epoch_action = None
        if not analysis_on:
            # ก่อนเก็บ baseline และระหว่างเก็บ baseline: ไม่ตรวจจับ แสดง No Signal ห้ามสั่งรถ
            self._epoch_buf = []
            self._since_step = 0
            self.decider.reset()
            self.detector._confirm_key, self.detector._confirm_n = None, 0
            self._last_scores = {k: 0.0 for k in SSVEP_CONFIG}
            self._last_det_key = None
            self.gate.force_stop()
        else:
            self._epoch_buf.extend(filtered.tolist())
            self._since_step += len(filtered)
            if len(self._epoch_buf) > self._epoch_n:          # เก็บเฉพาะ EPOCH_SEC ล่าสุด
                del self._epoch_buf[:len(self._epoch_buf) - self._epoch_n]
            if len(self._epoch_buf) >= self._epoch_n and self._since_step >= self._step_n:
                self._since_step = 0                          # ตัดสินใจทุก DECISION_STEP_SEC
                block = np.array(self._epoch_buf)
                _bad = self._artifact_reason(block) if ARTIFACT_REJECT else None
                if _bad:
                    print(f"[decision] REJECT epoch ({_bad}) → ไม่สั่ง")
                    self._last_det_key = None
                    self.detector._prev_best = None
                    self.detector._confirm_key, self.detector._confirm_n = None, 0   # v4.6: epoch เสียทำให้การยืนยันซ้ำเริ่มใหม่
                    return_skip = True
                else:
                    return_skip = False
                f_e, psd_e = self.proc.compute_psd(block)
                self._last_scores = self.detector.compute_scores(f_e, psd_e, smooth=False)
                if return_skip:
                    _avg = self._last_scores
                else:
                    self._last_det_key, _sc, _avg = self.decider.update(self._last_scores, self.detector)
                self._last_scores = _avg                      # แสดงคะแนนเฉลี่ยที่ใช้ตัดสินจริง
                if self._last_det_key is not None:
                    epoch_action = SSVEP_CONFIG[self._last_det_key]["action"]
                if PSDA_MODE:
                    if return_skip:
                        self.detector.last_z = self.detector.zscores(_avg)
                        self.detector.last_best = None
                    _z = self.detector.last_z or {}
                    # แท่งความถี่: เต็มสเกล (=3.0) เมื่อ z ถึงเกณฑ์ PSDA_Z_THR
                    self._last_scores = {k: max(_z.get(k, 0.0), 0.0) / PSDA_Z_THR * 3.0 for k in SSVEP_CONFIG}
                    _b = getattr(self.detector, "last_best", None)
                    _snr = "  ".join(f"{k}:{_avg[k]:.2f}" for k in SSVEP_CONFIG)
                    if _b:
                        print(f"[decision/PSDA] best={_b[0]}Hz z={_b[1]:.1f} (2nd z={_b[2]:.1f}) "
                              f"| thr z>={PSDA_Z_THR} strong>={PSDA_Z_STRONG} | SNR {_snr} "
                              f"| command={self._last_det_key} ({getattr(self.detector, 'last_reason', '')})")
                    else:
                        print(f"[decision/PSDA] SNR {_snr} | command={self._last_det_key} (epoch ถูกทิ้ง/ไม่มี null)")
                else:
                    _top = sorted(_avg.items(), key=lambda x: x[1], reverse=True)[:2]
                    print(f"[decision] best={_top[0][0]}Hz {_top[0][1]:.2f} | 2nd={_top[1][0]}Hz {_top[1][1]:.2f} "
                          f"| norm={getattr(self.detector, 'last_norm_best', (0, 0))[1]:.2f} "
                          f"ratio={getattr(self.detector, 'last_ratio', 0):.2f} "
                          f"| command={self._last_det_key}")
        scores = self._last_scores
        det_key = self._last_det_key

        # ── wheelchair control (ผ่าน CommandGate) ──
        res = self.gate.update(epoch_action, allow_move=analysis_on)
        action = res.action                       # "stop" หรือทิศที่กำลังค้างอยู่
        key_name = self._action_key.get(action, "STOP")

        if self.keyboard:
            if res.event == "start":
                self.keyboard.hold(action, res.hold_sec)
            elif res.event in ("end", "stop"):
                self.keyboard.release_all()

        # Update direction pad
        pad_color = "#4CAF50"
        for _cfg in SSVEP_CONFIG.values():
            if _cfg["action"] == action:
                pad_color = _cfg["color"]      # v4.3: ปุ่มติดสีเดียวกับความถี่ที่ชนะ
                break
        self.direction_pad.set_active(key_name, pad_color)

        # Update action label
        if not analysis_on:
            text = ("Status: NO SIGNAL — recording baseline (no detection)" if recording
                    else "Status: NO SIGNAL — record baseline first")
            color = CLR_DIM
        elif res.state == ACTIVE:
            text = f"Status: MOVING ({self._action_label[action]})  {res.remaining:.1f}s"
            color = CLR_GREEN
        elif res.state == LOCKOUT:
            text, color = f"Status: WAIT {res.remaining:.1f}s", "#ffc107"
        else:
            if len(self._epoch_buf) < self._epoch_n:        # ยังเก็บข้อมูลให้เต็มหน้าต่าง
                nxt = (self._epoch_n - len(self._epoch_buf)) / SAMPLE_RATE
            else:
                nxt = max(0.0, (self._step_n - self._since_step) / SAMPLE_RATE)
            text, color = f"Status: IDLE (STOPPED) — next analysis in {nxt:.1f}s", CLR_RED
        self.lbl_action.setText(text)
        self.lbl_action.setStyleSheet(f"color: {color};")

        # ── update metrics ──
        psd_max = np.max(psd) if len(psd) > 0 else 0
        psd_max_freq = f_psd[np.argmax(psd)] if len(psd) > 0 else 0
        filt_max = np.max(np.abs(filt_arr[-n_disp:])) if n_disp > 0 else 0

        self._met_labels[0].setText(f"{psd_max:.4f}")
        self._met_labels[1].setText(f"{filt_max:.2f}")
        self._met_labels[2].setText(f"{psd_max_freq:.1f} Hz")
        self._met_labels[3].setText(str(self._loop_count))

        # ── frequency bars ──
        max_score = max(scores.values()) if scores else 1.0
        bar_max = max(max_score * 1.2, 3.0)
        for k, fb in self.freq_bars.items():
            fb.set_value(scores.get(k, 0), bar_max)

        # no-signal bar shows inverse confidence
        if det_key is None:
            self.bar_nosignal.set_value(bar_max * 0.8, bar_max)
        else:
            self.bar_nosignal.set_value(0, bar_max)

        # ── detected index ──
        if det_key is not None:
            cfg = SSVEP_CONFIG[det_key]
            self.lbl_det_val.setText(str(det_key))
            self.lbl_det_val.setStyleSheet(f"color: {cfg['color']};")
            self.lbl_det_name.setText(cfg["label"])
            self.det_circle.set_color(cfg["color"])
        else:
            self.lbl_det_val.setText("—")
            self.lbl_det_val.setStyleSheet(f"color: {CLR_ACCENT};")
            self.lbl_det_name.setText("No Signal")
            self.det_circle.set_color(CLR_NOSIGNAL)

        # ── connection ──
        self._update_connection()

    def _emergency_stop(self):
        """กด Space: หยุดทันที ไม่ต้องรอรอบวิเคราะห์ถัดไป"""
        self.gate.force_stop()
        if self.keyboard:
            self.keyboard.release_all()
        if getattr(self, "daq", None):
            self.daq.send_command("stop")
        self.direction_pad.set_active("STOP")
        self.lbl_action.setText("Status: EMERGENCY STOP (Space)")
        self.lbl_action.setStyleSheet(f"color: {CLR_RED};")

    def _update_connection(self):
        self.lbl_status_hdr.setText(f"● {self.source.status_text}")
        self.lbl_conn.setText(self.source.status_text)
        self.lbl_samples.setText(f"Samples: {self.source.total_samples}")
        if self.source.connected:
            self.conn_circle.set_color(CLR_GREEN)
        else:
            self.conn_circle.set_color(CLR_RED)

    # ================================================================
    #  BASELINE — [BUG-2 & BUG-3 FIX] State Machine Implementation
    # ================================================================
    def _artifact_reason(self, block):
        """คืนข้อความเหตุผลถ้า epoch นี้เป็น artifact (ขยับ/spike/แอมพลิจูดพุ่ง) ไม่งั้น None
        ใช้อัตราส่วนสัมพัทธ์ จึงไม่ขึ้นกับหน่วยของสัญญาณ"""
        x = np.asarray(block, dtype=float)
        x = x - np.mean(x)
        rms = float(np.sqrt(np.mean(x * x))) + 1e-12
        hist = getattr(self, "_rms_hist", None)
        if hist is None:
            hist = self._rms_hist = []
        reason = None
        n_rej = getattr(self, "_rms_rej_n", 0)
        if float(np.max(np.abs(x))) / rms > ARTIFACT_PEAK_X:
            reason = f"spike {np.max(np.abs(x)) / rms:.1f}x"       # spike จริง: ทิ้งเสมอ
        elif len(hist) >= ARTIFACT_MIN_N and rms > ARTIFACT_RMS_X * float(np.median(hist)):
            n_rej += 1
            if n_rej >= ARTIFACT_RMS_MAX_CONSEC:
                # v4.6: ระดับสูงต่อเนื่อง = ระดับใหม่ของอิเล็กโทรด → รับ epoch และตั้งค่าอ้างอิงใหม่
                # (เดิมค่าอ้างอิงค้างที่ค่าเก่า ทำให้ทิ้งทุก epoch ตลอดไป)
                print(f"[artifact] RMS level shift {rms / float(np.median(hist)):.1f}x ต่อเนื่อง {n_rej} รอบ → ปรับค่าอ้างอิงใหม่")
                hist.clear()
                n_rej = 0
            else:
                reason = f"rms {rms / float(np.median(hist)):.1f}x"
        else:
            n_rej = 0
        self._rms_rej_n = n_rej
        if reason is None:            # เก็บเฉพาะ epoch ปกติ ไม่ให้ artifact ดึงค่าอ้างอิงขึ้น
            hist.append(rms)
            if len(hist) > ARTIFACT_HIST:
                hist.pop(0)
        return reason

    def _on_baseline_click(self):
        """Start baseline recording.

        [BUG-3 FIX] Transitions to RECORDING state and resets accumulator.
        [BUG-4 FIX] Warns if filter is not yet warmed up.
        """
        if self._bl_state == BaselineState.RECORDING:
            return  # already recording

        # Reset accumulator for fresh recording
        self.detector.reset_accumulator()
        self.detector.clear_baseline()
        self.detector.reset_null()
        self._bl_samples = []
        self._rms_hist = []
        self._rms_rej_n = 0

        self._bl_state  = BaselineState.RECORDING
        self._bl_start  = time.time()
        self._bl_frames = 0

        self.btn_baseline.setEnabled(False)
        self.btn_baseline.setText("⏺  Recording...")
        self.btn_bl_clear.setEnabled(False)
        self.bar_bl.setValue(0)

        if not self.proc.is_warmed_up:
            self.lbl_bl_status.setText("⚠ Waiting for filter warm-up, then recording...")
        else:
            self.lbl_bl_status.setText("Recording... 0/30s  (0 frames)")
        self.lbl_bl_status.setStyleSheet(f"color: {CLR_ACCENT};")

    def _finish_baseline(self):
        """Finalize baseline recording.

        [BUG-2 FIX] Calls finalize_baseline() which computes the mean PSD
        from all accumulated frames (Running Average), not a single snapshot.
        """
        self._bl_state = BaselineState.READY
        self.btn_baseline.setEnabled(True)
        self.btn_baseline.setText("▶  Re-record Baseline")
        self.btn_bl_clear.setEnabled(True)
        self.bar_bl.setValue(100)

        success = self.detector.finalize_baseline()
        if success and PSDA_MODE:
            success = self.detector.calibrate_null_from_signal(self.proc, self._bl_samples)
            self._bl_samples = []
            if not success:
                print("[WARN] baseline สั้นเกินไปสำหรับ null calibration (ต้องมี ≥ ~13 s ของสัญญาณที่กรองแล้ว) → ทำ baseline ใหม่")
            else:
                _m = self.detector.null_mu; _s = self.detector.null_sd
                print("[null] ln(SNR) median/SD ต่อเป้าหมาย: " +
                      "  ".join(f"{k}Hz {np.exp(_m[k]):.2f}/{_s[k]:.2f}" for k in SSVEP_CONFIG))
        if success:
            self.lbl_bl_status.setText(
                f"✔ Baseline ready  ({self._bl_frames} frames averaged)"
            )
            self.lbl_bl_status.setStyleSheet(f"color: {CLR_GREEN};")
        else:
            self._bl_state = BaselineState.IDLE
            self.lbl_bl_status.setText("✕ Failed — no valid data collected")
            self.lbl_bl_status.setStyleSheet(f"color: {CLR_RED};")
            self.btn_baseline.setText("▶  Record Baseline  (30s)")
            self.btn_bl_clear.setEnabled(False)

    def _on_baseline_clear(self):
        """Clear baseline and return to IDLE state.

        [BUG-3 FIX] New method — allows user to explicitly clear baseline.
        """
        self.detector.clear_baseline()
        self.detector.reset_accumulator()
        self.detector.reset_null()
        self._bl_samples = []
        self._rms_hist = []
        self._rms_rej_n = 0
        self._bl_state  = BaselineState.IDLE
        self._bl_frames = 0
        self.bar_bl.setValue(0)
        self.btn_baseline.setText("▶  Record Baseline  (30s)")
        self.btn_baseline.setEnabled(True)
        self.btn_bl_clear.setEnabled(False)
        self.lbl_bl_status.setText("Cleared — press ▶ to re-record")
        self.lbl_bl_status.setStyleSheet(f"color: {CLR_DIM};")

    # ================================================================
    #  CLEANUP
    # ================================================================
    def closeEvent(self, event):
        self._timer.stop()
        self.source.stop()
        if self.keyboard:
            self.keyboard.release_all()
        event.accept()


# ╔════════════════════════════════════════════════════════════════════╗
# ║               GANGLION SENDER (Standalone mode)                   ║
# ╚════════════════════════════════════════════════════════════════════╝

def run_sender(duration=120000):
    """Run as a standalone Ganglion sender (replaces ganglion_sender.py).

    Reads EEG data from OpenBCI Ganglion via BrainFlow and sends
    Channel 3 data over UDP to the analyzer.
    """
    if not HAS_BRAINFLOW:
        print("❌ BrainFlow is not installed. Install it with:")
        print("   pip install brainflow")
        return

    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    params.serial_port = SERIAL_PORT
    if BOARD_NAME:
        params.other_info = BOARD_NAME

    board = BoardShim(BOARD_ID, params)
    print("Preparing session...")
    print("ตรวจสอบให้แน่ใจว่าไฟสีน้ำเงินบนบอร์ดกำลัง 'กระพริบ'")

    udp_sock = None
    data_file = None
    try:
        board.prepare_session()
        board.start_stream()
        print(f"✅ Start streaming CH3 → UDP {UDP_IP}:{UDP_PORT}")
        print("Press Ctrl+C to stop.\n")

        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data_file = open("data.txt", "w")

        start_time = time.time()
        sample_count = 0

        while (time.time() - start_time) < duration:
            data = board.get_board_data()
            if data.shape[1] > 0:
                for i in range(data.shape[1]):
                    val_uv = data[SELECTED_CH][i] * SCALE_FACTOR
                    message = f"{val_uv:.3f}"
                    udp_sock.sendto(message.encode(), (UDP_IP, UDP_PORT))
                    data_file.write(message + "\n")
                    sample_count += 1

                if sample_count % 1000 == 0:
                    print(f"  Sent {sample_count} samples...")
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n⚠️ หยุดด้วย Ctrl+C")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        print("\nCleaning up...")
        if board.is_prepared():
            board.stop_stream()
            board.release_session()
            print("✅ Board session released.")
        else:
            BoardShim.release_all_sessions()
            print("✅ Sessions force-cleared.")
        if udp_sock:
            udp_sock.close()
        if data_file:
            data_file.close()
        print("✅ UDP + File closed. Done.")


# ╔════════════════════════════════════════════════════════════════════╗
# ║                  ENTRY POINT                                      ║
# ╚════════════════════════════════════════════════════════════════════╝

def main():
    global SERIAL_PORT, UDP_PORT, BOARD_NAME

    parser = argparse.ArgumentParser(
        description="LightBrain — SSVEP Two-Command Controller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
SSVEP Frequency Mapping:
  9 Hz  → [9, 18, 27] — เลี้ยวซ้าย (Left / A)
  15 Hz → [15, 30]    — เลี้ยวขวา (Right / D)
  No valid SSVEP → STOP; Space → Emergency STOP

Modes:
  --udp        Receive EEG data via UDP (needs ganglion_sender)
  --brainflow  Connect directly to Ganglion via BrainFlow
  --demo       Run with synthetic SSVEP signal
  --sender     Run as Ganglion sender only (no UI)
""")

    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--demo",      action="store_true", help="Synthetic signal demo")
    grp.add_argument("--brainflow", action="store_true", help="Direct BrainFlow connection")
    grp.add_argument("--udp",       action="store_true", help="UDP receiver (default)")
    grp.add_argument("--sender",    action="store_true", help="Ganglion sender mode (no UI)")

    parser.add_argument("--port",     type=str, default=SERIAL_PORT,  help="Serial port (BrainFlow)")
    parser.add_argument("--udp-port", type=int, default=UDP_PORT,     help="UDP port")
    parser.add_argument("--board-name", type=str, default="",         help="Board name (e.g. Ganglion-b2a4)")
    parser.add_argument("--no-keyboard", action="store_true",         help="Disable keyboard output")
    parser.add_argument("--duration", type=int, default=120000,       help="Sender duration (seconds)")

    args = parser.parse_args()

    SERIAL_PORT = args.port
    UDP_PORT    = args.udp_port
    if args.board_name:
        BOARD_NAME = args.board_name

    # ── Sender-only mode ──
    if args.sender:
        run_sender(args.duration)
        return

    # ── GUI mode ──
    if args.demo:
        mode = "demo"
    elif args.brainflow:
        mode = "brainflow"
    else:
        mode = "udp"

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    source = DataSource(mode)
    win = MainWindow(source, enable_keyboard=not args.no_keyboard)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
