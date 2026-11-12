"""Exact algorithmic definitions copied without edits from supplied two-command SSVEP source.
The only additions are module imports and the original artifact method wrapped in a mixin.
Do not change algorithm bodies; compare AST against original_sources in tests.
"""
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy import signal as sig

SCALE_FACTOR    = 0.02235
SELECTED_CH     = 2          # Channel 3 (0-indexed)


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

for hz in (7, 8, 17):
    harmonics = [n * hz for n in (1, 2, 3) if n * hz <= FILTER_HIGH]
    SSVEP_CONFIG[hz] = {"freqs": harmonics, "weights": [1.0, 0.5, 0.25][:len(harmonics)]}

TARGET_FREQUENCIES = tuple(sorted(SSVEP_CONFIG))

KEY_CODES = {"A": 0x1E, "D": 0x20}


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


class SSVEPArtifactMixin:
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



