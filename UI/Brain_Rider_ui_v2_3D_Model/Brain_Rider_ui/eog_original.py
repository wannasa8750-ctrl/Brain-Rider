"""Exact algorithmic definitions copied without edits from supplied EOG source.
Do not change algorithm bodies; compare AST against original_sources in tests.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.signal import butter, sosfilt, find_peaks

FS_DEFAULT = 200
CALIBRATION_SECONDS = 20
W_HOLD_SECONDS = 1.0
COOLDOWN_SECONDS = 2.0         # begins AFTER W releases
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


def extract_events(samples, fs, center, sigma, min_amplitude=None, threshold_factor=3.0):
    """Extract candidate peaks, group biphasic peaks, measure waveform features.

    Robust local amplitude threshold is for candidate extraction only, NOT a
    baseline quality threshold. Event decisions are made by the model.
    """
    x = np.asarray(samples, dtype=float)
    if x.size < max(20, int(.85*fs)) or not np.isfinite(x).all():
        return []
    # Use the local window median to compensate slow electrode / gaze drift.
    # This is NOT a baseline pass/fail test. Both calibration and LIVE use it.
    local_center = float(np.median(x))
    y = x - local_center
    thresh = max(float(threshold_factor)*float(sigma), 8.0)
    if min_amplitude is not None:
        thresh = max(thresh, float(min_amplitude))
    peaks, props = find_peaks(np.abs(y), height=thresh,
                              prominence=max(1.8*float(sigma), 4.0),
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
    def __init__(self, natural, intentional, noise_sigma, threshold_factor=3.0):
        self.natural = list(natural)
        self.intentional = list(intentional)
        self.noise_sigma = float(noise_sigma)
        self.ready = len(self.intentional)>=2
        self.warning = ''
        self.threshold = max(float(threshold_factor)*self.noise_sigma, 8.0)
        self.natural_count = len(self.natural)
        self.intentional_count = len(self.intentional)
        if not self.ready:
            self.warning = ('Need at least 2 INTENTIONAL blink candidates to build a template. '
                            f'Found natural={len(self.natural)}, intentional={len(self.intentional)}; select SENSITIVE or re-record intentional 20s.')
            return
        if len(self.natural) < 2:
            self.warning = (f'Only {len(self.natural)} natural blink examples found; simulator only, '
                            'false activations may be high. Run REST TEST before external control.')
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
        self.min_correlation=max(.40, float(np.percentile(i_corr, 10))-.24)
        # Use calibrated amplitude range to avoid enormous electrode spikes.
        amp_i=np.array([e['amplitude'] for e in self.intentional])
        self.amp_floor = max(self.threshold, .32*float(np.median(amp_i)))
        self.amp_ceiling=max(3.5*float(np.percentile(amp_i,95)), self.amp_floor*2)
        # Velocity-to-amplitude rejects sharp spikes of short duration.
        sharp=np.array([e['velocity']/max(e['amplitude'],1e-8) for e in self.intentional])
        self.max_sharp = max(2.2*float(np.percentile(sharp,95)), 5.0)
        # Normal class distance radius: learned nearest-neighbour spread.
        intra=[]
        for idx, v in enumerate(self.intent_feats):
            others=np.delete(self.intent_feats,idx,axis=0)
            if len(others): intra.append(float(np.min(np.linalg.norm(others-v,axis=1))))
        self.int_radius=max(2.3, 2.2*float(np.percentile(intra,90))) if intra else 2.8
        # Requiring examples from both classes is for classifier readiness; recordings always complete.

    def model_diagnostic(self):
        """Leave-one-out check on natural training events; not prospective validation."""
        if not self.ready:
            return {'false_intentional': None, 'natural_examples': len(self.natural),
                    'intentional_examples': len(self.intentional)}
        false_i = 0
        for j, ev in enumerate(self.natural):
            amp = ev['amplitude']
            sharp = ev['velocity']/max(amp, 1e-8)
            if amp < self.amp_floor or amp > self.amp_ceiling or sharp > self.max_sharp:
                continue
            if _corr(ev['shape'], self.template) < self.min_correlation:
                continue
            feat = _ln_features(ev)/self.scale
            di = float(np.min(np.linalg.norm(self.intent_feats-feat,axis=1)))
            natural_others = np.delete(self.natural_feats,j,axis=0)
            if len(natural_others) == 0:
                continue
            dn = float(np.min(np.linalg.norm(natural_others-feat,axis=1)))
            if di <= self.int_radius and di + .08 < dn:
                false_i += 1
        return {'false_intentional': false_i, 'natural_examples': len(self.natural),
                'intentional_examples': len(self.intentional)}

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
        intentional_wins=(di + .08 < dn) if np.isfinite(dn) else True
        if di <= self.int_radius and intentional_wins:
            label='INTENTIONAL'
            reason='INTENTIONAL_WINS'
        else:
            label='NATURAL / UNKNOWN' if dn <= di+.08 else 'ARTIFACT'
            reason='NATURAL_WINS' if dn <= di+.08 else 'OUT_OF_TRAINING_RANGE'
        return label, {'reason':reason, 'corr':corr, 'd_int':di, 'd_nat':dn,
                       'radius':self.int_radius, 'threshold':self.threshold, 'amp_floor':self.amp_floor}


@dataclass
class GateResult:
    state: str
    event: str
    remaining: float
    command: bool = False


class SingleBlinkGate:
    """One recognized intentional blink -> hold W 1.0s -> cooldown 2.0s.

    Cooldown starts AFTER releasing W. A monotonic clock is supplied by caller.
    A blink during HOLD/COOLDOWN is never queued. STOP/reset always wins.
    """
    def __init__(self, hold=W_HOLD_SECONDS, cooldown=COOLDOWN_SECONDS):
        self.hold = float(hold)
        self.cooldown = float(cooldown)
        if self.hold <= 0 or self.cooldown < 0:
            raise ValueError('Invalid motion timing')
        self.reset()

    def reset(self):
        self.state = 'READY'
        self.hold_until = None
        self.cooldown_until = None

    def tick(self, now):
        now = float(now)
        if self.state == 'HOLD' and now >= self.hold_until:
            self.state = 'COOLDOWN'
            # Start cooldown when release is processed, even if caller is late.
            self.cooldown_until = now + self.cooldown
            return GateResult('COOLDOWN', 'RELEASE_W', self.cooldown)
        if self.state == 'COOLDOWN' and now >= self.cooldown_until:
            self.state = 'READY'
            return GateResult('READY', 'READY', 0.0)
        if self.state == 'HOLD':
            return GateResult('HOLD', 'HOLDING', max(0.0, self.hold_until - now))
        if self.state == 'COOLDOWN':
            return GateResult('COOLDOWN', 'COOLDOWN', max(0.0, (self.cooldown_until or now) - now))
        return GateResult('READY', 'IDLE', 0.0)

    def accept(self, now):
        now = float(now)
        self.tick(now)
        if self.state != 'READY':
            return GateResult(self.state, 'IGNORED_LOCKOUT',
                              max(0.0, (self.cooldown_until or now) - now))
        self.state = 'HOLD'
        self.hold_until = now + self.hold
        self.cooldown_until = self.hold_until + self.cooldown
        return GateResult('HOLD', 'COMMAND_W', self.hold, True)


class VirtualWheelchair:
    """Non-physical straight-track simulator; output is a modelled position only."""
    def __init__(self, speed_m_per_sec=0.25, track_limit_m=10.0):
        self.speed = float(speed_m_per_sec)
        self.limit = float(track_limit_m)
        self.position = 0.0
        self.last_time = None

    def update(self, now, moving):
        now = float(now)
        if self.last_time is not None and moving:
            # Cap dt so a frozen GUI cannot create a large motion jump.
            dt = min(0.15, max(0.0, now-self.last_time))
            self.position = min(self.limit, self.position + self.speed*dt)
        self.last_time = now
        return self.position

    def reset(self):
        self.position = 0.0
        self.last_time = None


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


WARMUP_SEC = 2.0
