"""Hybrid orchestrator: unchanged original EOG + SSVEP algorithms and original gates.

Only NEW behavior: shared acquisition, sequential calibration, cross-modal arbitration,
simulator output, logging and latched emergency stop. No physical actuator output.
"""
import time
from collections import deque
from pathlib import Path
import csv
import numpy as np

import eog_original as E
import ssvep_original as S


class HybridEngine(S.SSVEPArtifactMixin):
    """Standalone research-simulator logic. Calls original code without changing its bodies."""
    def __init__(self, fs=200, sensitivity=3.0, log_dir=None):
        self.turns = {9: "left", 15: "right"}
        self.sensitivity = float(sensitivity)
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.events = deque(maxlen=500)
        self._events_csv = None
        if self.log_dir:
            f = (self.log_dir / ('events_' + time.strftime('%Y%m%d_%H%M%S') + '.csv')).open('w', newline='', encoding='utf-8-sig')
            self._events_csv = f
            self._csv = csv.writer(f)
            self._csv.writerow(['timestamp','kind','detail','movement','sample_index'])
        self.estop = False
        self.armed = False
        self.last_rx = None
        self.last_wall = None
        self.last_eog_class = '—'
        self.last_eog_features = None
        self.last_scores = {k: 0.0 for k in S.SSVEP_CONFIG}
        self.last_z = {k: 0.0 for k in S.SSVEP_CONFIG}
        self.last_ssvep = None
        self.last_ssvep_reason = 'WAITING FOR BASELINE'
        self.ss_epoch_count = 0
        self.ss_rejected_count = 0
        self.last_command = 'stop'
        self.rest_test = False
        self.rest_deadline = 0.0
        self.rest_events = 0
        self.rest_intentional = 0
        self.set_fs(fs)

    def set_turns(self, left, right):
        if self.armed or left == right or any(hz not in S.TARGET_FREQUENCIES for hz in (left, right)):
            return False
        self.turns = {left: "left", right: "right"}
        return True

    def message(self,kind,detail):
        note=(time.strftime('%H:%M:%S'),str(kind),str(detail))
        self.events.append(note)
        if self._events_csv:
            self._csv.writerow([time.strftime('%Y-%m-%d %H:%M:%S'),kind,detail,self.motion,self.total])
            self._events_csv.flush()

    def set_fs(self,fs):
        fs=int(fs)
        if fs < 65:
            raise ValueError('Sampling rate must be at least 65 Hz for the original 30 Hz harmonic')
        self.levels_ready=False
        self.fs=fs
        self.eog_filter=E.StreamingFilter(fs)
        self.ss_proc=S.SignalProcessor(fs)
        self.detector=S.SSVEPDetector(fs)
        self.decider=S.DecisionFilter()
        self.eog_gate=E.SingleBlinkGate()  # EXACT original: W 1s + 2s from release
        self.ss_gate=S.CommandGate(S.COMMAND_HOLD_SEC,S.COMMAND_LOCKOUT_SEC,
                                   0.0,0.0,S.LOCKOUT_AFTER_STOP_SEC) # EXACT source UI init
        self.raw_e=deque(maxlen=fs*15)
        self.filt_e=deque(maxlen=fs*15)
        self.raw_s=deque(maxlen=fs*30)
        self.filt_s=deque(maxlen=fs*30)
        self.total=0
        self.warmup=0
        self.calibration={'natural':None,'intentional':None}
        self.natural_events=[]
        self.intentional_events=[]
        self.center=self.sigma=None
        self.model=None
        self.phase=None
        self.phase_data=[]
        self.phase_samples_received=0
        self.bl_start=0.0
        self.bl_frames=0
        self.bl_samples=[]
        self._rms_hist=[]
        self._rms_rej_n=0
        self._epoch_n=int(S.EPOCH_SEC*fs)
        self._step_n=int(S.DECISION_STEP_SEC*fs)
        self._epoch_buf=[]
        self._since_step=0
        self._last_det_key=None
        self.last_processed_peak=-10**9
        self.live_next_index=0
        self.last_motion_update=None
        self.virtual=E.VirtualWheelchair() # untouched original forward simulator
        self.last_command='stop'
        self.armed=False

    @property
    def ss_ready(self):
        return (self.detector.baseline_psd is not None and
                (not S.PSDA_MODE or self.detector.null_mu is not None))

    @property
    def validated(self):
        return (self.calibration['natural'] is not None and
                self.calibration['intentional'] is not None and
                self.model is not None and self.model.ready and self.ss_ready)

    @property
    def motion(self):
        if not self.armed or self.estop or self.phase:
            return 'stop'
        if self.eog_gate.state=='HOLD':
            return 'forward'
        if self.ss_gate.state==S.ACTIVE and self.ss_gate._active in ('left','right'):
            return self.ss_gate._active
        return 'stop'

    def stop(self,reason='STOP',emergency=False):
        self.armed=False
        self.rest_test=False
        if emergency:self.estop=True
        self.eog_gate.reset()
        self.ss_gate.force_stop()
        self._epoch_buf=[]
        self._since_step=0
        self.decider.reset()
        self._last_det_key=None
        self.levels_ready=False
        self.last_ssvep=None
        self.last_command='stop'
        self.virtual.update(time.monotonic(),False)
        self.message('STOP', reason)

    def reset_estop(self):
        self.estop=False
        self.stop('ESTOP RESET (DISARMED)')

    def clear_all(self):
        self.stop('CLEAR BASELINES')
        self.estop=False
        self.phase=None
        self.phase_data=[]
        self.phase_samples_received=0
        self.calibration={'natural':None,'intentional':None}
        self.natural_events=[];self.intentional_events=[]
        self.center=self.sigma=None;self.model=None
        self.detector.clear_baseline()
        self.detector.reset_accumulator()
        self.detector.reset_null()
        self._rms_hist=[];self._rms_rej_n=0
        self.bl_frames=0;self.bl_samples=[]
        self.message('CALIBRATION','ALL CLEARED')

    def set_sensitivity(self,factor):
        self.stop('SENSITIVITY CHANGED, RE-VALIDATE')
        self.sensitivity=float(factor)
        if self.calibration['natural'] is not None:
            self.natural_events=E.extract_events(self.calibration['natural'],self.fs,self.center,self.sigma,
                                                 threshold_factor=self.sensitivity)
            if self.calibration['intentional'] is not None:
                self.intentional_events=E.extract_events(self.calibration['intentional'],self.fs,self.center,self.sigma,
                                                         threshold_factor=self.sensitivity)
                self.model=E.PersonalBlinkModel(self.natural_events,self.intentional_events,self.sigma,
                                                threshold_factor=self.sensitivity)
        self.message('SETTING',f'Original EOG sensitivity = {factor}')

    def start_calibration(self,phase,now=None,wall=None):
        if self.estop:
            return False,'Reset emergency stop first.'
        if self.phase:
            return False,'A baseline recording is already active.'
        if self.last_rx is None:
            return False,'No incoming signal yet.'
        if self.warmup < int(E.WARMUP_SEC*self.fs) or not self.ss_proc.is_warmed_up:
            return False,'Wait for the original 2-second filter warm-up.'
        if phase not in ('ssvep','natural','intentional'):
            return False,'Unknown baseline type.'
        if phase=='natural' and not self.ss_ready:
            return False,'Record SSVEP null baseline first.'
        if phase=='intentional' and self.calibration['natural'] is None:
            return False,'Record natural EOG baseline first.'
        self.stop('START CALIBRATION '+phase.upper())
        self.phase=phase
        self.phase_data=[]
        self.phase_samples_received=0
        if phase=='ssvep':
            self.detector.reset_accumulator()
            self.detector.clear_baseline()
            self.detector.reset_null()
            self.bl_frames=0
            self.bl_samples=[]
            self.bl_start=time.time() if wall is None else float(wall)
            self._rms_hist=[];self._rms_rej_n=0
            # SSVEP baseline 30 s by wall clock, exactly source MainWindow behavior.
        if phase=='natural':
            self.calibration={'natural':None,'intentional':None}
            self.model=None;self.center=None;self.sigma=None
            self.natural_events=[];self.intentional_events=[]
        if phase=='intentional':
            self.calibration['intentional']=None
            self.intentional_events=[];self.model=None
        self.message('BASELINE',f'START {phase.upper()} ({S.BASELINE_DURATION if phase=="ssvep" else E.CALIBRATION_SECONDS}s)')
        return True,'Recording '+phase

    def phase_progress(self,wall=None):
        if self.phase=='ssvep':
            elapsed=(time.time() if wall is None else wall)-self.bl_start
            return min(1.0,max(0.0,elapsed/S.BASELINE_DURATION))
        if self.phase in ('natural','intentional'):
            return min(1.0,self.phase_samples_received/(E.CALIBRATION_SECONDS*self.fs))
        return 0.0

    def _finish_calibration(self):
        phase=self.phase
        if phase not in ('natural','intentional'):return
        n=int(E.CALIBRATION_SECONDS*self.fs)
        data=np.concatenate(self.phase_data)[:n] if self.phase_data else np.array([],float)
        self.phase=None;self.phase_data=[];self.phase_samples_received=0
        self.calibration[phase]=data
        if self.log_dir:
            np.savetxt(self.log_dir/f'{phase}_{int(time.time())}.csv',
                       np.column_stack((np.arange(len(data))/self.fs,data)),delimiter=',',
                       header='seconds,filtered_eog_uv',comments='')
        # Original EOG MainWindow.finish_phase algorithm calls (unchanged).
        if phase=='natural':
            self.center,self.sigma=E.natural_reference(data,self.fs)
            self.natural_events=E.extract_events(data,self.fs,self.center,self.sigma,
                                                  threshold_factor=self.sensitivity)
        else:
            self.intentional_events=E.extract_events(data,self.fs,self.center,self.sigma,
                                                      threshold_factor=self.sensitivity)
            self.model=E.PersonalBlinkModel(self.natural_events,self.intentional_events,
                                            self.sigma,threshold_factor=self.sensitivity)
        self.message('BASELINE',f'{phase.upper()} COMPLETE {len(data)} samples; candidates = '
                                f'{len(self.natural_events) if phase=="natural" else len(self.intentional_events)}')
        if phase=='intentional' and self.model.warning:
            self.message('MODEL WARNING',self.model.warning)

    def _finish_ssvep_baseline(self):
        # Same original finalize + PSDA null from filtered baseline sequence.
        valid=self.detector.finalize_baseline()
        if valid and S.PSDA_MODE:
            valid=self.detector.calibrate_null_from_signal(self.ss_proc,self.bl_samples)
        self.phase=None
        self.bl_samples=[]
        self.message('BASELINE','SSVEP COMPLETE' if valid else 'SSVEP FAILED (no valid PSD / null)')
        if not valid:
            self.detector.clear_baseline()
            self.detector.reset_null()
        return valid

    def validate(self):
        if self.phase:return False,'Baseline in progress.'
        if not self.ss_ready:return False,'SSVEP baseline/null not ready.'
        if self.calibration['natural'] is None:return False,'Record EOG natural baseline first.'
        if self.calibration['intentional'] is None:return False,'Record EOG intentional baseline first.'
        if self.model is None or not self.model.ready:
            return False,'Original EOG model not ready: '+(self.model.warning if self.model else 'no model')
        d=self.model.model_diagnostic()
        if d['false_intentional']:
            self.message('WARNING',f'{d["false_intentional"]} natural training events resemble intentional; run REST test')
        self.message('VALIDATE','PASS · original models ready; simulator can ARM')
        return True,'Validated'

    def arm(self,now=None):
        now=time.monotonic() if now is None else float(now)
        if self.estop:return False,'E-STOP LATCHED'
        ok,msg=self.validate()
        if not ok:return False,msg
        if self.last_rx is None or now-self.last_rx>E.NO_DATA_SECONDS:
            return False,'No fresh signal.'
        self.eog_gate.reset()
        self.ss_gate.reset()
        self.last_processed_peak=self.total
        self.live_next_index=self.total
        self._epoch_buf=[];self._since_step=0;self.decider.reset()
        self._rms_hist=[];self._rms_rej_n=0
        self.armed=True
        self.message('ARM','SIMULATOR ARMED')
        return True,'ARMED'

    def start_rest_test(self,now=None):
        now=time.monotonic() if now is None else float(now)
        if not self.model or not self.model.ready:
            return False,'Calibrate and validate the original EOG model first.'
        self.stop('REST TEST: OUTPUT BLOCKED')
        self.rest_test=True
        self.rest_deadline=now+60.0
        self.rest_events=0;self.rest_intentional=0
        self.live_next_index=self.total
        self.last_processed_peak=self.total
        self.message('REST TEST','60s STARTED; output disabled')
        return True,'REST SCREEN 60s STARTED'

    def disarm(self): self.stop('MANUAL DISARM')

    def _eog_detection(self,now):
        # Exact original MainWindow.on_samples extraction/feature and time guard.
        if not self.model or not self.model.ready:return
        step=max(1,int(.10*self.fs))
        window_size=int(E.DETECT_WINDOW_SECONDS*self.fs)
        if self.total-self.live_next_index<step or len(self.filt_e)<window_size:return
        self.live_next_index=self.total
        window=np.asarray(list(self.filt_e)[-window_size:])
        min_amp=max(self.model.threshold,self.model.amp_floor*.62)
        events=E.extract_events(window,self.fs,self.center,self.sigma,min_amp,
                                threshold_factor=self.sensitivity)
        start_abs=self.total-len(window)
        for e in events:
            abs_pk=start_abs+e['index']
            if abs_pk-self.last_processed_peak<int(E.REFRACTORY_SECONDS*self.fs):continue
            if abs_pk<=0:continue
            self.last_processed_peak=abs_pk
            label,met=self.model.predict(e)
            self.last_eog_class=label
            self.last_eog_features={'amplitude':e['amplitude'],'duration':e['duration'],
                                    'velocity':e['velocity'],'corr':met.get('corr',0.0),
                                    'reason':met.get('reason','')}
            self.message('EOG',f'{label}: amp {e["amplitude"]:.1f} µV, NCC={met.get("corr",0):.2f}, '
                               f'{met.get("reason","")}')
            if self.rest_test:
                self.rest_events+=1
                if label=='INTENTIONAL':self.rest_intentional+=1
                continue
            if label=='INTENTIONAL' and self.armed:
                # New arbitration check BEFORE invoking original accept(); gate itself unchanged.
                if self.ss_gate.state==S.ACTIVE:
                    self.message('ARBITRATION','EOG accepted by classifier, command suppressed while SSVEP active')
                    continue
                result=self.eog_gate.accept(now)
                if result.command:
                    self.message('COMMAND','EOG → FORWARD/W 1s; cooldown starts after release')
                elif result.event=='IGNORED_LOCKOUT':
                    self.message('GATE','EOG blink ignored during original cooldown')

    def _ssvep_detection(self,filtered,now):
        # Exact original MainWindow._tick epoch scheduling: EPOCH_SEC == STEP_SEC == 5 seconds.
        self._epoch_buf.extend(filtered.tolist())
        self._since_step+=len(filtered)
        if len(self._epoch_buf)>self._epoch_n:
            del self._epoch_buf[:len(self._epoch_buf)-self._epoch_n]
        epoch_action=None
        if len(self._epoch_buf)>=self._epoch_n and self._since_step>=self._step_n:
            self._since_step=0
            self.ss_epoch_count+=1
            block=np.array(self._epoch_buf)
            bad=self._artifact_reason(block) if S.ARTIFACT_REJECT else None
            if bad:
                self.ss_rejected_count+=1
                self._last_det_key=None
                self.detector._prev_best=None
                self.detector._confirm_key,self.detector._confirm_n=None,0
                self.message('ARTIFACT','SSVEP epoch rejected: '+bad)
            f,p=self.ss_proc.compute_psd(block)
            scores=self.detector.compute_scores(f,p,smooth=False)
            if not bad:
                self._last_det_key,_,avg=self.decider.update(scores,self.detector)
                self.last_ssvep_reason=getattr(self.detector,'last_reason','')
            else:
                avg=scores
                self.last_ssvep_reason='ARTIFACT REJECTED'
            self.levels_ready=not bad
            self.last_scores=avg
            self.last_z=self.detector.zscores(avg) or {k:0.0 for k in S.SSVEP_CONFIG}
            self.last_ssvep=self._last_det_key
            if self._last_det_key is not None:
                epoch_action=self.turns.get(self._last_det_key)
                self.message('SSVEP',f'{self._last_det_key} Hz → {epoch_action or 'monitor only'} '
                            f'z={self.last_z.get(self._last_det_key,0):.2f}')
        # Keep original gate update at every acquisition tick, with candidate only on epoch.
        # New interlock prevents overlapping EOG and SSVEP commands.
        if self.eog_gate.state=='HOLD' and epoch_action:
            self.message('ARBITRATION','SSVEP command suppressed while EOG/W active')
            epoch_action=None
        result=self.ss_gate.update(epoch_action,now=now,allow_move=True)
        if result.event=='start':
            self.message('COMMAND',f'SSVEP → {result.action.upper()} / '
                        f'{"A" if result.action=="left" else "D"} 1s')
        return result

    def process_block(self,eog_raw,ssvep_raw,now=None,wall=None):
        now=time.monotonic() if now is None else float(now)
        wall=time.time() if wall is None else float(wall)
        a=np.asarray(eog_raw,dtype=np.float64)
        b=np.asarray(ssvep_raw,dtype=np.float64)
        if a.ndim!=1 or b.ndim!=1 or len(a)!=len(b) or not len(a) or not (np.isfinite(a).all() and np.isfinite(b).all()):
            self.stop('INVALID OR MISMATCHED CHANNEL SAMPLES',emergency=True)
            return
        fe=self.eog_filter.apply(a)
        fs=self.ss_proc.filter_block(b)
        self.raw_e.extend(a.tolist());self.filt_e.extend(fe.tolist())
        self.raw_s.extend(b.tolist());self.filt_s.extend(fs.tolist())
        self.total+=len(a);self.warmup+=len(a)
        self.last_rx=now;self.last_wall=wall
        if self.phase=='ssvep':
            # Original loop computes PSD of 5 seconds of recent filtered signal each tick.
            analysis=np.asarray(self.filt_s,dtype=float)[-int(S.ANALYSIS_SEC*self.fs):]
            f_psd,psd=self.ss_proc.compute_psd(analysis)
            if self.ss_proc.is_warmed_up and len(psd):
                if len(analysis)>=int(S.ANALYSIS_SEC*self.fs):
                    self.detector.accumulate_baseline(f_psd,psd)
                    self.bl_frames+=1
                self.bl_samples.extend(fs.tolist())
            if wall-self.bl_start>=S.BASELINE_DURATION:
                self._finish_ssvep_baseline()
            self._clear_live_ssvep()
            return
        if self.phase in ('natural','intentional'):
            need=int(E.CALIBRATION_SECONDS*self.fs)-self.phase_samples_received
            if need>0:
                self.phase_data.append(np.asarray(fe[:need]).copy())
                self.phase_samples_received+=min(len(fe),need)
            if self.phase_samples_received>=int(E.CALIBRATION_SECONDS*self.fs):
                self._finish_calibration()
            self._clear_live_ssvep()
            return
        if not self.armed:
            self._clear_live_ssvep()
            if self.rest_test:self._eog_detection(now)
            return
        self.eog_gate.tick(now)
        self._eog_detection(now)
        if self.ss_ready:
            self._ssvep_detection(fs,now)
        else:
            self.ss_gate.force_stop(now)
        self._update_motion(now)

    def _clear_live_ssvep(self):
        self._epoch_buf=[]
        self._since_step=0
        self.decider.reset()
        self.detector._confirm_key,self.detector._confirm_n=None,0
        self._last_det_key=None
        self.levels_ready=False
        self.last_ssvep=None
        self.ss_gate.force_stop()

    def _update_motion(self,now):
        movement=self.motion
        self.virtual.update(now,movement=='forward')
        if movement!=self.last_command:
            self.message('OUTPUT',movement.upper())
            self.last_command=movement

    def tick(self,now=None,wall=None):
        now=time.monotonic() if now is None else float(now)
        if self.last_rx is not None and now-self.last_rx>E.NO_DATA_SECONDS:
            if self.armed:self.stop('NO SIGNAL > 1.2 s',emergency=True)
        if self.rest_test and now >= self.rest_deadline:
            self.rest_test=False
            self.message('REST TEST',f'COMPLETE: {self.rest_intentional} apparent intentional / {self.rest_events} blink events in 60s')
        if self.phase=='ssvep' and (time.time() if wall is None else wall)-self.bl_start>=S.BASELINE_DURATION:
            self._finish_ssvep_baseline()
        if self.armed:
            r=self.eog_gate.tick(now)
            if r.event=='RELEASE_W':self.message('EOG GATE','W RELEASED; 2s cooldown begins')
            if r.event=='READY':self.message('EOG GATE','READY')
            # No decision candidate in clock-only ticks; original gate still expires hold/lockout.
            self.ss_gate.update(None,now=now,allow_move=self.ss_ready)
        self._update_motion(now)

    def close(self):
        self.stop('APPLICATION CLOSING')
        if self._events_csv:
            self._events_csv.close()
            self._events_csv=None
