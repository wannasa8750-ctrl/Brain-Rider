"""Single Ganglion acquisition session, paired EOG/EEG samples, or synthetic demo.
No physical actuator connection.
"""
import threading
import time
from collections import deque
import numpy as np
import ssvep_original as S

class Acquisition:
    def __init__(self,mode='demo',eog_ch=1,ssvep_ch=3,port='COM3',fw2=False,
                 board_name='',legacy_row2=False):
        if eog_ch == ssvep_ch and not legacy_row2:
            raise ValueError('EOG and SSVEP EEG channels must be distinct')
        self.mode=mode;self.eog_ch=eog_ch;self.ssvep_ch=ssvep_ch
        self.port=port;self.fw2=fw2;self.board_name=board_name;self.legacy_row2=legacy_row2
        self.fs=200
        self.status='WAITING'
        self.connected=False
        self.error=None
        self.running=False
        self.lock=threading.RLock()
        self.queue=deque(maxlen=300)
        self.dropped=False
        self.thread=None
        self.demo_time=0.0
        self.demo_target=0
        self.demo_training=None
        self.demo_injected=[]

    def start(self):
        if self.thread and self.thread.is_alive():return
        self.running=True
        self.thread=threading.Thread(target=self._run,daemon=True,name='Hybrid Ganglion Source')
        self.thread.start()

    def stop(self):
        self.running=False
        if self.thread and self.thread.is_alive():self.thread.join(timeout=3.0)

    def pull(self):
        with self.lock:
            if not self.queue:return None
            chunks=list(self.queue)
            self.queue.clear()
            dropped=self.dropped
            self.dropped=False
        return np.concatenate([x[0] for x in chunks]),np.concatenate([x[1] for x in chunks]),dropped

    def _push(self,eog,eeg):
        if len(eog)!=len(eeg) or not len(eog):return
        with self.lock:
            if len(self.queue)>=self.queue.maxlen:self.dropped=True
            self.queue.append((np.asarray(eog,float).copy(),np.asarray(eeg,float).copy()))

    def set_target(self,hz):
        with self.lock:self.demo_target=int(hz)

    def set_training(self,kind):
        with self.lock:self.demo_training=kind

    def inject_blink(self,kind):
        with self.lock:self.demo_injected.append((self.demo_time+0.18,kind))

    def _run_demo(self):
        self.status='CONNECTED · SYNTHETIC EOG+SSVEP · 200 Hz'
        self.connected=True
        rng=np.random.default_rng(42)
        pulses=[]
        next_auto=0.0
        while self.running:
            with self.lock:
                training=self.demo_training
                target=self.demo_target
                pulses.extend(self.demo_injected);self.demo_injected.clear()
                if training and self.demo_time>=next_auto:
                    pulses.append((self.demo_time+0.20,training))
                    next_auto=self.demo_time+(3.2 if training=='natural' else 2.15)
                if not training:next_auto=self.demo_time+1.0
            n=10
            t=self.demo_time+np.arange(n)/self.fs
            eog=rng.normal(0,8,n)+11*np.sin(2*np.pi*.2*t)
            ss=rng.normal(0,1.6,n)+.7*np.sin(2*np.pi*10*t)
            if target in (9,15):
                ss+=7*np.sin(2*np.pi*target*t)+2.5*np.sin(2*np.pi*2*target*t)
            keep=[]
            for start,kind in pulses:
                d=t-start
                if kind=='intentional':amp,width=460,.16
                elif kind=='artifact':amp,width=1200,.012
                else:amp,width=230,.085
                eog+=amp*np.exp(-.5*(d/width)**2)
                if kind!='artifact':eog-=.3*amp*np.exp(-.5*((d-.28)/(.10 if kind=='intentional' else .07))**2)
                if t[-1]<start+1.1:keep.append((start,kind))
            pulses=keep
            # In original SSVEP BrainFlow path, raw EEG is multiplied by SCALE_FACTOR=0.02235.
            self._push(eog,ss*S.SCALE_FACTOR)
            self.demo_time+=n/self.fs
            time.sleep(n/self.fs)
        self.connected=False
        self.status='DISCONNECTED'

    def _run_board(self):
        try:
            from brainflow.board_shim import BoardShim,BrainFlowInputParams,BoardIds
        except ImportError as exc:
            raise RuntimeError('BrainFlow unavailable. Install requirements.txt') from exc
        bid=BoardIds.GANGLION_BOARD if self.mode=='dongle' else BoardIds.GANGLION_NATIVE_BOARD
        params=BrainFlowInputParams()
        if self.mode in ('dongle','sender'):params.serial_port=self.port
        if self.mode!='dongle':
            if self.fw2:params.other_info='fw:2'
            elif self.board_name:params.other_info=self.board_name
        board=None;streaming=False
        try:
            self.status='CONNECTING · Ganglion'
            board=BoardShim(bid,params)
            board.prepare_session();board.start_stream();streaming=True
            self.fs=int(BoardShim.get_sampling_rate(bid))
            channels=BoardShim.get_eeg_channels(bid)
            if min(self.eog_ch,self.ssvep_ch)<1 or max(self.eog_ch,self.ssvep_ch)>len(channels):
                raise ValueError('EEG channel outside the BrainFlow Ganglion available channel set')
            eog_row=int(channels[self.eog_ch-1])
            ssvep_row=S.SELECTED_CH if self.legacy_row2 else int(channels[self.ssvep_ch-1])
            if ssvep_row==eog_row:
                raise ValueError('EOG and SSVEP are mapped to the same data row')
            self.connected=True
            self.status=(f'CONNECTED · Ganglion {self.fs} Hz · EOG EEG-CH{self.eog_ch} '
                         f'· SSVEP {"raw row 2" if self.legacy_row2 else "EEG-CH"+str(self.ssvep_ch)}')
            while self.running:
                data=board.get_board_data()
                if data.shape[1]:
                    ea=np.asarray(data[eog_row],float)
                    ss=np.asarray(data[ssvep_row],float) * S.SCALE_FACTOR # EXACT original SSVEP SCALE_FACTOR
                    if not (np.isfinite(ea).all() and np.isfinite(ss).all()):
                        raise ValueError('Invalid samples (NaN/Inf)')
                    self._push(ea,ss)
                else:time.sleep(.01)
        finally:
            self.connected=False
            if board:
                if streaming:
                    try:board.stop_stream()
                    except Exception:pass
                try:board.release_session()
                except Exception:pass

    def _run(self):
        try:
            if self.mode=='demo':self._run_demo()
            else:self._run_board()
        except Exception as exc:
            self.error=f'{type(exc).__name__}: {exc}'
            self.status='ERROR: '+self.error
        finally:
            self.running=False
            self.connected=False
