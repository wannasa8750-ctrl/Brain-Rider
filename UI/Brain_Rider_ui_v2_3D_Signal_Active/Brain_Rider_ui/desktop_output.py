"""Optional desktop STIMULATOR keyboard adapter (Windows only, never motor I/O).
Defaults DISABLED; an independent watchdog attempts key release on stalled GUI.
"""
import ctypes
from ctypes import wintypes
import os
import threading
import time
import ssvep_original as S

SCAN = {'forward': 0x11, 'left': S.KEY_CODES['A'], 'right': S.KEY_CODES['D']} # exact source W/A/D
KEYEVENTF_SCANCODE=0x0008
KEYEVENTF_KEYUP=0x0002

class DesktopOutput:
    def __init__(self,enabled=False):
        self.available=os.name=='nt'
        self.enabled=False
        self.current=None
        self.key_down_since=0.0
        self.last_heartbeat=time.monotonic()
        self.lock=threading.RLock()
        self.failed=False
        self.problem=''
        self.quit=threading.Event()
        if self.available:
            ULONG_PTR=getattr(wintypes,'ULONG_PTR',ctypes.c_size_t)
            class KEYBDINPUT(ctypes.Structure):
                _fields_=[('wVk',wintypes.WORD),('wScan',wintypes.WORD),
                          ('dwFlags',wintypes.DWORD),('time',wintypes.DWORD),
                          ('dwExtraInfo',ULONG_PTR)]
            class MOUSEINPUT(ctypes.Structure):
                _fields_=[('dx',wintypes.LONG),('dy',wintypes.LONG),
                          ('mouseData',wintypes.DWORD),('dwFlags',wintypes.DWORD),
                          ('time',wintypes.DWORD),('dwExtraInfo',ULONG_PTR)]
            class HARDWAREINPUT(ctypes.Structure):
                _fields_=[('uMsg',wintypes.DWORD),('wParamL',wintypes.WORD),('wParamH',wintypes.WORD)]
            class UNION(ctypes.Union):
                _fields_=[('ki',KEYBDINPUT),('mi',MOUSEINPUT),('hi',HARDWAREINPUT)]
            class INPUT(ctypes.Structure):
                _anonymous_=('u',)
                _fields_=[('type',wintypes.DWORD),('u',UNION)]
            self._kbi=KEYBDINPUT;self._input=INPUT
            self._sender=ctypes.WinDLL('user32',use_last_error=True).SendInput
            self._sender.argtypes=(wintypes.UINT,ctypes.POINTER(INPUT),ctypes.c_int)
            self._sender.restype=wintypes.UINT
        self.guard=threading.Thread(target=self._watchdog,daemon=True,name='STIMULATOR-Keyboard-Watchdog')
        self.guard.start()
        if enabled:self.set_enabled(True)

    def _send(self,name,key_up=False):
        if not self.available:return False
        flags=KEYEVENTF_SCANCODE | (KEYEVENTF_KEYUP if key_up else 0)
        x=self._input(type=1,ki=self._kbi(0,SCAN[name],flags,0,0))
        ctypes.set_last_error(0)
        n=self._sender(1,ctypes.byref(x),ctypes.sizeof(x))
        if n!=1:
            self.problem=f'SendInput failed, Win32 error={ctypes.get_last_error()}'
            self.failed=True
            self.enabled=False
            return False
        return True

    def release(self):
        with self.lock:
            if self.current:
                if not self._send(self.current,True):return False
                self.current=None
            return True

    def set_enabled(self,value):
        with self.lock:
            if not self.release():return False
            if value and (not self.available or self.failed):
                self.problem='Desktop Windows SendInput unavailable or previous release failed'
                return False
            self.enabled=bool(value)
            self.last_heartbeat=time.monotonic()
            return True

    def update(self,motion):
        with self.lock:
            now=time.monotonic()
            self.last_heartbeat=now
            if not self.enabled or motion not in SCAN:
                return self.release()
            if self.current==motion:return True
            if not self.release():return False
            if not self._send(motion,False):return False
            self.current=motion
            self.key_down_since=now
            return True

    def _watchdog(self):
        while not self.quit.wait(.025):
            with self.lock:
                if self.current and (time.monotonic()-self.last_heartbeat>.3 or
                                     time.monotonic()-self.key_down_since>1.2):
                    self.release()

    def close(self):
        self.quit.set()
        self.set_enabled(False)
        self.guard.join(timeout=.5)
