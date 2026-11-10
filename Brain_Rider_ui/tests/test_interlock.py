"""New hybrid arbitration only: original classifiers and gates themselves unchanged."""
import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import eog_original as E
import ssvep_original as S
from hybrid_engine import HybridEngine
from desktop_output import DesktopOutput

class IntegrationInterlockTests(unittest.TestCase):
    def test_eog_w_active_blocks_new_ssvep_left_command(self):
        engine=HybridEngine()
        engine.armed=True
        engine.eog_gate.accept(10.)
        # The original detector returns a 9Hz candidate; the NEW arbiter must suppress it.
        engine._epoch_buf=[0.0]*990
        engine._since_step=990
        engine.decider.update=lambda scores,detector:(9,3.0,scores)
        engine._ssvep_detection(np.zeros(10),now=10.1)
        self.assertEqual(engine.motion,'forward')
        self.assertNotEqual(engine.ss_gate.state,S.ACTIVE)
        self.assertTrue(any(k=='ARBITRATION' for _,k,_ in engine.events))

    def test_ssvep_active_blocks_new_eog_blink_command(self):
        engine=HybridEngine()
        engine.armed=True
        engine.ss_gate.update('left',now=10.)
        class DummyModel:
            ready=True
            threshold=8.0
            amp_floor=8.0
            def predict(self,ev):return 'INTENTIONAL',{'corr':.9,'reason':'INTENTIONAL_WINS'}
        engine.model=DummyModel()
        engine.center=0.;engine.sigma=1.
        engine.filt_e.extend([0.0]*480)
        engine.total=500
        engine.live_next_index=0
        engine.last_processed_peak=0
        original=E.extract_events
        E.extract_events=lambda *a,**kw:[{'index':300,'amplitude':400.,'duration':.2,'velocity':1000.}]
        try:
            engine._eog_detection(now=10.2)
        finally:
            E.extract_events=original
        self.assertEqual(engine.motion,'left')
        self.assertEqual(engine.eog_gate.state,'READY')
        self.assertTrue(any(k=='ARBITRATION' for _,k,_ in engine.events))

    def test_keyboard_is_disabled_by_default_and_nonwindows_no_io(self):
        output=DesktopOutput()
        self.assertFalse(output.enabled)
        self.assertIsNone(output.current)
        self.assertTrue(output.update('left'))
        self.assertIsNone(output.current)
        output.close()

if __name__=='__main__':unittest.main()
