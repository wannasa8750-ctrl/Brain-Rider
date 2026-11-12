import unittest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hybrid_engine import HybridEngine
import eog_original as E
import ssvep_original as S

class EngineTests(unittest.TestCase):
    def setUp(self):
        self.e=HybridEngine(fs=200)

    def test_original_eog_gate_hold_release_cooldown(self):
        g=self.e.eog_gate
        self.assertTrue(g.accept(1.0).command)
        self.assertEqual(g.tick(1.99).state,'HOLD')
        self.assertEqual(g.tick(2.0).event,'RELEASE_W')
        self.assertFalse(g.accept(2.1).command)
        self.assertEqual(g.tick(4.0).state,'READY')
        self.assertTrue(g.accept(4.01).command)

    def test_original_ssvep_gate_hold_and_lockout(self):
        g=self.e.ss_gate
        r=g.update('left',now=10.0)
        self.assertEqual((r.event,r.action,r.hold_sec),('start','left',1.0))
        r=g.update('right',now=10.5)
        self.assertEqual(r.action,'left')
        r=g.update(None,now=11.01)
        self.assertEqual(r.event,'end')
        r=g.update('right',now=11.5)
        self.assertEqual(r.state,S.LOCKOUT)
        r=g.update('right',now=12.01)
        self.assertEqual(r.event,'start')

    def test_no_arm_before_baselines(self):
        self.e.last_rx=10.0
        ok,msg=self.e.arm(now=10.0)
        self.assertFalse(ok)
        self.assertIn('SSVEP',msg)
        self.assertEqual(self.e.motion,'stop')

    def test_no_signal_stop_and_latch(self):
        self.e.last_rx=5.0
        self.e.armed=True
        self.e.eog_gate.accept(5.0)
        self.assertEqual(self.e.motion,'forward')
        self.e.tick(now=6.3)
        self.assertEqual(self.e.motion,'stop')
        self.assertTrue(self.e.estop)
        self.assertFalse(self.e.armed)

    def test_original_ssvep_5s_nonoverlap_timing(self):
        e=self.e
        # Build a valid null baseline with original functions; this test is epoch schedule,
        # not prospective classifier validation.
        rng=np.random.default_rng(10)
        baseline=rng.normal(0,.1,200*30)
        b=e.ss_proc.filter_block(baseline)
        for i in range(0,len(b)-1000+1,200):
            f,p=e.ss_proc.compute_psd(b[i:i+1000]);e.detector.accumulate_baseline(f,p)
        self.assertTrue(e.detector.finalize_baseline())
        self.assertTrue(e.detector.calibrate_null_from_signal(e.ss_proc,b))
        e.armed=True;e.last_processed_peak=10**12
        for i in range(0,2000,10):
            t=i/200.0+31.0
            e.process_block(np.zeros(10),.03*np.sin(2*np.pi*9*(np.arange(10)+i)/200),now=t,wall=t)
        self.assertEqual(e.ss_epoch_count,2)
        self.assertEqual(e._step_n,1000)
        self.assertEqual(e._epoch_n,1000)

    def test_stop_disarms_both_original_gates(self):
        e=self.e
        e.armed=True;e.eog_gate.accept(9.0)
        e.ss_gate.update('left',now=9.0)
        e.stop('test',emergency=True)
        self.assertEqual(e.motion,'stop')
        self.assertEqual(e.eog_gate.state,'READY')
        self.assertNotEqual(e.ss_gate.state,S.ACTIVE)

    def test_invalid_samples_fail_stop(self):
        self.e.armed=True
        self.e.process_block(np.array([1.0,float('nan')]),np.array([1.0,2.0]))
        self.assertTrue(self.e.estop)
        self.assertEqual(self.e.motion,'stop')

    def test_baseline_20s_sample_count(self):
        e=self.e
        e.last_rx=0.;e.warmup=400;e.ss_proc._count=400
        e.detector.set_baseline(np.linspace(0,100,513),np.ones(513))
        e.detector.null_mu={9:0.0,15:0.0};e.detector.null_sd={9:.1,15:.1}
        ok,_=e.start_calibration('natural',now=2.,wall=2.)
        self.assertTrue(ok)
        for i in range(20):
            e.process_block(np.ones(200),np.ones(200),now=3+i,wall=3+i)
        self.assertIsNone(e.phase)
        self.assertEqual(len(e.calibration['natural']),4000)

class FullSyntheticFlowTests(unittest.TestCase):
    def test_30_20_20_baseline_then_original_two_ssvep_commands(self):
        """All three original baselines and both SSVEP outputs, synthetic only."""
        rng=np.random.default_rng(42)
        e=HybridEngine(fs=200)
        fs=200;step=20
        def feed(t0,duration,kind=None,hz=0):
            for q in range(0,int(duration*fs),step):
                t=t0+(q+np.arange(step))/fs
                eog=rng.normal(0,8,step)+11*np.sin(2*np.pi*.2*t)
                ss=rng.normal(0,1.6,step)+.7*np.sin(2*np.pi*10*t)
                if hz in (9,15):
                    ss+=7*np.sin(2*np.pi*hz*t)+2.5*np.sin(2*np.pi*2*hz*t)
                if kind in ('natural','intentional'):
                    for pulse in np.arange(t0+1,t0+duration,3.2 if kind=='natural' else 2.15):
                        d=t-pulse
                        amp,width=((230,.085) if kind=='natural' else (460,.16))
                        eog+=amp*np.exp(-.5*(d/width)**2)
                        eog-=.3*amp*np.exp(-.5*((d-.28)/(.07 if kind=='natural' else .10))**2)
                now=t0+(q+step)/fs
                e.process_block(eog,ss*.02235,now=now,wall=now)
                e.tick(now=now,wall=now)
        feed(0,2.5)
        self.assertTrue(e.start_calibration('ssvep',wall=2.5)[0])
        feed(2.5,30.1)
        self.assertTrue(e.ss_ready)
        self.assertTrue(e.start_calibration('natural',wall=32.6)[0])
        feed(32.6,20,kind="natural")
        self.assertEqual(len(e.calibration['natural']),4000)
        self.assertTrue(e.start_calibration('intentional',wall=52.6)[0])
        feed(52.6,20,kind="intentional")
        self.assertEqual(len(e.calibration['intentional']),4000)
        self.assertTrue(e.model.ready)
        self.assertTrue(e.arm(now=72.7)[0])
        feed(72.7,5,kind='intentional')
        self.assertTrue(any(k=='COMMAND' and 'EOG → FORWARD' in detail for _,k,detail in e.events))
        feed(77.7,10,hz=9)
        self.assertEqual(e.last_ssvep,9)
        self.assertEqual(e.last_ssvep_reason.startswith('ชัดเจน'),True)
        feed(87.7,10,hz=15)
        self.assertEqual(e.last_ssvep,15)
        self.assertTrue(any(k=='COMMAND' and 'RIGHT' in detail for _,k,detail in e.events))
        self.assertTrue(any(k=='COMMAND' and 'LEFT' in detail for _,k,detail in e.events))

if __name__=="__main__": unittest.main()
