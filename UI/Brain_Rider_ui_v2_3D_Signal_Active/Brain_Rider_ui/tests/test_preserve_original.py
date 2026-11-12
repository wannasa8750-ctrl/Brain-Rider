"""Audit that algorithm ASTs match exact uploaded sources (not just similar formulas)."""
import ast
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]

def node_at(tree,name,kind=ast.ClassDef):
    for n in tree.body:
        if isinstance(n,kind) and n.name==name:return n
    raise AssertionError(f'Missing {name}')

def normal(node):
    return ast.dump(node,include_attributes=False)

class OriginalCodePreservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e0=ast.parse((ROOT/'original_sources/eog_lightbrain_STIMULATOR_SINGLE_BLINK.py').read_text(encoding='utf-8'))
        cls.e1=ast.parse((ROOT/'eog_original.py').read_text(encoding='utf-8'))
        cls.s0=ast.parse((ROOT/'original_sources/stimulator_control_two_commands.py').read_text(encoding='utf-8'))
        cls.s1=ast.parse((ROOT/'ssvep_original.py').read_text(encoding='utf-8'))

    def test_original_eog_signal_functions_identical(self):
        for name in ('robust_median_mad','natural_reference','waveform','extract_events','_ln_features','_corr'):
            with self.subTest(function=name):
                self.assertEqual(normal(node_at(self.e0,name,ast.FunctionDef)),
                                 normal(node_at(self.e1,name,ast.FunctionDef)))

    def test_original_eog_classifiers_filters_gates_identical(self):
        for name in ('PersonalBlinkModel','SingleBlinkGate','StreamingFilter','VirtualWheelchair','GateResult'):
            with self.subTest(class_name=name):
                self.assertEqual(normal(node_at(self.e0,name)),normal(node_at(self.e1,name)))

    def test_original_ssvep_processors_detectors_gates_identical(self):
        for name in ('SignalProcessor','SSVEPDetector','DecisionFilter','CommandGate','GateResult'):
            with self.subTest(class_name=name):
                self.assertEqual(normal(node_at(self.s0,name)),normal(node_at(self.s1,name)))

    def test_artifact_rejection_method_identical(self):
        method0=node_at(self.s0,'MainWindow')
        method0=next(n for n in method0.body if isinstance(n,ast.FunctionDef) and n.name=='_artifact_reason')
        method1=node_at(self.s1,'SSVEPArtifactMixin')
        method1=next(n for n in method1.body if isinstance(n,ast.FunctionDef) and n.name=='_artifact_reason')
        self.assertEqual(normal(method0),normal(method1))

    def test_frequency_mapping_original(self):
        import ssvep_original as S
        self.assertEqual(S.SSVEP_CONFIG[9]['action'],'left')
        self.assertEqual(S.SSVEP_CONFIG[15]['action'],'right')
        self.assertEqual(S.EPOCH_SEC,5.0)
        self.assertEqual(S.DECISION_STEP_SEC,5.0)
        self.assertEqual((S.PSD_NPERSEG,S.PSD_NOVERLAP),(800,700))
        self.assertTrue(S.PSDA_MODE)
        import eog_original as E
        self.assertEqual(E.CALIBRATION_SECONDS,20)
        self.assertEqual(E.REFRACTORY_SECONDS,.38)
        self.assertEqual(E.StreamingFilter(200).sos.shape[1],6)

if __name__=='__main__':unittest.main()
