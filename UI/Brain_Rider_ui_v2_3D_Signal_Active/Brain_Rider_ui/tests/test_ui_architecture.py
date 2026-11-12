"""Headless UI structure checks (PyQt5 GUI test requires Windows env)."""
import ast
import hashlib
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

class UIArchitectureTests(unittest.TestCase):
    def test_four_page_components_exist(self):
        for page in ('overview','calibration','monitor','logs'):
            self.assertTrue((ROOT/'ui'/'pages'/f'{page}.py').is_file())
        for comp in ('common','brain_visual','signal_chart'):
            self.assertTrue((ROOT/'ui'/'components'/f'{comp}.py').is_file())

    def test_launcher_preserves_cli_arguments(self):
        src=(ROOT/'main.py').read_text(encoding='utf-8')
        for flag in ('--mode','--port','--eog-ch','--ssvep-ch','--keyboard','--legacy-ssvep-row2'):
            self.assertIn(flag,src)

    def test_all_ui_sources_are_syntactically_valid(self):
        for path in (ROOT/'ui').rglob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'),str(path))

    def test_ui_does_not_duplicate_signal_processing(self):
        for path in (ROOT/'ui'/'components').glob('*.py'):
            source=path.read_text(encoding='utf-8')
            for name in ('PersonalBlinkModel','SSVEPDetector','CommandGate','extract_events'):
                self.assertNotIn(name,source)

    def test_original_modules_unchanged_from_previous_deliverable(self):
        prior=Path('/mnt/data/LightBrain_Hybrid_OriginalLogic')
        if not prior.exists(): self.skipTest('Previous deliverable path unavailable')
        for name in ('eog_original.py','ssvep_original.py','hybrid_engine.py',
                     'acquisition.py','desktop_output.py'):
            def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest(prior/name),digest(ROOT/name))
