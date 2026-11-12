"""Sequential baseline acquisition, original durations, and validation controls."""
from PyQt5.QtWidgets import QWidget,QVBoxLayout,QHBoxLayout,QGridLayout,QComboBox,QProgressBar
from ui.components.common import Panel,StepCard,label,button
from ui.components.signal_chart import SignalChart
from ui.styles.theme import MINT,CYAN,PURPLE


class CalibrationPage(QWidget):
    def __init__(self,on_phase,on_validate,on_arm,on_clear,on_sensitivity,on_rest):
        super().__init__()
        outer=QVBoxLayout(self)
        outer.setContentsMargins(2,4,2,10);outer.setSpacing(17)
        outer.addWidget(label('PERSONALIZED SIGNAL BASELINES','eyebrow'))
        outer.addWidget(label('Calibration Studio','largeTitle'))
        outer.addWidget(label('Three ordered baselines. Original SSVEP null 30 s, natural EOG 20 s, and intentional EOG 20 s.','muted',True))
        main=QHBoxLayout();main.setSpacing(17)
        left=QVBoxLayout();left.setSpacing(14)
        self.steps={}
        definitions=[('ssvep',1,'SSVEP Null Reference','Look away from flickering targets. Build PSD + null z-score reference.','30 SECONDS'),
                     ('natural',2,'Natural Blink Reference','Relax and blink normally. Background EOG noise reference.','20 SECONDS'),
                     ('intentional',3,'Intentional Blink Training','Blink intentionally several times to build a personal model.','20 SECONDS')]
        for kind,n,title,desc,dur in definitions:
            step=StepCard(n,title,desc,dur,lambda checked=False,k=kind:on_phase(k))
            left.addWidget(step)
            self.steps[kind]=step
        prog=Panel('Recording progress')
        self.progress=QProgressBar();self.progress.setRange(0,100);prog.layout_box.addWidget(self.progress)
        self.progress_label=label('Waiting for calibration','muted');prog.layout_box.addWidget(self.progress_label)
        left.addWidget(prog)
        left.addStretch()
        main.addLayout(left,6)
        right=QVBoxLayout();right.setSpacing(14)
        preview=Panel('Live acquisition preview','Visual display only. Feature extraction uses the unchanged original filters.')
        self.eog=SignalChart('Fp1  /  FILTERED EOG',MINT)
        self.eeg=SignalChart('O1  /  FILTERED EEG',CYAN)
        preview.layout_box.addWidget(self.eog)
        preview.layout_box.addWidget(self.eeg)
        right.addWidget(preview,4)
        actions=Panel('Validation & activation')
        sens=QHBoxLayout();sens.addWidget(label('EOG SENSITIVITY','eyebrow'))
        self.sensitivity=QComboBox()
        self.sensitivity.addItem('Balanced · 3.0',3.0)
        self.sensitivity.addItem('Sensitive · 2.0',2.0)
        self.sensitivity.addItem('Strict · 4.5',4.5)
        self.sensitivity.currentIndexChanged.connect(lambda:on_sensitivity(self.sensitivity.currentData()))
        sens.addWidget(self.sensitivity,1)
        actions.layout_box.addLayout(sens)
        row=QHBoxLayout()
        self.validate=button('✓  VALIDATE',on_validate,'ghost')
        self.arm=button('ARM SIMULATOR  →',on_arm,'primary')
        row.addWidget(self.validate);row.addWidget(self.arm)
        actions.layout_box.addLayout(row)
        row2=QHBoxLayout()
        self.rest=button('REST TEST 60s',on_rest,'ghost')
        self.clear=button('CLEAR ALL BASELINES',on_clear,'ghost')
        row2.addWidget(self.rest);row2.addWidget(self.clear)
        actions.layout_box.addLayout(row2)
        self.feedback=label('Validate after all three baselines have completed.','muted',True)
        actions.layout_box.addWidget(self.feedback)
        right.addWidget(actions,3)
        main.addLayout(right,5)
        outer.addLayout(main,1)

    def render(self,engine,connected):
        ready={'ssvep':engine.ss_ready,
               'natural':engine.calibration['natural'] is not None,
               'intentional':engine.calibration['intentional'] is not None}
        for kind,step in self.steps.items():
            recording=engine.phase==kind
            step.set_state('● RECORDING' if recording else '✓ READY' if ready[kind] else 'NOT STARTED',ready[kind],recording)
            allowed=(kind=='ssvep' or kind=='natural' and ready['ssvep'] or kind=='intentional' and ready['natural'])
            step.record.setEnabled(connected and not engine.estop and engine.phase is None and allowed)
        self.progress.setValue(int(engine.phase_progress()*100))
        self.progress_label.setText('Recording '+engine.phase.upper() if engine.phase else '✓ All three recordings complete' if all(ready.values()) else 'Complete each baseline in order')
        self.arm.setText('DISARM' if engine.armed else 'ARM SIMULATOR  →')
        self.arm.setEnabled(connected and (engine.armed or engine.validated and not engine.estop))
        self.validate.setEnabled(engine.phase is None)
        self.sensitivity.setEnabled(engine.phase is None)
        self.rest.setEnabled(bool(engine.model and engine.model.ready) and not engine.rest_test and engine.phase is None)
