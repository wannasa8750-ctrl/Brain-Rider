"""Real-time readout and SAFE desktop-simulator controls."""
from PyQt5.QtWidgets import QWidget,QVBoxLayout,QHBoxLayout,QGridLayout,QCheckBox,QComboBox,QProgressBar
from ssvep_original import TARGET_FREQUENCIES
from ui.components.common import Panel,StatusPill,label,button
from ui.components.signal_chart import SignalChart
from ui.styles.theme import MINT,CYAN,PURPLE,ORANGE,RED


class MonitorPage(QWidget):
    def __init__(self,on_stop,on_reset,on_arm,on_keyboard,on_target,on_blink,on_turns,mode='demo'):
        super().__init__()
        root=QVBoxLayout(self);root.setContentsMargins(2,4,2,10);root.setSpacing(15)
        root.addWidget(label('REAL-TIME NEURAL TELEMETRY','eyebrow'))
        root.addWidget(label('Live Control Monitor','largeTitle'))
        bar=QHBoxLayout();bar.setSpacing(12)
        control=Panel('DETECTED COMMAND','The application defaults to STOP unless explicitly armed.','hero',19)
        self.motion=label('■  STOP','commandValue')
        self.motion.setStyleSheet('color:'+RED+';')
        control.layout_box.addWidget(self.motion)
        self.gate_status=label('EOG: READY    •    SSVEP: IDLE','muted')
        control.layout_box.addWidget(self.gate_status)
        bar.addWidget(control,3)
        for name,color in [('FORWARD',MINT),('LEFT',CYAN),('RIGHT',PURPLE)]:
            tile=Panel(name,'Original command gate','panelAlt',15)
            active=label('●  IDLE','eyebrow')
            active.setStyleSheet('color:'+color+';')
            tile.layout_box.addWidget(active)
            setattr(self,'tag_'+name.split()[0].lower(),active)
            if name in ('LEFT', 'RIGHT'):
                combo = QComboBox()
                for hz in TARGET_FREQUENCIES:
                    combo.addItem(f'{hz} Hz', hz)
                combo.setCurrentIndex(combo.findData(9 if name == 'LEFT' else 15))
                setattr(self, name.lower() + '_hz', combo)
                tile.layout_box.addWidget(combo)
            bar.addWidget(tile,1)
        self.on_turns = on_turns
        self.left_hz.currentIndexChanged.connect(self.update_turns)
        self.right_hz.currentIndexChanged.connect(self.update_turns)
        self.update_turns()
        root.addLayout(bar)
        content=QHBoxLayout();content.setSpacing(14)
        signals=Panel('SIGNAL EXPLORER','Live visuals are separate from the unchanged signal-processing pipeline.',padding=17)
        plots=QGridLayout();plots.setSpacing(12)
        self.charts={}
        for index,(key,desc,color) in enumerate([
            ('eog_raw','FP1  /  RAW EOG',PURPLE),
            ('eog_filt','FP1  /  FILTERED EOG',MINT),
            ('ss_raw','O1  /  RAW EEG',ORANGE),
            ('ss_filt','O1  /  FILTERED EEG',CYAN)]):
            c=SignalChart(desc,color)
            self.charts[key]=c
            plots.addWidget(c,index//2,index%2)
        signals.layout_box.addLayout(plots,3)
        self.charts['psd']=SignalChart('WELCH PSD  /  O1',CYAN,frequency=True)
        signals.layout_box.addWidget(self.charts['psd'],2)
        content.addWidget(signals,7)
        side=QVBoxLayout();side.setSpacing(12)
        safety=Panel('SAFETY CENTER','STOP overrides every command. Physical wheelchair outputs are not implemented.')
        self.stop=button('■  EMERGENCY STOP  [SPACE]',on_stop,'danger')
        self.reset=button('RESET E-STOP',on_reset,'ghost')
        self.arm=button('ARM SIMULATOR',on_arm,'primary')
        safety.layout_box.addWidget(self.stop)
        safety.layout_box.addWidget(self.reset)
        safety.layout_box.addWidget(self.arm)
        self.keyboard=QCheckBox('Desktop STIMULATOR keys (W/A/D)')
        self.keyboard.toggled.connect(on_keyboard)
        safety.layout_box.addWidget(self.keyboard)
        safety.layout_box.addWidget(label('Keyboard output is Windows-only, optional and NOT suitable for motor actuation.','muted',True))
        side.addWidget(safety)
        tele=Panel('DETECTION TELEMETRY')
        self.levels = {}
        for hz in TARGET_FREQUENCIES:
            meter = QProgressBar()
            meter.setRange(0, 100)
            meter.setFormat(f'{hz} Hz')
            tele.layout_box.addWidget(meter)
            self.levels[hz] = meter
        self.details=label('EOG classifier: —\nPSDA: —\nEpochs: 0','muted',True)
        self.distance=label('Simulated distance: 0.00 m','eyebrow')
        tele.layout_box.addWidget(self.details)
        tele.layout_box.addWidget(self.distance)
        side.addWidget(tele)
        demo=Panel('DEMO INJECTION','For synthetic-signal tests only; hidden/disabled on live hardware.')
        self.demo_panel=demo
        drow=QHBoxLayout()
        target = QComboBox()
        target.addItem('NONE', 0)
        for hz in TARGET_FREQUENCIES:
            target.addItem(f'{hz} Hz', hz)
        target.currentIndexChanged.connect(lambda _: on_target(target.currentData()))
        drow.addWidget(target)
        demo.layout_box.addLayout(drow)
        brow=QHBoxLayout()
        for title,kind in [('NATURAL','natural'),('INTENT','intentional'),('ARTIFACT','artifact')]:
            brow.addWidget(button(title,lambda checked=False,k=kind:on_blink(k),'ghost'))
        demo.layout_box.addLayout(brow)
        side.addWidget(demo)
        side.addStretch()
        content.addLayout(side,3)
        root.addLayout(content,1)
        self.set_demo_mode(mode=='demo')

    def update_turns(self, *_):
        left, right = self.left_hz.currentData(), self.right_hz.currentData()
        for combo, excluded in ((self.left_hz, right), (self.right_hz, left)):
            for i in range(combo.count()):
                combo.model().item(i).setEnabled(combo.itemData(i) != excluded)
        self.on_turns(left, right)

    def set_demo_mode(self,value):
        self.demo_panel.setVisible(bool(value))

    def render(self,engine,connected):
        m=engine.motion
        self.motion.setText({'stop':'■  STOP','forward':'↑  FORWARD','left':'↶  TURN LEFT','right':'↷  TURN RIGHT'}.get(m,'■  STOP'))
        self.motion.setStyleSheet('color:'+({'stop':RED,'forward':MINT,'left':CYAN,'right':PURPLE}.get(m,RED))+';')
        self.gate_status.setText('EOG: '+engine.eog_gate.state+'    •    SSVEP: '+engine.ss_gate.state+'    •    '+('ARMED' if engine.armed else 'DISARMED'))
        for cmd in ('forward','left','right'):
            tag=getattr(self,'tag_'+cmd)
            tag.setText('●  ACTIVE' if m==cmd else '●  IDLE')
        for combo in (self.left_hz, self.right_hz):
            combo.setEnabled(not engine.armed)
        for hz, meter in self.levels.items():
            role = engine.turns.get(hz, 'monitor')
            valid = connected and engine.armed and engine.levels_ready
            z = engine.last_z.get(hz, 0) if valid else 0
            meter.setValue(int(max(0, min(100, z * 100 / 6))))
            value = f'z {z:+.2f}' if valid else '—'
            meter.setFormat(f'{hz} Hz   {role.upper()}   {value}')
        e=engine.last_eog_features
        self.details.setText('EOG: '+engine.last_eog_class+
            (f'  |  amplitude={e["amplitude"]:.0f} µV  NCC={e["corr"]:.2f}' if e else '')+
            '\nPSDA: '+engine.last_ssvep_reason+
            f'\nSSVEP epochs: {engine.ss_epoch_count}  |  Rejected: {engine.ss_rejected_count}'+
            f'\nBlink templates: intentional {len(engine.intentional_events)}, natural {len(engine.natural_events)}')
        self.distance.setText(f'Virtual forward distance: {engine.virtual.position:.2f} / {engine.virtual.limit:.2f} m')
        self.arm.setText('DISARM' if engine.armed else 'ARM SIMULATOR')
        self.arm.setEnabled(connected and (engine.armed or engine.validated and not engine.estop))
        self.reset.setEnabled(engine.estop)
