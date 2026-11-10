"""Real-time readout and SAFE desktop-simulator controls."""
from PyQt5.QtWidgets import QWidget,QVBoxLayout,QHBoxLayout,QGridLayout,QCheckBox
from ui.components.common import Panel,StatusPill,label,button
from ui.components.signal_chart import SignalChart
from ui.styles.theme import MINT,CYAN,PURPLE,ORANGE,RED


class MonitorPage(QWidget):
    def __init__(self,on_stop,on_reset,on_arm,on_keyboard,on_target,on_blink,mode='demo'):
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
        for name,color in [('FORWARD',MINT),('LEFT 9 Hz',CYAN),('RIGHT 15 Hz',PURPLE)]:
            tile=Panel(name,'Original command gate','panelAlt',15)
            active=label('●  IDLE','eyebrow')
            active.setStyleSheet('color:'+color+';')
            tile.layout_box.addWidget(active)
            setattr(self,'tag_'+name.split()[0].lower(),active)
            bar.addWidget(tile,1)
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
        self.charts['psd']=SignalChart('WELCH PSD  /  9 Hz + 15 Hz',CYAN,frequency=True)
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
        self.psda=label('9 Hz z: —     15 Hz z: —','eyebrow')
        self.details=label('EOG classifier: —\nPSDA: —\nEpochs: 0','muted',True)
        self.distance=label('Simulated distance: 0.00 m','eyebrow')
        tele.layout_box.addWidget(self.psda)
        tele.layout_box.addWidget(self.details)
        tele.layout_box.addWidget(self.distance)
        side.addWidget(tele)
        demo=Panel('DEMO INJECTION','For synthetic-signal tests only; hidden/disabled on live hardware.')
        self.demo_panel=demo
        drow=QHBoxLayout()
        for title,hz in [('NONE',0),('9 Hz',9),('15 Hz',15)]:
            drow.addWidget(button(title,lambda checked=False,f=hz:on_target(f),'ghost'))
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
        self.psda.setText('9 Hz z={:+.2f}    15 Hz z={:+.2f}'.format(engine.last_z.get(9,0),engine.last_z.get(15,0)))
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
