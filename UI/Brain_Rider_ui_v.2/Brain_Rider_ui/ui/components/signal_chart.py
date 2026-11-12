"""pyqtgraph chart component used by calibration and monitor screens."""
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QWidget
from ui.styles.theme import CYAN


class SignalChart(QWidget):
    def __init__(self,title,color=CYAN,frequency=False,parent=None):
        super().__init__(parent)
        layout=QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(9)
        heading=QLabel(title)
        heading.setObjectName('eyebrow')
        layout.addWidget(heading)
        self.plot=pg.PlotWidget(background='#0B1726')
        self.plot.setMinimumHeight(145)
        self.plot.setMouseEnabled(x=False,y=False)
        self.plot.showGrid(x=True,y=True,alpha=.15)
        self.plot.getAxis('left').setTextPen('#7894A8')
        self.plot.getAxis('bottom').setTextPen('#7894A8')
        self.plot.setLabel('bottom','Frequency' if frequency else 'Time','Hz' if frequency else 's')
        self.plot.setLabel('left','PSD' if frequency else 'Amplitude')
        self.plot.getPlotItem().setMenuEnabled(False)
        self.curve=self.plot.plot(pen=pg.mkPen(color,width=1.7))
        self.plot.getPlotItem().setClipToView(True)
        layout.addWidget(self.plot,1)
    def update_series(self, values, fs=200, frequency=False, x_values=None):
        y=np.asarray(values,float)
        if len(y)==0:
            self.curve.setData([],[])
            return
        if x_values is None:
            x=np.arange(len(y))/fs
        else:
            x=np.asarray(x_values,float)
        self.curve.setData(x,y)
