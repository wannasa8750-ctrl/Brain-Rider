"""Animated vector brain concept for the UI; does not indicate electrode accuracy.
No images/assets/network calls required. O1/Fp1 marker positions are illustrative.
"""
import math
import random
from PyQt5.QtCore import Qt,QPointF,QTimer
from PyQt5.QtGui import (QPainter,QPen,QColor,QPainterPath,QRadialGradient,QFont)
from PyQt5.QtWidgets import QWidget


class BrainVisual(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setMinimumSize(320,270)
        self.t=0.0
        self._timer=QTimer(self)
        self._timer.timeout.connect(self.animate)
        self._timer.start(65)
        rng=random.Random(17)
        self.nodes=[(rng.uniform(-.48,.48),rng.uniform(-.48,.46),rng.uniform(1.2,3.0)) for _ in range(50)]

    def animate(self):
        if self.isVisible():
            self.t+=.06
            self.update()

    def paintEvent(self,event):
        p=QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w,h=self.width(),self.height()
        scale=min(w/500.,h/320.)
        p.translate(w/2.,h/2.)
        p.scale(scale,scale)
        # orbital rings, centered behind the illustrative brain
        p.setBrush(Qt.NoBrush)
        for i,rx in enumerate((105,136,168)):
            ring=QColor('#315F73');ring.setAlpha(38 if i<2 else 22)
            p.setPen(QPen(ring,1.3))
            p.drawEllipse(QPointF(0,-3),rx,rx*.86)
        contour=QPainterPath()
        contour.moveTo(-3,111)
        contour.cubicTo(-39,128,-73,107,-86,89)
        contour.cubicTo(-125,85,-147,51,-144,21)
        contour.cubicTo(-160,-7,-141,-46,-119,-50)
        contour.cubicTo(-126,-82,-91,-113,-60,-103)
        contour.cubicTo(-43,-125,-11,-124,-2,-113)
        contour.cubicTo(16,-132,48,-116,63,-104)
        contour.cubicTo(101,-117,132,-77,126,-51)
        contour.cubicTo(151,-29,158,17,138,37)
        contour.cubicTo(143,70,113,93,86,89)
        contour.cubicTo(56,120,27,127,-3,111)
        glow=QRadialGradient(QPointF(0,0),172)
        glow.setColorAt(0,QColor(29,150,162,68))
        glow.setColorAt(.6,QColor(14,63,96,94))
        glow.setColorAt(1,QColor(9,30,59,130))
        p.fillPath(contour,glow)
        p.setPen(QPen(QColor('#50D8D2'),2.4))
        p.drawPath(contour)
        # center separation: two hemispheres
        stem=QPainterPath();stem.moveTo(0,-111)
        stem.cubicTo(-16,-62,15,-35,0,3)
        stem.cubicTo(-12,43,8,79,-3,108)
        p.setPen(QPen(QColor(71,177,181,135),2.0));p.drawPath(stem)
        # abstract sulci — rendered lines are intentionally not anatomical EEG localization
        for side in (-1,1):
            for j in range(7):
                yy=-79+j*26
                curve=QPainterPath()
                curve.moveTo(side*(21+3*(j%2)),yy)
                curve.cubicTo(side*(54+(j%3)*13),yy-20,side*(88-(j%3)*7),yy+30,side*(117-(j%4)*5),yy+5)
                p.setPen(QPen(QColor(67,179,181,75+(j%3)*24),1.6))
                p.drawPath(curve)
        for xx,yy,r in self.nodes:
            x=xx*260;y=yy*210
            if not contour.contains(QPointF(x,y)):
                continue
            bright=int(90+60*(1+math.sin(self.t+xx*9+yy*5))/2)
            p.setPen(Qt.NoPen);p.setBrush(QColor(106,234,218,bright))
            p.drawEllipse(QPointF(x,y),r,r)
        self.draw_electrode(p,-94,-68,'Fp1',QColor('#B2A7FF'),math.sin(self.t*2))
        self.draw_electrode(p,97,76,'O1',QColor('#66F0C3'),math.sin(self.t*1.5+1.0))
        p.end()

    def draw_electrode(self,p,x,y,text,color,pulse):
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(color.red(),color.green(),color.blue(),38))
        p.drawEllipse(QPointF(x,y),18+3*pulse,18+3*pulse)
        p.setBrush(QColor(color.red(),color.green(),color.blue(),85))
        p.drawEllipse(QPointF(x,y),10,10)
        p.setBrush(color)
        p.drawEllipse(QPointF(x,y),5.1,5.1)
        p.setPen(QPen(color,1.1))
        ex=x-36 if x<0 else x+37
        ey=y-32 if y<0 else y+32
        p.drawLine(QPointF(x,y),QPointF(ex,ey))
        p.setFont(QFont('Segoe UI',12,QFont.Bold))
        p.drawText(QPointF(ex-16 if x<0 else ex+5,ey-6),text)
