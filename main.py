import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp
import math


def Rot2D(X, Y, Alpha):
    RX = X * np.cos(Alpha) - Y * np.sin(Alpha)
    RY = X * np.sin(Alpha) + Y * np.cos(Alpha)
    return RX, RY


ArConst = 1.2
R1 = 4
t = sp.Symbol('t')
r = 2 + sp.sin(6 * t)
phi = 7 * t + 1.2 * sp.cos(6 * t)

x = r * sp.cos(phi)
y = r * sp.sin(phi)
Vx = sp.diff(x, t)
Vy = sp.diff(y, t)
V = sp.sqrt(Vx ** 2 + Vy ** 2)
At = sp.diff(V, t)

Ax = sp.diff(Vx, t)
Ay = sp.diff(Vy, t)

An = sp.sqrt(Ax ** 2 + Ay ** 2 - At ** 2)  # A ** 2 = Ax ** 2 + Ay ** 2
R = (V ** 2) / An

T = np.linspace(0, 6.29, 1000)
X = np.zeros_like(T)
Y = np.zeros_like(T)
VX = np.zeros_like(T)
VY = np.zeros_like(T)
AX = np.zeros_like(T)
AY = np.zeros_like(T)
Rr = np.zeros_like(T)

for i in np.arange(len(T)):
    X[i] = sp.Subs(x, t, T[i])
    Y[i] = sp.Subs(y, t, T[i])
    VX[i] = sp.Subs(Vx, t, T[i])
    VY[i] = sp.Subs(Vy, t, T[i])
    AX[i] = sp.Subs(Ax, t, T[i])
    AY[i] = sp.Subs(Ay, t, T[i])
    Rr[i] = sp.Subs(R, t, T[i])

fig = plt.figure()

ax1 = fig.add_subplot(1, 1, 1)
ax1.axis('equal')
ax1.set(xlim=[-R1, R1], ylim=[-R1, R1])

ax1.plot([-R1 * 2, R1 * 2], [0, 0], 'c')
ax1.plot([0, 0], [-R1 * 2, R1 * 2], 'c')

ax1.plot(X, Y, color='royalblue')

P, = ax1.plot(X[0], Y[0], 'b', marker='o')
rLine, = ax1.plot([0, X[0]], [0, Y[0]], 'b')
VLine, = ax1.plot([X[0], X[0] + VX[0]], [Y[0], Y[0] + VY[0]], 'r')
ALine, = ax1.plot([X[0], X[0] + AX[0]], [Y[0], Y[0] + AY[0]], 'g')

ArrowX = np.array([-0.3 * ArConst, 0, -0.3 * ArConst])
ArrowY = np.array([0.1 * ArConst, 0, -0.1 * ArConst])

RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[0], VX[0]))
VArrow, = ax1.plot(RArrowX + X[0] + VX[0], RArrowY + Y[0] + VY[0], 'r')

RAArrowX, RAArrowY = Rot2D(ArrowX, ArrowY, math.atan2(AY[0], AX[0]))
AArrow, = ax1.plot(RAArrowX + X[0] + AX[0], RAArrowY + Y[0] + AY[0], 'g')

RrArrowX, RrArrowY = Rot2D(ArrowX, ArrowY, math.atan2(Y[0], X[0]))
rArrow, = ax1.plot(RrArrowX + X[0], RrArrowY + Y[0], 'b')

RadX, RadY = Rot2D(Rr[0], 0, math.atan2(VY[0], VX[0]) + 3.141592 / 2)
RLine, = ax1.plot([X[0],X[0] + RadX], [Y[0], Y[0] + RadY], 'm')

Vconstdel = 4
Aconstdel = 50


def anima(i):
    P.set_data(X[i], Y[i])

    VLine.set_data([X[i], X[i] + VX[i] / Vconstdel], [Y[i], Y[i] + VY[i] / Vconstdel])
    RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[i], VX[i]))
    VArrow.set_data((RArrowX + X[i] + VX[i] / Vconstdel), (RArrowY + Y[i] + VY[i] / Vconstdel))

    rLine.set_data([0, X[i]], [0, Y[i]])
    RrArrowX, RrArrowY = Rot2D(ArrowX, ArrowY, math.atan2(Y[i], X[i]))
    rArrow.set_data(RrArrowX + X[i], RrArrowY + Y[i])

    ALine.set_data([X[i], X[i] + AX[i] / Aconstdel], [Y[i], Y[i] + AY[i] / Aconstdel])
    RAArrowX, RAArrowY = Rot2D(ArrowX, ArrowY, math.atan2(AY[i], AX[i]))
    AArrow.set_data(RAArrowX + X[i] + AX[i] / Aconstdel, RAArrowY + Y[i] + AY[i] / Aconstdel)

    RadX, RadY = Rot2D(Rr[i], 0, math.atan2(VY[i], VX[i]) + 3.141592 / 2)
    RLine.set_data([X[i], X[i] + RadX], [Y[i], Y[i] + RadY])

    return P, VLine, VArrow, ALine, AArrow, rLine, rArrow, RLine,


anim = FuncAnimation(fig, anima, frames=1000, interval=5, blit=True)

plt.show()
