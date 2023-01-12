# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 11:29:27 2022

@author: bmick
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# d)
#Constants
G= 6.67 * 10**(-11)
me=5.9736*(10**24)
ms=1.89*(10**30)
tstep = 100000
tmax = 34536000

'''tstep = 100
tmax = 31536000'''

def A(m_op, d):
    d1=d.copy()
    dmag=np.linalg.norm(d)    
    
    return d1*((m_op*G)/(dmag**3))

def wdot(t, q0):
    q=q0.copy()
    
    des= np.array([q0[0]-q0[2], q0[1]-q0[3]])
    
    Ae=A(ms, -des)
    As=A(me, des)
   
  
   
    
    q_2=np.array([q[4], q[5], q[6], q[7], Ae[0], Ae[1], As[0], As[1]])
    
    return q_2

def integral(F, t, q, tmax, tstep):

    def runkut(F, t, q, tstep):
        k0 = tstep * F(t, q)
        k1 = tstep * F(t + tstep/2.0, q + k0/2.0)
        k2 = tstep * F(t + tstep/2.0, q + k1/2.0)
        k3 = tstep * F(t + tstep, q + k2)
        return (k0 + 2.0 * k1 + 2.0 * k2 + k3)/6.0

    T = []
    Q = []
    T.append(t)
    Q.append(q)
    while t < tmax:
        tstep = min(tstep, tmax - t)
        q = q + runkut(F, t, q, tstep)
        t = t + tstep
        T.append(t)
        Q.append(q)
    return np.array(T), np.array(Q)

#Initital Conditions
"""
w = [r
 theta
 r dot
 theta dot]
"""
xe=1.467425829*(10**11)
ye=1.0120227668*(10**10)
xs=0.0
ys=0.0
vxe=-2083.5317978
vye=30211.0631877
vxs=0.0
vys=0.0



w0=np.array([xe, ye, xs, ys, vxe, vye, vxs, vys])
t0 = 0


T, q = integral(wdot, t0, w0, tmax, tstep)



fig = plt.figure()
ax = fig.add_subplot(aspect='equal')

eradius = 6378000000
sradius = 9957000000
sun = ax.add_patch(plt.Circle([xs , ys], sradius,
                      fc='r', zorder=3))

earth = ax.add_patch(plt.Circle([xe, ye], eradius,
                      fc='b', zorder=3))

line, = ax.plot(xe, ye, color='k')

ax.set_xlim(-3.0*(10**11), 2.467425829*(10**11))
ax.set_ylim(-2.467425829*(10**11), 2.467425829*(10**11))


def animate(i):
    """Update the animation at frame i."""
    #print('q[i,0]', q[i,0])
    #print('q[i,1]', q[i,1])
    
    sun.set_center((q[i,2], q[i,3]))
    earth.set_center((q[i,0], q[i,1]))
    line.set_data(i,i)#q[i,0], q[i,1])
    
    return line

nframes = q.shape[0]
print(q.shape)
interval = tstep * 1000
anim = animation.FuncAnimation(fig, animate, frames=nframes, repeat=True,
                              interval=interval)


plt.show()

f = 'Sun-Earth-Moon-Simulation.gif'
writergif = animation.PillowWriter(fps=60) 
anim.save(f, writer=writergif)
