import numpy as np
import scipy.special as sp
import math
from NMLA import NMLA


## Test

# parameters
x0, y0 = 0.0, 0.0              # observation point
c0 = 1.0                     # wavespeed at observation point
omega = 1000.0               # frequency
Rest = 3.0                   # estimate of the distance from source to oberservation point

# Add source 
xs, ys = 2, 2
rxy = lambda x, y: np.sqrt((x-xs)**2 + (y-ys)**2)
u = lambda x, y, omega: 1j/4*np.sqrt(omega)*sp.hankel1(0, omega*rxy(x,y))
ux = lambda x, y, omega: -1j/4*(omega**1.5)*sp.hankel1(1,omega*rxy(x,y))*(x-xs)/rxy(x,y)
uy = lambda x, y, omega: -1j/4*(omega**1.5)*sp.hankel1(1,omega*rxy(x,y))*(y-ys)/rxy(x,y)

# numerical and exact ray direction 
est_ang = NMLA(x0,y0,c0,omega,Rest,u,ux,uy)  
ex_ang = math.atan2(y0-ys,x0-xs) + 2*np.pi

# print result
print '\nThe exact ray direction:', ex_ang/np.pi, r'$\pi$', '\n'
print 'The ray direction estimated ny NMLA:', est_ang/np.pi, r'$\pi$', '\n'
print 'Relative error:', abs(est_ang - ex_ang)/abs(ex_ang)


# Compared to MPM, NMLA is more stable, it allows much more noisy level to the source, but MPM is very sensitive to the noise.
