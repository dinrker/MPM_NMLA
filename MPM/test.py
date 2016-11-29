import numpy as np
import Matrix_Pencil_Method


## Test
x = np.zeros((11,1))
for i in range(11):
    x[i, 0] = i*1.0
y = np.exp(1j*10*np.pi*x/100)
ex_pole = np.exp(1j*10*np.pi/100)
num_pole = Matrix_Pencil_Method.Matrix_Pencil(y)
print '\nThe exact pole:', [ex_pole], '\n'
print 'The pole found by Matrix Pencil Method:', num_pole, '\n'
print 'Relative error:', abs(ex_pole - num_pole[0])/abs(ex_pole)