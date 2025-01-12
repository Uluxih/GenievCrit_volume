l1x, l2x, l3x = 0.92, 0.3, -0.24
l1y, l2y, l3y = -0.11, 0.81, 0.57
l1z, l2z, l3z = 0.37, -0.5, 0.78

import numpy, math

matr=numpy.array([[0.92, 0.3, -0.24], [-0.11, 0.81, 0.57], [0.37, -0.5, 0.78]])
a=numpy.array([1,1,1])
b=numpy.dot(matr,a)
print(b)
b2=numpy.array([[0.98], [1.27], [0.65]])
b1=numpy.array([0.98, 1.27, 0.65])
a2=numpy.dot(matr.transpose(), b1)[0]
print(a2)
print(math.degrees(math.atan(90)))