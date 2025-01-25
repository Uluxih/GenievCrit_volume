import numpy, math
import GenCrit as gc
import numpy as np

tens = gc.Tensor(40, 25, 30, 20, 50, 60)
tensCor = tens.get_tens_cord()
val, vec = np.linalg.eig(tensCor)

t=tensCor
val, vec = np.linalg.eig(t)

print("собственные вектора, matr:")
print(vec)
print('значения собственных векторов:', val)

print('тензор до:')
print(t)

v1 = np.dot(t, vec.transpose()[0])
v2 = np.dot(t, vec.transpose()[1])
v3 = np.dot(t, vec.transpose()[2])
print('v1, длина до смены', v1, np.linalg.norm(v1))

v1 = np.dot(v1, vec)
v2 = np.dot(v2, vec)
v3 = np.dot(v3, vec)

print('v1, длина после смены', v1, np.linalg.norm(v1))

t = np.array([v1,v2,v3])
print("тензор на главных площадках ")
print(t)