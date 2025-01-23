l1x, l2x, l3x = 0.92, 0.3, -0.24
l1y, l2y, l3y = -0.11, 0.81, 0.57
l1z, l2z, l3z = 0.37, -0.5, 0.78

import numpy, math
import GenCrit as Gc
import numpy as np

n=np.array((0.5773502,0.5773502,0.5773502))
lmn=n
tens = Gc.Tensor(95,49,287,0,0,0)
a = tens.get_shear(n)
len_a = np.linalg.norm(a)
tau = math.sqrt(
            ((tens.sigma1 - tens.sigma2) * lmn[0] * lmn[1]) ** 2 + (
                    (tens.sigma3 - tens.sigma2) * lmn[1] * lmn[2]) ** 2
            + ((tens.sigma1 - tens.sigma3) * lmn[0] * lmn[2]) ** 2)
sigma = np.linalg.norm(tens.get_sigma(lmn))
print(a, len_a)
print(tau, sigma)
P_n = np.dot(tens.get_tens_cord(), n)
print(np.linalg.norm(P_n))
print(math.sqrt(tau**2+sigma**2))
L = np.array(((l1x, l2x, l3x), (l1y, l2y, l3y), (l1z, l2z, l3z)))
l_mx = Gc.lm_xyz(lmn[0], lmn[1], lmn[2], L[0][0], L[0][1], L[0][2])
print(l_mx)
l_mx = np.dot(L[0], lmn)
print(l_mx)