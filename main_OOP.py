import math
import numpy as np
import matplotlib.pyplot as plt


class Tensor:
    def __init__(self, sigma1, sigma2, sigma3):
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.sigma3 = sigma3


class Main_strength:
    def __init__(self, Cx, Cy, Cz, Rx, Ry, Rz, k):
        self.Cx = Cx
        self.Cy = Cy
        self.Cz = Cz
        self.Rx = Rx
        self.Ry = Ry
        self.Rz = Rz
        self.k = k


class Plate:
    def __init__(self, main_strength: Main_strength, tensor: Tensor, lmn: [3], L: ((3), (3), (3))):
        self.main_strength = main_strength
        self.tensor = tensor
        self.lmn = lmn
        self.L = L
        l_mx = lm_xyz(lmn[0], lmn[1], lmn[2], L[0][0], L[0][1], L[0][2])
        l_my = lm_xyz(lmn[0], lmn[1], lmn[2], L[1][0], L[1][1], L[1][2])
        l_mz = lm_xyz(lmn[0], lmn[1], lmn[2], L[2][0], L[2][1], L[2][2])
        self.C_m = main_strength.Cx * l_mx ** 2 + main_strength.Cy * l_my ** 2 + main_strength.Cz * l_mz ** 2
        self.R_m = main_strength.Rx * l_mx ** 2 + main_strength.Ry * l_my ** 2 + main_strength.Rz * l_mz ** 2
        self.tau = math.sqrt(
            ((tensor.sigma1 - tensor.sigma2) * lmn[0] * lmn[1]) ** 2 + (
                        (tensor.sigma3 - tensor.sigma2) * lmn[1] * lmn[2]) ** 2
            + ((tensor.sigma1 - tensor.sigma3) * lmn[0] * lmn[2]) ** 2)
        self.sigma = tensor.sigma1 * lmn[0] ** 2 + tensor.sigma2 * lmn[1] ** 2 + tensor.sigma3 * lmn[2] ** 2
        # if (self.sigma != 0):
        print(self.tau, self.C_m, main_strength.k * self.sigma, self.tau - self.C_m - main_strength.k * self.sigma)
        if self.tau > self.C_m + main_strength.k * self.sigma:
            self.danger_plate = (True,
                                 (self.tau) /
                                 (abs(self.C_m + main_strength.k * self.sigma)+0.001))
        else:
            self.danger_plate = (False,
                                 (self.tau) /
                                 (abs(self.C_m + main_strength.k * self.sigma)+0.001))


class Stress_point:
    def __init__(self, tensor: Tensor, main_strength: Main_strength, L: tuple):
        self.tensor = tensor
        self.plates = []
        self.L = L
        self.main_strength=main_strength
        self.danger_plates = []
    def get_plates(self, lmn_arr: list()):
        for i in lmn:
            plate = Plate(self.main_strength, self.tensor, i, self.L)
            self.plates.append(plate)
            if plate.danger_plate[0]:
                self.danger_plates.append(plate)


def lm_xyz(l, m, n, l1, l2, l3):
    return (l * l1 + m * l2 + n * l3)


def get_crit_result(tau, Cm):
    return tau < Ccm


def get_lmn(step):
    lmn = []
    l = 0
    m = 0
    n = 0
    while (l < 1):
        m = 0
        while (m < 1):
            n = 0
            while (n < 1):
                lmn.append((l, m, n))
                n += step
            m += step
        l += step
    true_lmn = []
    for i in lmn:
        if (i[0] ** 2 + i[1] ** 2 + i[2] ** 2 <= 1):
            true_lmn.append(i)
    return true_lmn


def get_plates(lmn_arr: list(), main_strength: Main_strength, tensor: Tensor, L: tuple ):
    plates = []
    for i in lmn:
        plate = Plate(main_strength, tensor, i, L)
        plates.append(plate)
    return plates



C_x = 4
C_y = 0.5
C_z = 0.2
q = 0.2

R_x = 4.5
R_y = 3.5
R_z = 5

main_strength = Main_strength(C_x, C_y, C_z, R_x, R_y, R_z, q)
# l1x, l2x, l3x = 1, 0, 0
# l1y, l2y, l3y = 0, 1, 0
# l1z, l2z, l3z = 0, 0, 1

# l1x, l2x, l3x = 0.71, 0.71, 0
# l1y, l2y, l3y = -0.71, 0.71, 0
# l1z, l2z, l3z = 0, 0, 1

l1x, l2x, l3x = 0.92, 0.3, -0.24
l1y, l2y, l3y = -0.11, 0.81, 0.57
l1z, l2z, l3z = 0.37, -0.5, 0.78

L = ((l1x, l2x, l3x), (l1y, l2y, l3y), (l1z, l2z, l3z))

sigs1,sigs2,sigs3=[],[],[]

def get_stressPointsArr(lower_bound, upper_bound, step, lmn_arr):
    stress_points = []
    sig1 = lower_bound
    while (sig1 < upper_bound):
        sig2 = lower_bound
        while (sig2 < upper_bound):
            sig3 = lower_bound
            while (sig3 < upper_bound):
                sp = Stress_point(Tensor(sig1, sig2, sig3), main_strength, L)
                sp.get_plates(lmn_arr)
                stress_points.append(sp)
                if len(sp.danger_plates)!=0:
                    sigs1.append(sp.tensor.sigma1)
                    sigs2.append(sp.tensor.sigma2)
                    sigs3.append(sp.tensor.sigma3)
                sig3 += step
            sig2 += step
        sig1 += step
    return stress_points


lmn = get_lmn(0.2)
stressPoints = get_stressPointsArr(-2, 5, 0.33, lmn)


fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.set_title(f"Cx = {C_x}, Cy={C_y}, Cz={C_z}")
ax.scatter(sigs1, sigs2, sigs3)  # plot the point (2,3,4) on the figure
ax.set_xlabel("sigma1")
ax.set_ylabel("sigma2")
ax.set_zlabel("sigma3")
plt.show()
