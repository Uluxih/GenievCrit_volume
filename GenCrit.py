import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class Tensor:
    def __init__(self, sigma1, sigma2, sigma3, sigma12, sigma13, sigma23):
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.sigma3 = sigma3
        self.sigma12 = sigma12
        self.sigma13 = sigma13
        self.sigma23 = sigma23

    def get_tens_cord(self):
        tens_cord = np.array([[self.sigma1, self.sigma12, self.sigma13],
                              [self.sigma12, self.sigma2, self.sigma23],
                              [self.sigma13, self.sigma23, self.sigma3]])
        return tens_cord

    def get_sigma(self, n):
        P_n = np.dot(self.get_tens_cord(), n)
        sigma = P_n[0] * n[0] + P_n[1] * n[1] + P_n[2] * n[2]
        return sigma

    def get_shear(self, n):
        P_n = np.dot(self.get_tens_cord(), n)
        sigma = P_n[0] * n[0] + P_n[1] * n[1] + P_n[2] * n[2]
        shear_n = np.subtract(P_n, np.dot(n, sigma))
        return shear_n

    def get_tensCord_byL(self, L):
        pass

    def get_thetaZxZyXy(self, n, L):
        shear = self.get_shear(n)
        Pxx = np.dot(L.transpose(), shear)
        XM, YM, ZM = Pxx[0], Pxx[1], Pxx[2]

        if (YM != 0):
            Qzy = ZM / YM
        else:
            Qzy = 0

        if (XM != 0):
            Qzx = ZM / XM
        else:
            Qzx = 0

        if (YM != 0):
            Qxy = XM / YM
        else:
            Qxy = 0
        return [Qzx, Qzy, Qxy]


class Plate_strength:
    def __init__(self, S_ij, S_ik, S_i45, Theta_jk):
        self.S_ij = S_ij
        self.S_ik = S_ik
        self.Theta_jk = Theta_jk
        self.S_i45 = S_i45


def get_C(S_ij, S_ik, S_i45, Theta_jk):
    cosTh = math.cos(Theta_jk)
    sinTh = math.sin(Theta_jk)
    C = cosTh ** 4 / S_ij + (4 / S_i45 - 1 / S_ij - 1 / S_ik) \
        * sinTh ** 2 * cosTh ** 2 + sinTh ** 4 / S_ik
    return 1 / C


class Main_strength:
    def __init__(self, Cxz, Cxy, Cx45, Cyz, Cyx, Cy45, Czx, Czy, Cz45, k):
        self.Cxz = Cxz
        self.Cxy = Cxy
        self.Cyz = Cyz
        self.Cyx = Cyx
        self.Czx = Czx
        self.Czy = Czy
        self.Cx45 = Cx45
        self.Cy45 = Cy45
        self.Cz45 = Cz45
        self.k = k

    def get_Cm(self, Cx, Cy, Cz):
        C_m = self.Cx * l_mx ** 2 + self.Cy * l_my ** 2 + self.Cz * l_mz ** 2
        return C_m

    def get_Cxyz(self, Qzx, Qzy, Qxy):
        Cx = get_C(self.Cxz, self.Cxy, self.Cx45, math.atan(abs(Qzy)))
        Cy = get_C(self.Cyz, self.Cyx, self.Cy45, math.atan(abs(Qzx)))
        Cz = get_C(self.Czx, self.Czy, self.Cz45, math.atan(abs(Qxy)))
        return [Cx, Cy, Cz]


class Plate:
    def __init__(self, main_strength: Main_strength, tensor: Tensor, lmn: [3], L: ((3), (3), (3))):
        self.main_strength = main_strength
        self.tensor = tensor
        self.lmn = lmn
        self.L = L
        l_mx = np.dot(L[0], lmn)
        l_my = np.dot(L[1], lmn)
        l_mz = np.dot(L[2], lmn)
        Q = tensor.get_thetaZxZyXy(lmn, L)
        Qzx, Qzy, Qxy = Q[0], Q[1], Q[2]
        self.Cx, self.Cy, self.Cz = self.main_strength.get_Cxyz(Qzx, Qzy, Qxy)

        self.C_m = self.Cx * l_mx ** 2 + self.Cy * l_my ** 2 + self.Cz * l_mz ** 2
        self.tau = np.linalg.norm(tensor.get_shear(lmn))
        self.sigma = tensor.sigma1 * lmn[0] ** 2 + tensor.sigma2 * lmn[1] ** 2 + tensor.sigma3 * lmn[2] ** 2
        if self.tau > self.C_m + main_strength.k * self.sigma:
            self.danger_plate = (True,
                                 (self.tau) /
                                 (abs(self.C_m - main_strength.k * self.sigma) + 0.001))
            # вариант когда добавка перебьет прочность
        else:
            self.danger_plate = (False,
                                 (self.tau) /
                                 (abs(self.C_m - main_strength.k * self.sigma) + 0.001))


class Stress_point:
    def __init__(self, tensor: Tensor, main_strength: Main_strength, L: tuple):
        self.tensor = tensor
        self.plates = []
        self.L = L
        self.main_strength = main_strength
        self.danger_plates = []

    def get_plates(self, lmn_arr: list()):
        for i in lmn_arr:
            plate = Plate(self.main_strength, self.tensor, i, self.L)
            self.plates.append(plate)
            if plate.danger_plate[0]:
                self.danger_plates.append(plate)

    def sort_plate(self):
        def a(b):
            return abs(b.tau) / (abs(b.C_m + b.main_strength.k * b.sigma))
        self.plates.sort(key=a, reverse=True)

    def get_sorted_plate_by_shear(self):
        def a(b):
            return abs(b.tau)
        return sorted(self.plates, key=a, reverse=True)

    def get_sorted_plate_by_sigma(self):
        def a(b):
            return abs(b.sigma)
        return sorted(self.plates, key=a, reverse=True)

    def get_sorted_plate_by_C(self):
        def a(b):
            return abs(b.C_m)
        return sorted(self.plates, key=a, reverse=True)

def lm_xyz(l, m, n, l1, l2, l3):
    # бесполезный кусок кода
    return (l * l1 + m * l2 + n * l3)


def get_crit_result(tau, Cm):
    return tau < Ccm


def get_lmn(step, max_angle):
    lmn = []
    theta = 0.1
    fi = 0.1
    while theta <= max_angle:
        fi = 0.1
        while fi <= max_angle:
            l = round(1 * math.sin(math.radians(theta)) * math.cos(math.radians(fi)), 3)
            m = round(1 * math.sin(math.radians(theta)) * math.sin(math.radians(fi)), 3)
            n = round(1 * math.cos(math.radians(theta)), 3)
            lmn.append((l, m, n))
            fi += step
        theta += step
    return sorted(list(set(lmn)))


def rotate_matrix_x(theta, L):
    theta = np.radians(theta)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((1, 0, 0), (0, c, -s), (0, s, c)))
    res = np.dot(R, L)
    return res


def rotate_matrix_z(theta, L):
    theta = np.radians(theta)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
    res = np.dot(R, L)
    return res


def rotate_matrix_y(theta, L):
    theta = np.radians(theta)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)))
    res = np.dot(R, L)
    return res


def get_plates(lmn_arr: list(), main_strength: Main_strength, tensor: Tensor, L: tuple):
    plates = []
    for i in lmn:
        plate = Plate(main_strength, tensor, i, L)
        plates.append(plate)
    return plates


def get_stressPointsArr(lower_bound, upper_bound, step, lmn_arr, main_strength, L, sigs1, sigs2, sigs3, colors):
    stress_points = []
    sig1 = lower_bound
    while (sig1 < upper_bound):
        sig2 = lower_bound
        while (sig2 < upper_bound):
            sig3 = lower_bound
            while (sig3 < upper_bound):
                sp = Stress_point(Tensor(sig1, sig2, sig3, 0, 0, 0), main_strength, L)
                sp.get_plates(lmn_arr)
                sp.sort_plate()
                stress_points.append(sp)
                if len(sp.danger_plates) == 0:
                    sigs1.append(sp.tensor.sigma1)
                    sigs2.append(sp.tensor.sigma2)
                    sigs3.append(sp.tensor.sigma3)
                    colors.append(
                        (round(sp.plates[0].lmn[0], 3), round(sp.plates[0].lmn[1], 3), round(sp.plates[0].lmn[2], 3)))
                sig3 += step
            sig2 += step
        sig1 += step
    return stress_points