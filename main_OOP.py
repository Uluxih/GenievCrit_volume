import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import GenCrit as gc

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
        if self.tau > self.C_m + main_strength.k * self.sigma:
            self.danger_plate = (True,
                                 (self.tau) /
                                 (abs(self.C_m + main_strength.k * self.sigma) + 0.001))
        else:
            self.danger_plate = (False,
                                 (self.tau) /
                                 (abs(self.C_m + main_strength.k * self.sigma) + 0.001))


class Stress_point:
    def __init__(self, tensor: Tensor, main_strength: Main_strength, L: tuple):
        self.tensor = tensor
        self.plates = []
        self.L = L
        self.main_strength = main_strength
        self.danger_plates = []

    def get_plates(self, lmn_arr: list()):
        for i in lmn:
            plate = Plate(self.main_strength, self.tensor, i, self.L)
            self.plates.append(plate)
            if plate.danger_plate[0]:
                self.danger_plates.append(plate)

    def sort_plate(self):
        def a(b):
            return abs(b.tau) / (abs(b.C_m + b.main_strength.k * b.sigma))
        self.plates.sort(key=a, reverse=True)


def lm_xyz(l, m, n, l1, l2, l3):
    return (l * l1 + m * l2 + n * l3)


def get_crit_result(tau, Cm):
    return tau < Ccm


def get_lmn(step):
    lmn=[]
    theta=0
    fi=0
    while theta<=90:
        fi=0
        while fi<=90:
            l=round(1*math.sin(math.radians(theta))*math.cos(math.radians(fi)),3)
            m=round(1*math.sin(math.radians(theta))*math.sin(math.radians(fi)),3)
            n=round(1*math.cos(math.radians(theta)),3)
            lmn.append((l,m,n))
            # print(l**2+m**2+n**2)
            fi+=step
        theta+=step
    return list(set(lmn))


def get_plates(lmn_arr: list(), main_strength: Main_strength, tensor: Tensor, L: tuple):
    plates = []
    for i in lmn:
        plate = Plate(main_strength, tensor, i, L)
        plates.append(plate)
    return plates

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
                # print(sp.plates[0].lmn)
                # print('do:')
                # for i in sp.plates:
                #     print(i.lmn, i.danger_plate, i.tau)
                sp.sort_plate()
                # print('posle:')
                # for i in sp.plates:
                #     print(i.lmn, i.danger_plate, i.tau, i.C_m)
                # print('posle:', sp.plates[0].lmn, sp.plates[0].danger_plate, i.tau)
                stress_points.append(sp)
                if len(sp.danger_plates) == 0:
                    sigs1.append(sp.tensor.sigma1)
                    sigs2.append(sp.tensor.sigma2)
                    sigs3.append(sp.tensor.sigma3)
                    colors.append((round(sp.plates[0].lmn[0],3),round(sp.plates[0].lmn[1],3),round(sp.plates[0].lmn[2],3)))
                sig3 += step
            sig2 += step
        sig1 += step
    return stress_points

Cxz = 1.41
Cxy = 1.9
Cyz = 1.41
Cyx = 1.0
Czx = 1.41
Czy = 1.0
Cx45= 0.2
Cy45 = 1.2
Cz45 = 1.2

q = -0.25

R_x = 0
R_y = 0
R_z = 0

main_strength = gc.Main_strength(Cxz, Cxy, Cx45, Cyz, Cyx,Cy45, Czx, Czy,Cz45, q)

l1x, l2x, l3x = 1, 0, 0
l1y, l2y, l3y = 0, 1, 0
l1z, l2z, l3z = 0, 0, 1

# l1x, l2x, l3x = 0.71, 0.71, 0
# l1y, l2y, l3y = -0.71, 0.71, 0
# l1z, l2z, l3z = 0, 0, 1

# l1x, l2x, l3x = 0.92, 0.3, -0.24
# l1y, l2y, l3y = -0.11, 0.81, 0.57
# l1z, l2z, l3z = 0.37, -0.5, 0.78
# L = ((l1x, l2x, l3x), (l1y, l2y, l3y), (l1z, l2z, l3z))
L = np.array(((l1x, l2x, l3x), (l1y, l2y, l3y), (l1z, l2z, l3z)))
a=0
b=0
c=0
L=gc.rotate_matrix_x(a, L)
L=gc.rotate_matrix_y(b, L)
L=gc.rotate_matrix_z(c, L)
sigs1, sigs2, sigs3 = [], [], []
colors = []

lmn = gc.get_lmn(22.5, 90)
main_strength = gc.Main_strength(Cxz, Cxy, Cx45, Cyz, Cyx,Cy45, Czx, Czy, Cz45, q)
stressPoints = gc.get_stressPointsArr(-5, 5, 0.5, lmn, main_strength, L, sigs1,sigs2,sigs3,colors)
print(colors)

uniq_color = list(set(colors))
print(len(uniq_color))

fig, ax = plt.subplots()

ax = fig.add_subplot(111, projection='3d')
# ax.set_title(f"Cx = {C_x}, Cy={C_y}, Cz={C_z}")
scatter = ax.scatter(sigs1, sigs2, sigs3, c = colors, label='1')  # plot the point (2,3,4) on the figure
ax.set_xlabel("sigma1")
ax.set_ylabel("sigma2")
ax.set_zlabel("sigma3")
recs = []
for i in range(0, len(uniq_color)):
    recs.append(mpatches.Rectangle((0, 0), 0.5, 0.5, fc=uniq_color[i]))
plt.legend(recs, uniq_color, loc=8, ncol=6)
plt.show()