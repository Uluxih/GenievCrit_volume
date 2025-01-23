import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import GenCrit as gc


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


def get_plates(lmn_arr: list(), main_strength: gc.Main_strength, tensor: gc.Tensor, L: tuple):
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
                sp.sort_plate()
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
Cxy = 1.84
Cyz = 1.41
Cyx = 1.4
Czx = 1.41
Czy = 1.84
Cx45= 1.41
Cy45 = 1.41
Cz45 = 1.41

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

lmn = gc.get_lmn(20, 90)
main_strength = gc.Main_strength(Cxz, Cxy, Cx45, Cyz, Cyx,Cy45, Czx, Czy, Cz45, q)
stressPoints = gc.get_stressPointsArr(-5, 3, 0.5, lmn, main_strength, L, sigs1,sigs2,sigs3,colors)
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