import numpy as np
import GenCrit
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm


l1x, l2x, l3x = 1, 0, 0
l1y, l2y, l3y = 0, 1, 0
l1z, l2z, l3z = 0, 0, 1

# l1x, l2x, l3x = 0.71, 0.71, 0
# l1y, l2y, l3y = -0.71, 0.71, 0
# l1z, l2z, l3z = 0, 0, 1

# l1x, l2x, l3x = 0.92, 0.3, -0.24
# l1y, l2y, l3y = -0.11, 0.81, 0.57
# l1z, l2z, l3z = 0.37, -0.5, 0.78



# Коэф трения
q = 0.33

#Нормальные напряжения нет влияния
R_x = 0
R_y = 0
R_z = 0

L = np.array(((l1x, l2x, l3x), (l1y, l2y, l3y), (l1z, l2z, l3z)))

#Повернуть матрицу относительно оси x в градусах:
L=GenCrit.rotate_matrix_x(0, L)
#Повернуть матрицу относительно оси y:
L=GenCrit.rotate_matrix_y(0, L)
#Повернуть матрицу относительно оси z:
L=GenCrit.rotate_matrix_z(0, L)

Cxz = 0.81
Cxy = 0.8
Cyz = 1.41
Cyx = 1.4
Czx = 1.41
Czy = 1.4
Cx45= 0.4
Cy45 = 1.4
Cz45 = 1.4

sigma1=3
sigma2=0
sigma3=3

# Прочности на срез
C_x = GenCrit.get_C(Cxz,Cxy,Cx45,math.radians(45))
print(C_x)
C_y = GenCrit.get_C(Cyz,Cyx,Cy45,math.radians(0))
C_z = GenCrit.get_C(Czx,Czy,Cz45,math.radians(0))

# main_strength = GenCrit.Main_strength(C_x, C_y, C_z, R_x, R_y, R_z, q)
main_strength = GenCrit.Main_strength(Cxz, Cxy, Cx45, Cyz, Cyx,Cy45, Czx, Czy,Cz45, q)
#С каким шагом делить сферу в градусах
lmn1 = GenCrit.get_lmn(5, 360)
# Тензор напряжний сигма1, сигма2, сигма3
sp = GenCrit.Stress_point(GenCrit.Tensor(sigma1, sigma2, sigma3), main_strength, L)
sp.get_plates(lmn1)
print(lmn1)
j=0

x,y,z=[],[],[]
c=[]
txt=[]
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for i in sp.plates:
    j += 1
    crit=i.danger_plate[1] # Фигура по коэфу использования
    crit = i.tau # Фигура по касательным
    # crit = i.sigma # Фигура по нормальным
    crit = i.C_m
    c.append(crit)
    txt.append(crit)
    x.append(i.lmn[0]*crit)
    y.append(i.lmn[1]*crit)
    z.append(i.lmn[2]*crit)
    # x.append(i.lmn[0]*1)
    # y.append(i.lmn[1]*1)
    # z.append(i.lmn[2]*1)
    # scatter = ax.scatter(i.lmn[0], i.lmn[1], i.lmn[2])
    # ax.text(i.lmn[0], i.lmn[1], i.lmn[2] + 0.05, round(i.C_m,2))
    # ax.text(i.lmn[0], i.lmn[1], i.lmn[2], round(crit, 3), size=5)
    # print(j, i.lmn, i.C_m, i.tau)


ax.text(sp.plates[0].lmn[0], sp.plates[0].lmn[1], sp.plates[0].lmn[2], round(sp.plates[0].C_m,2))


if crit==i.C_m:
    ax.set_title(f"Cx = {round(C_x,2)}, Cy={round(C_y,2)}, Cz={round(C_z,2)}")
if crit==i.tau:
    ax.set_title(f"σ1 = {sp.tensor.sigma1}, σ2={sp.tensor.sigma2}, σ3={sp.tensor.sigma3}")
if crit == i.sigma:
    ax.set_title(f"σ1 = {sp.tensor.sigma1}, σ2={sp.tensor.sigma2}, σ3={sp.tensor.sigma3}")

scatter = ax.scatter(x, y, z, c=c, cmap=cm.jet)  # plot the point (2,3,4) on the figure

fig.colorbar(scatter, shrink=0.9, location="right")
ax.set_xlabel("1")
ax.set_ylabel("2")
ax.set_zlabel("3")
max_val=max([C_x, C_y, C_z], key=abs)
# max_val=crit
ax.set_xlim3d(-max_val*1.2, max_val*1.2)
ax.set_ylim3d(-max_val*1.2, max_val*1.2)
ax.set_zlim3d(-max_val*1.2, max_val*1.2)

ax.set_box_aspect([1,1,1])
ax.view_init(30,30)
# ax.view_init(27,139)
plt.show()