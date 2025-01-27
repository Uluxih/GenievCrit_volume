import numpy as np
import GenCrit
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def get_lmn(plates, n):
    list = []
    for i in plates:
        list.append(i.lmn)
        n=n-1
        if n==0:
            break
    return list

l1x, l2x, l3x = 1, 0, 0
l1y, l2y, l3y = 0, 1, 0
l1z, l2z, l3z = 0, 0, 1

# Коэф трения
q = 0.33

L = np.array(((l1x, l2x, l3x), (l1y, l2y, l3y), (l1z, l2z, l3z)))

Cxz = 1.61
Cxy = 1.2
Cyz = 1.61
Cyx = 1.6
Czx = 1.21
Czy = 1.2
Cx45= 1.22
Cy45 = 1.2
Cz45 = 1.2



# Прочности на срез
C_x = GenCrit.get_C(Cxz,Cxy,Cx45,math.radians(0))
C_y = GenCrit.get_C(Cyz,Cyx,Cy45,math.radians(0))
C_z = GenCrit.get_C(Czx,Czy,Cz45,math.radians(0))

main_strength = GenCrit.Main_strength(Cxz, Cxy, Cx45, Cyz, Cyx, Cy45, Czx, Czy,Cz45, q)
#С каким шагом делить сферу в градусах
lmn1 = GenCrit.get_lmn(5, 360)


# Тензор напряжний
sigmaX=-0.079
sigmaY=0.159
sigmaZ=-0.2054
sigmaXY=-0.007
sigmaXZ=-0.674
sigmaYZ=0.075
sp = GenCrit.Stress_point(GenCrit.Tensor(sigmaX, sigmaY, sigmaZ, sigmaXY, sigmaXZ, sigmaYZ), main_strength, L)
sp.get_plates(lmn1)
sp.sort_plate()

print("Площадка с наибольшим коэфом",get_lmn(sp.plates, 3))
print("Площадка с наибольшими касательными напряжениями", get_lmn(sp.get_sorted_plate_by_shear(),3))
print("Площадка с наибольшими нормальными напряжениями", get_lmn(sp.get_sorted_plate_by_sigma(),3))