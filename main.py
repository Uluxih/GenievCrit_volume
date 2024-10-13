import math
import numpy as np
import matplotlib.pyplot as plt

def lm_xyz(l, m, n, l1, l2, l3):
    return (l*l1+m*l2+n*l3)

def get_crit_result(tau, Cm):
    return tau<Ccm

class Stress_point:
    def __init__(self, tensor, plates):
        self.tensor=tensor
        self.plates=plates
class Tensor:
    def __init__(self, sigma1, sigma2, sigma3):
        self.sigma1=sigma1
        self.sigma2=sigma2
        self.sigma3=sigma3
class Plate:
    def __init__(self, strength_plate, stress_state_plate, lmn):
        self.strength_plate=strength_plate
        self.stress_state_plate=stress_state_plate
        self.lmn=lmn
class Strength_plate:
    def __init__(self, cohesion, k, normal_resist):
        self.cohesion = cohesion
        self.k=k
        self.normal_resist=normal_resist
class Stress_state_plate:
    def __init__(self, normal, shear):
        self.normal=normal
        self.shear=shear
C_x=4
C_y=0.5
C_z=0.2
q=0.2

R_x=4.5
R_y=3.5
R_z=5

l=0
m=0
n=math.sqrt(1-l**2-m**2)

# l1x, l2x, l3x = 1, 0, 0
# l1y, l2y, l3y = 0, 1, 0
# l1z, l2z, l3z = 0, 0, 1

# l1x, l2x, l3x = 0.71, 0.71, 0
# l1y, l2y, l3y = -0.71, 0.71, 0
# l1z, l2z, l3z = 0, 0, 1

l1x, l2x, l3x = 0.92, 0.3, -0.24
l1y, l2y, l3y = -0.11, 0.81, 0.57
l1z, l2z, l3z = 0.37, -0.5, 0.78



def get_lmn():
    lmn = []
    l=0
    m=0
    n=0
    d=0.1
    while(l<1):
        m=0
        while(m<1):
            n=0
            while(n<1):
                lmn.append((l,m,n))
                n+=d
            m+=d
        l+=d
    true_lmn=[]
    for i in lmn:
        if (i[0] ** 2 + i[1] ** 2 + i[2] ** 2 <= 1):
            true_lmn.append(i)
    return true_lmn

def get_Cm(lmn):
    lmn_Cm={}
    lmn_taum={}
    lmn_sigm={}
    for lmn_comb in lmn:
        l_mx = lm_xyz(lmn_comb[0], lmn_comb[1], lmn_comb[2], l1x, l2x, l3x)
        l_my = lm_xyz(lmn_comb[0], lmn_comb[1], lmn_comb[2], l1y, l2y, l3y)
        l_mz = lm_xyz(lmn_comb[0], lmn_comb[1], lmn_comb[2], l1z, l2z, l3z)
        C_m = C_x * l_mx ** 2 + C_y * l_my ** 2 + C_z * l_mz ** 2
        lmn_Cm[lmn_comb]=C_m
    return lmn_Cm

def get_taum(lmn, sigs):
    sigma1=sigs[0]
    sigma2 = sigs[1]
    sigma3 = sigs[2]
    lmn_taum={}
    for lmn_comb in lmn:
        l=lmn_comb[0]
        m = lmn_comb[1]
        n = lmn_comb[2]
        tau=math.sqrt(((sigma1-sigma2)*l*m)**2+((sigma3-sigma2)*m*n)**2+((sigma1-sigma3)*l*n)**2)
        lmn_taum[lmn_comb]=tau
    return lmn_taum

def get_sigm(lmn, sigs):
    sigma1=sigs[0]
    sigma2 = sigs[1]
    sigma3 = sigs[2]
    lmn_sigm = {}
    for lmn_comb in lmn:
        l = lmn_comb[0]
        m = lmn_comb[1]
        n = lmn_comb[2]
        sigma=sigma1*l**2+sigma2*m**2+sigma3*n**2
        #print(lmn_comb, sigma)
        lmn_sigm[lmn_comb]=sigma
    return lmn_sigm

def get_res(lmn, lmn_taum, lmn_sigm, lmn_Cm):
    danger_plate = []
    for lmn_comb in lmn:
        #print(lmn_comb, lmn_taum[lmn_comb]/lmn_sigm[lmn_comb], lmn_Cm[lmn_comb])
        if (lmn_sigm[lmn_comb]!=0):
            if lmn_taum[lmn_comb]>=lmn_Cm[lmn_comb]+q*lmn_sigm[lmn_comb]:
                danger_plate.append(lmn_comb)
                continue
                #print(i)
            # l_mx = lm_xyz(lmn_comb[0], lmn_comb[1], lmn_comb[2], l1x, l2x, l3x)
            # l_my = lm_xyz(lmn_comb[0], lmn_comb[1], lmn_comb[2], l1y, l2y, l3y)
            # l_mz = lm_xyz(lmn_comb[0], lmn_comb[1], lmn_comb[2], l1z, l2z, l3z)
            # R_m = R_x * l_mx ** 2 + R_y * l_my ** 2 + R_z * l_mz ** 2
            # if lmn_sigm[lmn_comb]>=R_m:
            #     danger_plate.append(lmn_comb)
    return danger_plate

def get_res_dict(lmn, lmn_taum, lmn_sigm, lmn_Cm):
    dangPlate_lmn = {}
    for lmn_comb in lmn:
        #print(lmn_comb, lmn_taum[lmn_comb]/lmn_sigm[lmn_comb], lmn_Cm[lmn_comb])
        if (lmn_sigm[lmn_comb]!=0):
            if lmn_taum[lmn_comb]>=lmn_Cm[lmn_comb]+q*lmn_sigm[lmn_comb]:
                danger_plate.append(lmn_comb)
                continue
                #print(i)
            # l_mx = lm_xyz(lmn_comb[0], lmn_comb[1], lmn_comb[2], l1x, l2x, l3x)
            # l_my = lm_xyz(lmn_comb[0], lmn_comb[1], lmn_comb[2], l1y, l2y, l3y)
            # l_mz = lm_xyz(lmn_comb[0], lmn_comb[1], lmn_comb[2], l1z, l2z, l3z)
            # R_m = R_x * l_mx ** 2 + R_y * l_my ** 2 + R_z * l_mz ** 2
            # if lmn_sigm[lmn_comb]>=R_m:
            #     danger_plate.append(lmn_comb)
    return danger_plate

sigs=[]
sig1=-2
lmn=get_lmn()
while (sig1<5):
    sig2 = -2
    while (sig2<5):
        sig3 = -2
        while (sig3<5):
            sigs.append((sig1, sig2, sig3))
            sig3+=0.33
        sig2+=0.33
    sig1+=0.33
print (len(sigs))
danger_point=[]
sigs1=[]
sigs2=[]
sigs3=[]
for sig in sigs:
    taum=get_taum(lmn, sig)
    sigm = get_sigm(lmn, sig)
    cm=get_Cm(lmn)
    danger_plate=get_res(lmn, taum, sigm, cm)
    if len(danger_plate)==0:
        danger_point.append(sig)
        sigs1.append(sig[0])
        sigs2.append(sig[1])
        sigs3.append(sig[2])


#Быстрый, но непонятный алгоритм


fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.set_title(f"Cx = {C_x}, Cy={C_y}, Cz={C_z}")
ax.scatter(sigs1,sigs2,sigs3) # plot the point (2,3,4) on the figure
ax.set_xlabel("sigma1")
ax.set_ylabel("sigma2")
ax.set_zlabel("sigma3")
plt.show()