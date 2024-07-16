import math
import numpy as np
from manim import *
import pyautocad as pyA
from pyautocad import Autocad, APoint, ACAD
import matplotlib.pyplot as plt

def lm_xyz(l, m, n, l1, l2, l3):
    return (l*l1+m*l2+n*l3)

def get_crit_result(tau, Cm):
    return tau<Cm

C_x=1/5
C_y=1/5
C_z=1/5

sigma1=2
sigma2=2
sigma3=2

l=0
m=0
n=math.sqrt(1-l**2-m**2)

# l1x=0.707106781
# l2x=-0.707106781
# l3x=math.sqrt(1-l1x**2-l2x**2)
#
# l1y=0.707106781
# l2y=0.707106781
# l3y=math.sqrt(1-l1y**2-l2y**2)
#
# l1z=math.sqrt(1-l1x**2-l1y**2)
# l2z=math.sqrt(1-l2x**2-l2y**2)
# l3z=math.sqrt(1-l3x**2-l3y**2)

l1x=1
l2x=0
l3x=0

l1y=0
l2y=1
l3y=0

l1z=0
l2z=0
l3z=1

#print(l1x,"\t", l2x,"\t",l3x)
#print(l1y,"\t",l2y,"\t",l3y)
#print(l1z,"\t",l2z,"\t",l3z)

l_mx=lm_xyz(l, m, n, l1x, l2x, l3x)
l_my=lm_xyz(l, m, n, l1y, l2y, l3y)
l_mz=lm_xyz(l, m, n, l1z, l2z, l3z)
#print(l_mx, l_my, l_mz, math.sqrt(l_mx**2+l_my**2+l_mz**2) )

C_m=C_x*l_mx**2+C_y*l_my**2+C_z*l_mz**2

def get_lmn():
    lmn = []
    l=0
    m=0
    n=0
    while(l<1):
        m=0
        while(m<1 and l**2+m**2+n**2<=1):
            n=math.sqrt(1-l**2-m**2)
            lmn.append((l,m,n))
            m=m+0.05
            n=0
        l=l+0.05
    return lmn
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
            if lmn_taum[lmn_comb]/lmn_sigm[lmn_comb]>=lmn_Cm[lmn_comb]:
                #print("Сомнительно, но окэй", lmn_taum[lmn_comb]/lmn_sigm[lmn_comb]-lmn_Cm[lmn_comb])
                danger_plate.append(lmn_comb)
                #print(i)
    return danger_plate
sigs=[]
sig1=0
lmn=get_lmn()
while (sig1<5):
    sig2 = 0
    while (sig2<5):
        sig3 = 0
        while (sig3<5):
            sigs.append((sig1, sig2, sig3))
            sig3+=0.2
        sig2+=0.2
    sig1+=0.2
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
print(sigs)

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.set_title(f"Cx = {C_x}, Cy={C_y}, Cz={C_z}")
ax.scatter(sigs1,sigs2,sigs3) # plot the point (2,3,4) on the figure
ax.set_xlabel("sigma1")
ax.set_ylabel("sigma2")
ax.set_zlabel("sigma3")
plt.show()