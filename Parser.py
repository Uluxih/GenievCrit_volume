import ModelGeomPy
import pandas as pd
import math
import pyautocad
import GenCrit
import numpy as np
from time import sleep

def get_lmn_byPlate(plates, n):
    list = []
    for i in plates:
        list.append(i.lmn)
        n=n-1
        if n==0:
            break
    return list

def get_elemByStr(str_elems):
    count = len(str_elems)
    elems=[]
    i=0
    while i<count:
        if i+1<count:
            if str_elems[i+1] == 'r':

                upper_band = int(str_elems[i])
                bottom_band = int(str_elems[i+2])
                # print('r', bottom_band)
                j = upper_band
                while j <= bottom_band:
                    # print(j)
                    elems.append(j)
                    j+=int(str_elems[i+3])
                i=i+4
                continue
        # print(str_elems[i])
        if len(str_elems[i].split('-')) > 1:
            # print(str_elems[i].split('-'))
            upper_band = str_elems[i].split('-')[1]
            bottom_band = str_elems[i].split('-')[0]
            i=bottom_band
            while i<=upper_band:
                elems.append(i)
                i+=1
        else:
            elems.append(int(str_elems[i]))
        i=i+1
    return elems

df_elems = pd.read_excel('СВОД2.xls', sheet_name='1. Элементы', skiprows=3)
df_nodes1 = pd.read_excel('СВОД2.xls', sheet_name='2. Координаты и связи', skiprows=5)
df_nodes2 = pd.read_excel('СВОД2.xls', sheet_name='2. Координаты и связи', skiprows=6)
num_nodes = df_nodes1['Номер узла'].tolist()
num_nodes.pop(0)
# print(df_nodes2['X'])
# df_nodes2['X']=df_nodes2['X'].str.replace(',','.')
# df_nodes2['Y']=df_nodes2['Y'].str.replace(',','.')
# df_nodes2['Z']=df_nodes2['Z'].str.replace(',','.')
x_nodes=df_nodes2['X'].tolist()
y_nodes=df_nodes2['Y'].tolist()
z_nodes=df_nodes2['Z'].tolist()
i=0
while i < len(num_nodes):
    ModelGeomPy.Node(num_nodes[i], x_nodes[i], y_nodes[i], z_nodes[i])
    i+=1
i=0
while i < len(df_elems):
    num_nodes_list = df_elems.iloc[i].tolist()
    num = num_nodes_list[0]
    del num_nodes_list[:3]
    nodes_list=[]
    for j in num_nodes_list:
        if math.isnan(j)==False:
            nodes_list.append(ModelGeomPy.Node.num_toNode[j])
    ModelGeomPy.Elem(num, nodes_list)
    ModelGeomPy.Elem.num_toElem[num].setCenter()
    i+=1

df_tens1 = pd.read_excel('СВОД_нагрузки.xls', sheet_name='1. Величины усилий', skiprows=9)
df_tens2 = pd.read_excel('СВОД_нагрузки.xls', sheet_name='1. Величины усилий', skiprows=10)
num_elems = df_tens1['Элемент'].tolist()
num_elems.pop(0)
tens_elemsXx = df_tens2['sX'].tolist()
tens_elemsYy = df_tens2['sY'].tolist()
tens_elemsZz = df_tens2['sZ'].tolist()
tens_elemsXy = df_tens2['txy'].tolist()
tens_elemsXz = df_tens2['txz'].tolist()
tens_elemsYz = df_tens2['tyz'].tolist()
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
lmn1 = GenCrit.get_lmn(5, 360)
main_strength = GenCrit.Main_strength(Cxz, Cxy, Cx45, Cyz, Cyx, Cy45, Czx, Czy,Cz45, q)

i = 0
while i < len (num_elems):
    print(num_elems[i])
    i = int(i)
    s1=tens_elemsXx[i]
    s2 = tens_elemsYy[i]
    s3= tens_elemsZz[i]
    s12 = tens_elemsXy[i]
    s13 = tens_elemsXz[i]
    s23 = tens_elemsYz[i]
    tens = GenCrit.Tensor(s1,s2,s3,s12,s13,s23)
    elem = ModelGeomPy.Elem.num_toElem[num_elems[i]]
    sp = GenCrit.Stress_point(GenCrit.Tensor(s1, s2, s3, s12, s13, s23), main_strength, L)
    sp.get_plates(lmn1)
    sp.sort_plate()
    elem.stressPoint = sp
    i += 1

acad = pyautocad.Autocad()
acad.prompt("Hello epti")
# print(acad.doc.Name)

file = open('СВОД2.txt', 'r')
text = file.read()
text = text.split('(')
numToKeyDic = {}
for i in text:
    i = i.split('/')
    numToKeyDic[i[0]] = i[1:]
numToKeyDic['33'].pop
# print(numToKeyDic['33'])
basisElems = []

for i in numToKeyDic['33']:
    # print(i)
    if i==')' or len(i.split(':'))==1:
        continue
    eCord = i.split(':')[0].split()
    if len(eCord)>5:
        e1 = np.array([float(eCord[1]), float(eCord[2]),float(eCord[3])])
        e2 = np.array([float(eCord[4]), float(eCord[5]),float(eCord[6])])
    else:
        continue
    new_e1 = e1 / np.linalg.norm(e1)
    new_e2 = e2 / np.linalg.norm(e2)

    new_e3 = np.cross(new_e1, new_e2)
    new_e3 = new_e3 / np.linalg.norm(new_e3)
    basis = np.array([new_e1, new_e2, new_e3])
    # print(basis)
    # print(new_e1, new_e2, new_e3)
    elems_str = i.split(':')[1].split()
    elems = get_elemByStr(elems_str)
    basisElems.append([basis,elems])

num_toBasis={}
for i in basisElems:
    basis = i[0]
    numElems = i[1]
    for j in numElems:
        num_toBasis[j] = basis

for i in ModelGeomPy.Elem.num_toElem.items():
    # n = get_lmn_byPlate(sp.get_sorted_plate_by_shear(),3))
    n = get_lmn_byPlate(i[1].stressPoint.get_sorted_plate_by_shear(),1)[0]
    # print(n)
    # n = (0, 2, 3)
    i[1].basis=num_toBasis[i[1].number]
    # print(i[1].number, i[1].basis)

    points1, points2, points3, points4 = ModelGeomPy.get_draw_points(n, i[1].center)
    points1n = ModelGeomPy.get_newCord(i[1].basis, points1)
    points2n = ModelGeomPy.get_newCord(i[1].basis, points2)
    points3n = ModelGeomPy.get_newCord(i[1].basis, points3)
    points4n = ModelGeomPy.get_newCord(i[1].basis, points4)
    points = [points1n, points2n, points3n, points4n]
    nodesX = []
    nodesY = []
    nodesZ = []
    for j in points:
        nodesX.append(float(j[0]))
        nodesY.append(float(j[1]))
        nodesZ.append(float(j[2]))
    meanX = np.mean(nodesX)
    meanY = np.mean(nodesY)
    meanZ = np.mean(nodesZ)
    center1 = (round(meanX, 3), round(meanY, 3), round(meanZ, 3))

    delta_centerX = i[1].center[0] - meanX
    delta_centerY = i[1].center[1] - meanY
    delta_centerZ = i[1].center[2] - meanZ

    p1 = pyautocad.APoint(points1n[0]+delta_centerX, points1n[1]+delta_centerY, points1n[2]+delta_centerZ)
    p2 = pyautocad.APoint(points2n[0]+delta_centerX, points2n[1]+delta_centerY, points2n[2]+delta_centerZ)
    p3 = pyautocad.APoint(points3n[0]+delta_centerX, points3n[1]+delta_centerY, points3n[2]+delta_centerZ)
    p4 = pyautocad.APoint(points4n[0]+delta_centerX, points4n[1]+delta_centerY, points4n[2]+delta_centerZ)




    acad.model.addpoint(p1)
    acad.model.addpoint(p2)
    acad.model.addpoint(p3)
    acad.model.addpoint(p4)
    acad.model.AddLine(p1, p2)
    acad.model.AddLine(p2, p3)
    acad.model.AddLine(p3, p4)
    acad.model.AddLine(p4, p1)