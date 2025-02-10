import ModelGeomPy
import pandas as pd
import math
import pyautocad
import GenCrit
import numpy as np

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

oldCord = np.array([0,0.5547002,0.83205029])
new_y = np.array([0,2,3])
new_y = new_y/np.linalg.norm(new_y)
new_z = np.cross(np.array([1,0,0]), new_y)
new_basis = np.array([[1,0,0], new_y, new_z])
a = ModelGeomPy.get_newCord(new_basis, oldCord)


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
    basis = [new_e1, new_e2, new_e3]
    # print(basis)
    # print(new_e1, new_e2, new_e3)
    elems_str = i.split(':')[1].split()
    elems = get_elemByStr(elems_str)
    print(elems)
    basisElems.append(basis,elems)