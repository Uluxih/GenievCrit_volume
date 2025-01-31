import ModelGeomPy
import pandas as pd
import math
import pyautocad
import GenCrit

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
i = 0
while i < len(num_elems):
    s1=tens_elemsXx[i]
    s2 = tens_elemsYy[i]
    s3= tens_elemsZz[i]
    s12 = tens_elemsXy[i]
    s13 = tens_elemsXz[i]
    s23 = tens_elemsYz[i]
    tens = GenCrit.Tensor(s1,s2,s3,s12,s13,s23)
print(tens_elemsYz)
