import ModelGeomPy
import pandas as pd
import math

df_elems = pd.read_excel('СВОД2.xls', sheet_name='1. Элементы', skiprows=3)
df_nodes1 = pd.read_excel('СВОД2.xls', sheet_name='2. Координаты и связи', skiprows=5)
df_nodes2 = pd.read_excel('СВОД2.xls', sheet_name='2. Координаты и связи', skiprows=6)
num_nodes = df_nodes1['Номер узла'].tolist()
num_nodes.pop(0)
df_nodes2['X']=df_nodes2['X'].str.replace(',','.')
df_nodes2['Y']=df_nodes2['Y'].str.replace(',','.')
df_nodes2['Z']=df_nodes2['Z'].str.replace(',','.')
x_nodes=df_nodes2['X'].tolist()
y_nodes=df_nodes2['Y'].tolist()
z_nodes=df_nodes2['Z'].tolist()
i=0
while i < len(num_nodes):
    ModelGeomPy.Node(i, x_nodes[i], y_nodes[i], z_nodes[i])
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
print(ModelGeomPy.Elem.num_toElem[10].center)