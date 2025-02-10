import numpy as np

import GenCrit
import GenCrit as gc

class Node():
    num_toNode={}
    def __init__(self, num, x, y, z):
        self.cords=(x,y,z)
        self.num=num
        Node.num_toNode[num]=self
    def __repr__(self):
        return '(num =  '+str(self.num)+'; x = '+str(self.cords[0])+\
               '; y = '+str(self.cords[1])+'; z = '+str(self.cords[2])+')'


class Elem():
    num_toElem = {}
    def __init__(self, num, nodes):
        self.number=num
        self.nodes=nodes
        self.tensor = None
        self.stressPoint = GenCrit.Stress_point
        Elem.num_toElem[num] = self
        self.basis = np.array([[1,0,0],[0,1,0],[0,0,1]])
    def __repr__(self):
        return 'number: '+str(self.number)+'; (nodes): '+str(self.nodes)
    def setCenter(self):
        nodesX = []
        nodesY = []
        nodesZ = []
        for i in self.nodes:
            nodesX.append(float(i.cords[0]))
            nodesY.append(float(i.cords[1]))
            nodesZ.append(float(i.cords[2]))
        meanX = np.mean(nodesX)
        meanY = np.mean(nodesY)
        meanZ = np.mean(nodesZ)
        self.center = (round(meanX,3), round(meanY,3), round(meanZ,3))
        return (round(meanX,3), round(meanY,3), round(meanZ,3))



def get_draw_points(n,cords):
    d = 0.05
    D = -n[0]*cords[0] - n[1]*cords[1] - n[2]*cords[2]
    x1 = cords[0] + d
    y1 = cords[1] + d
    z1 = (-D - n[0]*x1 - n[1]*y1)/n[2]
    x2 = cords[0] - d
    y2 = cords[1] + d
    z2 = (-D - n[0]*x2 - n[1]*y2)/n[2]
    x3 = cords[0] - d
    y3 = cords[1] - d
    z3 = (-D - n[0]*x3 - n[1]*y3)/n[2]
    x4 = cords[0] + d
    y4 = cords[1] - d
    z4 = (-D - n[0]*x4 - n[1]*y4)/n[2]

    return (x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)


def get_newCord(new_basis, oldCord):
    old_basis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    basis_array = np.dot(old_basis, new_basis)
    newCord = np.dot(basis_array, oldCord)
    return newCord
