import numpy as np
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
        Elem.num_toElem[num] = self
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



def get_Scad_nodes():
    'Парсер узлов из документа скада'
    pass
