#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 14:25:18 2020

@author: poumaziz
"""
import salome
from salome.smesh import smeshBuilder


Gamma = salome.myStudy.FindObjectID(salome.sg.getSelected(0)).GetObject()
mesh = Gamma.GetMesh()
### On cherche les éléments appartenant à l'interface
elemInterf = Gamma.GetIDs()
### On cherche les noeuds appartenant à l'interface
nodeInterf = Gamma.GetNodeIDs()
### On cherche la connectivité de l'interface
connecInterf = {elem:mesh.GetElemNodes(elem) for elem in elemInterf}


from salome.geom import geomBuilder

geompy = geomBuilder.New()

### Création des vertex pour chaque noeud de l'interface
Vertex = dict()
for ii,node in enumerate(nodeInterf):
    XYZ = mesh.GetNodeXYZ(node)
    Vertex[node] = geompy.MakeVertex(XYZ[0],XYZ[1],XYZ[2])
    geompy.addToStudy( Vertex[node], 'Vertex_{}'.format(ii))
print("On crée les vertex dans la géométrie")

### Création des edges pour chaque élément de l'interface
Edge = dict()
for ii,elem in enumerate(elemInterf):
    Edge[elem] = geompy.MakeEdge(Vertex[connecInterf[elem][0]],Vertex[connecInterf[elem][1]])
    geompy.addToStudy( Edge[elem], 'Edge_{}'.format(ii))
print("On crée les edge dans la géométrie")
### Création de la wire

Wire = geompy.MakeWire([Edge[elem] for elem in elemInterf], 1e-07)
geompy.addToStudy( Wire, 'Wire')
print("On crée la wire dans la géométrie")
try:
    ### Création de la face 
    Face = geompy.MakeFaceWires([Wire], 1)
    geompy.addToStudy( Face, 'Face')
    print("On crée la face dans la géométrie")
except:
    print("There is a problem")

if salome.sg.hasDesktop():
  salome.sg.updateObjBrowser()
