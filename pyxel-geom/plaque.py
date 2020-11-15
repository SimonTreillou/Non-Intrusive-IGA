#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 09:52:07 2020

@author: poumaziz
"""

import numpy as np
import matplotlib.pyplot as plt

import Geometries as geo
import pyNURBS as pn

import pyxel as px

xmin = 0
xmax = 10
ymin = 0
ymax = 2
ne_xi = 6
ne_eta = 4
degree_xi = 2
degree_eta = 2

m = geo.RectangularPlate(xmin,xmax,ymin,ymax, ne_xi, ne_eta, degree_xi, degree_eta)

P_IGA = m.Get_Btot()

#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.axis('equal')
#plt.plot(P_IGA[0],P_IGA[1],'o',label='Points de contrôle')


type_Q = 'Q9'
C = m.Matrix_N2ToL2(type_Q)
N, E = m.ToLagrange2(type_Q)

m_EF = px.Mesh(E,N.T)

with open('plate.npy','wb') as f:
    np.save(f,[C,N,E])

m_EF.SaveMeshGMSH('./test.msh')