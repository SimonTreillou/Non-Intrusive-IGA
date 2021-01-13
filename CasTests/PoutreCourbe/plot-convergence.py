#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 20:23:25 2020

@author: treillou
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) == 1:
   print("No arguments\nPlot: girder - residue")
   case = 'girder'
   curv = 'residue'
else:
   curv = sys.argv[2]
   case = sys.argv[1]
   print("Plot: "+case+" - "+curv+"\n")
   
if case=='girder':
   if curv=='residue':
   	normr = np.load('normr.npy') 
   	normr = normr/normr[0] 	
   elif curv=='error':
   	normr = np.load('err.npy')	
   	normr = normr/normr[0]
   else:
   	print('Invalid argument')
elif case=='toy':
   if curv=='residue':
   	normr = np.load('../Toy-case/normr.npy')
   elif curv=='error':
   	print('Not implemented yet')
   else:
   	print('Invalid argument')
   	
nbiter = np.linspace(1,normr.size,normr.size)
plt.plot(nbiter,normr)
plt.yscale('log')
plt.grid()
plt.xlabel('iterations')
plt.title(case+' convergence')
plt.ylabel('relative normalized '+curv)
plt.show()
