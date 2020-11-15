# -*- coding: utf-8 -*-
"""
Fonctions utiles pour les calculs
"""

import numpy as np



def round_to_n(ar,n):
    """ar a numpy array, n number of significant digits"""
    tmp = 10**np.floor(np.log10(np.abs(ar)))    
    return tmp*np.round(ar/tmp,decimals=n-1)

#%%

#A=np.array([[0,0],[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[0,2]])
#A = np.array([[0,3,3,1,0]]).T
def UniqueStable(A):
    ic=np.zeros_like(A[:,0])-1
    indc = np.zeros_like(A[:,0])
    h=np.max(np.linalg.norm(A-np.mean(A,axis=0),axis=1))
    c = 0
    for i in range(A.shape[0]):
        if ic[i]==-1:
            rep,=np.where(np.linalg.norm(A-A[i,:],axis=1)<h*1e-8)
            ic[rep]=i
            indc[rep]=c
            c+=1
    return indc,c # c final length
#ic=UniqueStable(A)
#print(ic)





