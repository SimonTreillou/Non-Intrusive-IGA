#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 00:16:45 2020

@author: Treillou
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splalg
import subprocess
import os 
import glob

# %%
workdir = os.getcwd()

def from_aster(model):
    # model : global
    #       : local
    asterexp = workdir+'/'+model+'.export'
    command = '~/dir-code-aster/aster-14.4.0/bin/as_run '+asterexp
    ## run global.export and create npz and npy files of interest
    subprocess.call(command,shell=True,)
    ## Global model operator KG
    KG = sp.load_npz('KG.npz')
    ## Global model displacement UG
    UG = np.load('UG.npy')
    ## RHS
    FS = KG@UG
    return KG,UG,FS

## => arriver à modifier le .comm ?
## automatiser
## sortir les groupes de mailles pour C1,C2..
## même chose mais en rentrant des paramètres à aster ??
    
KG,UG,FS = from_aster('global')
KL = sp.load_npz('KL.npz')
print(KG)
