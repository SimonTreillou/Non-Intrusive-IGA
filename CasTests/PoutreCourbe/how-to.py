#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 12:30:41 2020

@author: Treillou
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splalg
import subprocess
import os 

# %% HOW-TO 
###
### CALL ASTER AS_RUN
###

## subprocess.call('~/dir-code-aster/aster-14.4.0/bin/as_run globalt.export',shell=True,)

####
#### LOAD NPY FILES (numpy arrays)
####

# KIGA = np.load('./NPY/poutrecourbeKIGA.npy',allow_pickle=True)

###
### LOAD NPZ FILES (sparse arrays)
###

# KG = sp.load_npz('./NPZ/KG.npz')


# %% EXAMPLES 

## need to specify the working directory before execution
## if COPY ERROR => 2nd execution
workdir = os.getcwd()
asterexp = workdir+'/global.export'
command = '~/dir-code-aster/aster-14.4.0/bin/as_run '+asterexp
## run global.export and create npz and npy files of interest
subprocess.call(command,shell=True,)

## Global model operator KG
KG = sp.load_npz('KG.npz')
## Global model displacement UG
UG = np.load('UG.npy')

## it works
FS = KG@UG
KGLU = splalg.splu(KG)
US = KGLU.solve(FS)



