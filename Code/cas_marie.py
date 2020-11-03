#!/usr/bin
# -*- coding: utf-8 -*-
""" Finite Element method 
    JC Passieux, INSA Toulouse, 2019       """

#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as splalg
import scipy.sparse as spsp
import pyxel as px

'''poutre courbe '''
(Draf,Dl,n,e)=np.load('poutrecourbe.npy',allow_pickle=True)
(Dl,n,e)=np.load('poutrecourbeQ9.npy',allow_pickle=True)


n=n.T
m=px.Mesh(e,n)

m.Plot(facecolor='#888888')
#%%
#nr=np.array([1132,1131,1130,1129,1128,1094,1054,1053,1052,1019,978,977,976,944,902,870,828,796,754,753,752,721,678,647,604,605,606,607,608,575,534,535,536,576,610,650,684,685,686,725,760,761,762,800,836,874,910,948,984,1022,1058,1096])
#nr2=nr[np.arange(0,52,2)]
#m2.Plot()
#plt.plot(m2.n[nr2,0],m2.n[nr2,1],'r.')
#for i in range(len(nr2)):
#    print('Point(%2d) = {%6.10e,  %6.10e, 0, .1};' % (i+9,m.n[nr2[i],0],m.n[nr2[i],1]))

mf=px.ReadMeshGMSH('fish.msh')
mf.Plot(facecolor='r')

#m.PlotElemLabels()

#mf.PlotNodeLabels()

#%%
# liste des elements recouverts par le patch
er=np.array([322,321,320,296,272,248,247,223,222,198,199,200,176,224,225,249,273,274,298,297,250,295,271,345,346])


### Création du maillage du modèle complémentaire et du maillage du modèle local
m2=m.Copy()
m3=m.Copy()
e2=dict()
e3=dict()
ne2=0
ne3=0

### C'est sale ^^
#for i in range(len(e)):
#    if len(np.where(er==i)[0])==0:
#        e2[ne2]=m.e[i]
#        ne2+=1
#    else:
#        e3[ne3]=m.e[i]
#        ne3+=1


setEr = set(er) # On convertit en ensemble pour le test d'appartenance
for ii in range(len(e)):
    if not(ii in setEr):   # Si l'élément ne recouvre pas le patch local
        e2[ne2] = m.e[ii]
        ne2 +=1
    else:                  # Si l'élément recouvre le patch local
        e3[ne3] = m.e[ii]
        ne3 +=1
m2.e=e2
m2.Plot(edgecolor='b')
m3.e=e3
m3.Plot(edgecolor='r')
m3.VTKMesh('marie_mask')

#%%
#m2.PlotNodeLabels()


# déplace les noeuds des milieux des aretes de mf à la même place que le 
# maillage Q9 m uniquement pour les noeuds d'interface.
# node_iga=np.array([1131,1129,1053,977,753,605,607,535,685,761]) # Q8
node_iga=np.array([1491,1489,1389,1289,993,797,799,703,901,1001]) # Q9
node_efa=np.array([60,61,63,65,69,72,73,75,78,80])
for i in range(len(node_iga)):
    mf.n[node_efa[i],:]=m.n[node_iga[i],:]


#%% Connectivity
m2.Connectivity()
mf.Connectivity()
m.Connectivity()

m.Plot(edgecolor='b')

row1=np.zeros(110)
col1=np.zeros(110)
val1=np.zeros(110)
row2=np.zeros(110)
col2=np.zeros(110)
val2=np.zeros(110)
row3=np.zeros(110)
col3=np.zeros(110)
val3=np.zeros(110)
nv=0
for jn in range(m2.n.shape[0]):
    xn=m2.n[jn,:]
    a=np.where(np.linalg.norm(mf.n-xn,axis=1)<1e-5)[0]
    if len(a)>0:
        plt.plot(m2.n[jn,0],m2.n[jn,1],'k.')
        plt.plot(mf.n[a,0],mf.n[a,1],'r+')
        row1[nv]=nv
        col1[nv]=m2.conn[jn,0]
        val1[nv]=1
        row3[nv]=nv
        col3[nv]=m.conn[jn,0]
        val3[nv]=1
        row2[nv]=nv
        col2[nv]=mf.conn[a,0]
        val2[nv]=-1
        row1[nv+1]=nv+1
        col1[nv+1]=m2.conn[jn,1]
        val1[nv+1]=1
        row3[nv+1]=nv+1
        col3[nv+1]=m.conn[jn,1]
        val3[nv+1]=1
        row2[nv+1]=nv+1
        col2[nv+1]=mf.conn[a,1]
        val2[nv+1]=-1
        nv+=2

import scipy as sp
CGEF=sp.sparse.csc_matrix((val1, (row1, col1)), shape=(nv,m2.ndof))
CLEF=sp.sparse.csc_matrix((val2, (row2, col2)), shape=(nv,mf.ndof))
CGEFtot=sp.sparse.csc_matrix((val3, (row3, col3)), shape=(nv,m.ndof))

#%% 
m2.GaussIntegration()
mf.GaussIntegration()

E=100e+3 ; v=0.3
hooke=E/(1-v**2)*np.array([[1,v,0],[v,1,0],[0,0,(1-v)/2]])

KGEF=m2.Stiffness(hooke)
KLEF=mf.Stiffness(hooke)

KK=sp.sparse.bmat([[KGEF, None, CGEF.T ],
                   [None, KLEF, CLEF.T ],
                   [CGEF, CLEF, None   ]])
KK=KK.tocsc()
ndof=KK.shape[0]
UU=np.zeros(ndof)

#rep1=px.SelectMeshLine(m2)
# rep1=np.array([   0,   49,   74,  123,  148,  197,  222,  271,  296,  345,  370, \
#        419,  444,  493,  518,  567,  592,  641,  666,  715,  740,  789, \
#        814,  863,  888,  937,  962, 1011, 1036, 1085, 1110, 1159, 1184])
rep1=np.array([   0,   49,   98,  147,  196,  245,  294,  343,  392,  441,  490,
        539,  588,  637,  686,  735,  784,  833,  882,  931,  980, 1029,
       1078, 1127, 1176, 1225, 1274, 1323, 1372, 1421, 1470, 1519, 1568])
rep11=m2.conn[rep1]
rep=np.arange(ndof)
rep=np.delete(rep,rep11)
repk=np.ix_(rep,rep)

# selection des noeuds du haut pour force
#rep2=px.SelectMeshLine(m2)
rep2=np.array([  48,   73,  122,  147,  196,  221,  270,  295,  344,  369,  418, \
        443,  492,  517,  566,  591,  640,  665,  714,  739,  788,  813, \
        862,  887,  936,  961, 1010, 1035, 1084, 1109, 1158, 1183, 1232])
rep22=m2.conn[rep2]
FF=np.zeros(ndof)
FF[rep22[:,1]]=100


KLU=splalg.splu(KK[repk])
UU[rep]=KLU.solve(FF[rep])

repG=np.arange(m2.ndof)
repL=np.arange(m2.ndof,m2.ndof+mf.ndof)
UGEF=UU[repG]
ULEF=UU[repL]

m2.Plot(edgecolor='y')
mf.Plot(edgecolor='y')

m2.Plot(UGEF,2,edgecolor='k')
mf.Plot(ULEF,2,edgecolor='k')

m2.VTKSol('marie_glob',UGEF)
mf.VTKSol('marie_loc',ULEF)

#%%

E=100e+3 ; v=0.3
hooke=E/(1-v**2)*np.array([[1,v,0],[v,1,0],[0,0,(1-v)/2]])



#(val,row,col,siz)=np.load('poutrecourbeKIGA.npy')
#KIGA=spsp.csc_matrix((val, row, col), shape=siz)
#(val,row,col,siz)=np.load('poutrecourbeKBIGA.npy')
#KBIGA=spsp.csc_matrix((val, row, col), shape=siz)

import Geometries as g
from pyNURBS import Mechanalysis
nspanu = 6 ## Rq : pour la plaque trouée, on a 2*nspanu
nspanv = 4
cspanu = 4 # nspanu_analysis = cspanu*nspanu
cspanv = 4
ppu = 2
ppv = 2
a=5;b=10
#m = g.geo_poutre2D_ct_curve(a,b,nspanu,nspanv,ppu,ppv) ### poutre courbe
m_analysis = g.geo_poutre2D_ct_curve(a,b,nspanu*cspanu,nspanv*cspanv,ppu,ppv) ### poutre courbe fin

analysis = Mechanalysis(m_analysis,hooke)
analysis.Stiffness(np.arange(384))
KIGA=analysis.K
analysis.Stiffness(er)
KBIGA=analysis.K

Btot = m_analysis.Get_Btot()
plt.plot(Btot[0,:],Btot[1,:],'ko')
plt.axis('equal')

#m_analysis.SelectMeshNodes()
rep1=np.array([[ 25,  51,  77, 103, 129, 155, 181, 207, 233, 259, 285, 311, 337, 363, 389, 415, 441, 467], \
       [493, 519, 545, 571, 597, 623, 649, 675, 701, 727, 753, 779, 805, 831, 857, 883, 909, 935]])
rep2=np.array([[  0,  26,  52,  78, 104, 130, 156, 182, 208, 234, 260, 286, 312, 338, 364, 390, 416, 442], \
               [468, 494, 520, 546, 572, 598, 624, 650, 676, 702, 728, 754, 780, 806, 832, 858, 884, 910]])
ndofiga=KIGA.shape[0]
U=np.zeros(ndofiga)
F=np.zeros(ndofiga)
F[rep1[1]]=10

F[rep1[1]]= np.array([3.3333333,6.6667,10.0000,10.0000,10.0000, 10.0000,10.0000, 10.0000,10.0000, 10.0000,10.0000, 10.0000,10.0000, 10.0000,10.0000, 10.0000, 6.6667, 3.3333])


rep=np.arange(ndofiga)
rep=np.delete(rep,rep2.ravel())
repk=np.ix_(rep,rep)

IGALU=splalg.splu(KIGA[repk])
U[rep]=IGALU.solve(F[rep])

m_analysis.VTKFull(1e-7,hooke,30,U,'toto')


#%% EF

mf.GaussIntegration()
KLEF=mf.Stiffness(hooke)
KL=sp.sparse.bmat([[KLEF, CLEF.T ],
                   [CLEF, None   ]])
EFLU=splalg.splu(KL)



#%%

ndof_interf=CLEF.shape[0]
UG   =np.zeros(ndofiga)
UGold=np.zeros(ndofiga)
LAM=np.zeros(ndof_interf)
(Dl,n,e)=np.load('poutrecourbeQ9.npy',allow_pickle=True)
Dl=Dl.toarray()
AA=np.zeros((m.ndof//2,m.ndof//2))
for i in range(m.ndof//2):
    AA[i,m.conn[i,0]]=1
Dl=Dl.dot(AA)

Dl=spsp.csc_matrix(Dl)
DDLL=sp.sparse.bmat([[Dl  , None], \
                     [None, Dl  ]])

CG = CGEFtot.dot(DDLL.T)

for ifp in range(30):
    print('iter #'+str(ifp))
    #% PB1
    FG=F - CG.T.dot(LAM) + KBIGA.dot(UGold)
    UG[rep]=IGALU.solve(FG[rep])
    #m_analysis.VTKFull(1e-7,hooke,30,UG,'toto')
    #print(np.linalg.norm(UG))
    #UEF=DDLL.T.dot(UG)
    #m.Plot(color='y')
    #m.Plot(UEF)
    if ifp==0:
        err=np.array([1])
    else:
        err=np.r_[err,np.linalg.norm(UG-UGold)/np.linalg.norm(UG)]

    #% PB2
    FL=np.zeros(mf.ndof+ndof_interf)
    FL[mf.ndof:]=-CG.dot(UG)
    UL=EFLU.solve(FL)
    LAM=UL[mf.ndof:]
    #print(np.linalg.norm(UL[:mf.ndof]))
    #mf.VTKSol('marie_loc',UL[:mf.ndof])
    UGold=UG.copy()

plt.semilogy(err,'k.-')
plt.grid('on')
plt.xlabel('iteration number',fontsize=14)
plt.ylabel('Stagnation',fontsize=14)
plt.gca().xaxis.grid(True, which='minor')

m_analysis.VTKFull(1e-7,hooke,30,UG,'toto')
mf.VTKSol('marie_loc',UL[:mf.ndof])
    
UEF=DDLL.T.dot(UG)
m.Plot(color='y')
#mf.Plot(color='r')
m.Plot(UEF,10)
mf.Plot(UL[:mf.ndof],10,edgecolor='r')






#%%

roi=np.array([[0,0],[400,40]])
m=px.StructuredMeshT3(roi,10)
m.Plot()
m.PlotElemLabels()
m.PlotNodeLabels()


#%%

p=50
l=10
x=np.linspace(0,l,100)
n1=x/l
n2=1-x/l
plt.plot(x,n1)
plt.plot(x,n2)

f=n1*p*l*0.5+n2*p*l*0.5

plt.plot(x,f)

    """
#%% test rigidite interieur + grande (pas fini)

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as splalg
import scipy as sp
import pyxel as px

mg=px.ReadMeshINP('./data/dic_composite/olfa3.inp')
ml=mg.Copy()
mg.Plot()
mg.Connectivity()
mg.GaussIntegration()
E=100e+3 ; v=0.3
hooke=E/(1-v**2)*np.array([[1,v,0],[v,1,0],[0,0,(1-v)/2]])
KG=mg.Stiffness(hooke)

c=np.array([0.015,0.025])
r=0.08




KL=sp.sparse.bmat([[KLEF, CLEF.T ],
                   [CLEF, None   ]])
"""