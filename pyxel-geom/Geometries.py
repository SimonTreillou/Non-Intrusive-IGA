# NURBS GEOMETRY

import numpy as np
import pyNURBS as pn

import scipy.sparse as sps


###############################################################################
#%% Curved beam ###############################################################
###############################################################################

def geo_quart_disque(a,b,nspanu,nspanv,ppu,ppv):

    #Maillage initial
    #=====================
    
    # ordre des fonctions nurbs initial
    ppu0 = 2
    ppv0 = 1
    
    # knot vectors initiaux
    XXsiu = np.array([0.,0.,0.,1.,1.,1.])
    XXsiv = np.array([0.,0.,1.,1.])
    
#    # points de contrôle (cntrl(4,u,v,w)) (rq : pas plus de 100 /diviser sinon) DIMENSION D+1
    #ctrlPts0 = np.array([[b,0],[b,b],[0,b],[a,0],[a,a],[0,a]])
    x0 = np.array([[b,b,0],[0,0,0]])
    y0 = np.array([[0,b,b],[0,b/2,b]])
    ### Ajout de poids pour les NURBS
    weight0 = np.array([[1,2**(-1/2),1],[1,2**(-1/2),1]])
#    weight0 = np.ones_like(weight0) # poids de 1 pour B spline
    ctrlPts = np.array([x0,y0,weight0])
    ctrlPts = np.transpose(ctrlPts,[0,2,1]) # pour avoir ctrlPts.shape = dim, nu, nv

    XXsi=dict()
    XXsi[0]=XXsiu
    XXsi[1]=XXsiv
    m=pn.Mesh(ctrlPts,np.array([ppu0,ppv0]),XXsi)
    
    # raffinement
    #=================
    # order elevation
    m.DegElevation(np.array([ppu,ppv]))
#    print('Après élévation de degré : XXsiu = ',str(m.xxsi[0]),'XXsiv = ',str(m.xxsi[1]),'ctrlPts = ',str(m.ctrlPts))
    
    # knot insertion (k-refinement uniforme)
    ubar=dict()
#    ubar[0] = 1/nspanu*np.arange(1,nspanu)
    ubar[0] = np.log10(1+9*np.arange(1,nspanu)/nspanu)
    ubar[1] = 1/nspanv*np.arange(1,nspanv)
    m.KnotInsertion(ubar)  
#    print('Après ajout de nd : XXsiu = ',str(m.xxsi[0]),'XXsiv = ',str(m.xxsi[1]),'ctrlpts = ',str(m.ctrlPts))

    
    # recuperation des donnees
    #============================= 
    nu,nv= m.Get_nunv()
#    print('nu et nv')
#    print(nu,nv)


    # Parametre deduits du probleme
    #===========================================
    # Table de connectivite, nbr elem elem et global
    m.Connectivity()
#    print('IEN')
#    print(m.ien[0],m.ien[1])
    # liste des elements
    # pour assemblage global
#    print('NOELEM')
#    print(m.noelem)
#    print('TRIPLEIEN')
#    print(m.tripleien)
    
    return m

def geo_poutre2D_ct_curve(a,b,nspanu,nspanv,ppu,ppv):

    #Maillage initial
    #=====================
    
    # ordre des fonctions nurbs initial
    ppu0 = 2
    ppv0 = 1
    
    # knot vectors initiaux
    XXsiu = np.array([0.,0.,0.,1.,1.,1.])
    XXsiv = np.array([0.,0.,1.,1.])
    
#    # points de contrôle (cntrl(4,u,v,w)) (rq : pas plus de 100 /diviser sinon) DIMENSION D+1
    #ctrlPts0 = np.array([[b,0],[b,b],[0,b],[a,0],[a,a],[0,a]])
    x0 = np.array([[b,b,0],[a,a,0]])
    y0 = np.array([[0,b,b],[0,a,a]])
    ### Ajout de poids pour les NURBS
    weight0 = np.array([[1,2**(-1/2),1],[1,2**(-1/2),1]])
#    weight0 = np.ones_like(weight0) # poids de 1 pour B spline
    ctrlPts = np.array([x0,y0,weight0])
    ctrlPts = np.transpose(ctrlPts,[0,2,1]) # pour avoir ctrlPts.shape = dim, nu, nv
    import pdb
    pdb.set_trace()
    XXsi=dict()
    XXsi[0]=XXsiu
    XXsi[1]=XXsiv
    m=pn.Mesh(ctrlPts,np.array([ppu0,ppv0]),XXsi)
    
    # raffinement
    #=================
    # order elevation
    m.DegElevation(np.array([ppu,ppv]))
#    print('Après élévation de degré : XXsiu = ',str(m.xxsi[0]),'XXsiv = ',str(m.xxsi[1]),'ctrlPts = ',str(m.ctrlPts))
    
    # knot insertion (k-refinement uniforme)
    ubar=dict()
    ubar[0] = 1/nspanu*np.arange(1,nspanu)
    ubar[1] = 1/nspanv*np.arange(1,nspanv)
    m.KnotInsertion(ubar)  
#    print('Après ajout de nd : XXsiu = ',str(m.xxsi[0]),'XXsiv = ',str(m.xxsi[1]),'ctrlpts = ',str(m.ctrlPts))

    
    # recuperation des donnees
    #============================= 
    nu,nv= m.Get_nunv()
#    print('nu et nv')
#    print(nu,nv)


    # Parametre deduits du probleme
    #===========================================
    # Table de connectivite, nbr elem elem et global
    m.Connectivity()
#    print('IEN')
#    print(m.ien[0],m.ien[1])
    # liste des elements
    # pour assemblage global
#    print('NOELEM')
#    print(m.noelem)
#    print('TRIPLEIEN')
#    print(m.tripleien)
    
    return m


###############################################################################
#%% Rectangular Plate #########################################################
###############################################################################


def RectangularPlate(xmin,xmax,ymin,ymax, ne_xi, ne_eta, degree_xi, degree_eta): 
    
    # Parametric space properties 
    p=1; q=1
    Xi = np.concatenate((np.repeat(0,p+1),np.repeat(1,p+1)))
    Eta = np.concatenate((np.repeat(0,q+1),np.repeat(1,q+1)))
    
    # Control points for a recangular plate  
    x = np.array([[xmin,xmax],
                  [xmin,xmax]]) 
    y  = np.array([[ymin,ymin],
                   [ymax,ymax]]) 
#    x = np.array([[xmin,(xmin+xmax)/2,xmax],
#                  [xmin,(xmin+xmax)/2,xmax]]) 
#    y  = np.array([[ymin,ymin,ymin],
#                   [ymax,ymax,ymax]]) 
    w =  np.ones((q+1,p+1))
 
    ctrlPts = np.array([x,y,w])
    ctrlPts = np.transpose(ctrlPts,[0,2,1])
  
    
    # Dictionary for the knot vector 
    knot_vector=dict()
    knot_vector[0]= Xi
    knot_vector[1]= Eta 
     
    m = pn.Mesh(ctrlPts,np.array([p,q]),knot_vector)
    """ Degree Elevation"""
    m.DegElevation(np.array([degree_xi, degree_eta]))
 
    """ Knot refinement """ 
    ubar=dict()
    ubar[0] = 1/ne_xi*np.arange(1,ne_xi)
    ubar[1] = 1/ne_eta*np.arange(1,ne_eta)
    
    m.KnotInsertion(ubar)
 
    m.Connectivity()
    return m 



###############################################################################
#%% Plate with hole ###########################################################
###############################################################################

def PlateWithHole(L,R,nspanu,nspanv,ppu,ppv,w=0):
    ''' Plaque carrée de côté L et de trou de rayon R'''

    #Maillage initial
    #=====================
    
    # ordre des fonctions nurbs initial
    ppu0 = 2
    ppv0 = 2
    
    # knot vectors initiaux
    XXsiu = np.array([0,0,0,0.5,1,1,1])
    XXsiv = np.array([0.,0.,0.,1.,1.,1.])
    
#    # points de contrôle (cntrl(4,u,v,w)) (rq : pas plus de 100 /diviser sinon) DIMENSION D+1
    x0 = np.array([[-R,-R,R*(1-np.sqrt(2)),0],[-(R+L)/2,-(R+L)/2,(R+L)/2*(1-np.sqrt(2)),0],[-L,-L,-L,0]])
    y0 = np.array([[0,R*(np.sqrt(2)-1),R,R],[0,(R+L)/2*(np.sqrt(2)-1),(R+L)/2,(R+L)/2],[0,L,L,L]])
    ### Ajout de poids pour les NURBS
    weight0 = np.array([[1,(1+1/np.sqrt(2))/2,(1+1/np.sqrt(2))/2,1],[1,1,1,1],[1,1,1,1]])
    if w==1:
        weight0 = np.array([[1,1,1,1],[1,1,1,1],[1,1,1,1]])
    ctrlPts = np.array([x0,y0,weight0])
    ctrlPts = np.transpose(ctrlPts,[0,2,1]) # pour avoir ctrlPts.shape = dim, nu, nv
    
    XXsi=dict()
    XXsi[0]=XXsiu
    XXsi[1]=XXsiv
    m=pn.Mesh(ctrlPts,np.array([ppu0,ppv0]),XXsi)
    
    # raffinement
    #=================
    # order elevation
    m.DegElevation(np.array([ppu,ppv]))
#    print('Après élévation de degré : XXsiu = ',str(m.xxsi[0]),'XXsiv = ',str(m.xxsi[1]),'ctrlPts = ',str(m.ctrlPts))
    
    # knot insertion (k-refinement uniforme)
    ubar=dict()
    tmp1 = 1/nspanu*np.arange(1,nspanu)*0.5
    tmp2 = 1/nspanu*np.arange(1,nspanu)*0.5+0.5
    ubar[0] = np.r_[tmp1,tmp2]    
    ubar[1] = 1/nspanv*np.arange(1,nspanv)
    m.KnotInsertion(ubar)  
#    print('Après ajout de nd : XXsiu = ',str(m.xxsi[0]),'XXsiv = ',str(m.xxsi[1]),'ctrlpts = ',str(m.ctrlPts))

    
    # recuperation des donnees
    #============================= 
    nu,nv= m.Get_nunv()
#    print('nu et nv')
#    print(nu,nv)


    # Parametre deduits du probleme
    #===========================================
    # Table de connectivite, nbr elem elem et global
    m.Connectivity()
#    print('IEN')
#    print(m.ien[0],m.ien[1])
    # liste des elements
    # pour assemblage global
#    print('NOELEM')
#    print(m.noelem)
#    print('TRIPLEIEN')
#    print(m.tripleien)

    
    return m





###############################################################################
#%% Plate with hole ###########################################################
###############################################################################

def PlateWithHoleInit(L,a,nspanu,nspanv,ppu,ppv):
    '''Plaque de côté L avec un trou dans l'angle de côté a (surface ôtée : (a^2)/2)'''

    #Maillage initial
    #=====================
    
    # ordre des fonctions nurbs initial
    ppu0 = 2
    ppv0 = 2
    
    # knot vectors initiaux
    XXsiu = np.array([0,0,0,0.5,1,1,1])
    XXsiv = np.array([0.,0.,0.,1.,1.,1.])
    
#    # points de contrôle (cntrl(4,u,v,w)) (rq : pas plus de 100 /diviser sinon) DIMENSION D+1
    x0 = np.array([[-a,-3/4*a,-a/4,0],[-(a+L)/2,-(a+L)/2,(a+L)/2*(1-np.sqrt(2)),0],[-L,-L,-L,0]])
    y0 = np.array([[0,a/4,3*a/4,a],[0,(a+L)/2*(np.sqrt(2)-1),(a+L)/2,(a+L)/2],[0,L,L,L]])
#    x0 = np.array([[-a,-2/3*a,-a/3,0],[-(a+L)/2,-(a+L)/2,(a+L)/2*(1-np.sqrt(2)),0],[-L,-L,-L,0]])
#    y0 = np.array([[0,a/3,2*a/3,a],[0,(a+L)/2*(np.sqrt(2)-1),(a+L)/2,(a+L)/2],[0,L,L,L]])
    ### Ajout de poids pour les NURBS
    weight0 = np.array([[1,(1+1/np.sqrt(2))/2,(1+1/np.sqrt(2))/2,1],[1,1,1,1],[1,1,1,1]])
    weight0 = np.array([[1,1,1,1],[1,1,1,1],[1,1,1,1]])
    ctrlPts = np.array([x0,y0,weight0])
    ctrlPts = np.transpose(ctrlPts,[0,2,1]) # pour avoir ctrlPts.shape = dim, nu, nv
    
    XXsi=dict()
    XXsi[0]=XXsiu
    XXsi[1]=XXsiv
    m=pn.Mesh(ctrlPts,np.array([ppu0,ppv0]),XXsi)
    
    # raffinement
    #=================
    # order elevation
    m.DegElevation(np.array([ppu,ppv]))
#    print('Après élévation de degré : XXsiu = ',str(m.xxsi[0]),'XXsiv = ',str(m.xxsi[1]),'ctrlPts = ',str(m.ctrlPts))
    
    # knot insertion (k-refinement uniforme)
    ubar=dict()
    tmp1 = 1/nspanu*np.arange(1,nspanu)*0.5
    tmp2 = 1/nspanu*np.arange(1,nspanu)*0.5+0.5
    ubar[0] = np.r_[tmp1,tmp2]    
    ubar[1] = 1/nspanv*np.arange(1,nspanv)
    m.KnotInsertion(ubar)  
#    print('Après ajout de nd : XXsiu = ',str(m.xxsi[0]),'XXsiv = ',str(m.xxsi[1]),'ctrlpts = ',str(m.ctrlPts))

    
    # recuperation des donnees
    #============================= 
    nu,nv= m.Get_nunv()
#    print('nu et nv')
#    print(nu,nv)


    # Parametre deduits du probleme
    #===========================================
    # Table de connectivite, nbr elem elem et global
    m.Connectivity()
#    print('IEN')
#    print(m.ien[0],m.ien[1])
    # liste des elements
    # pour assemblage global
#    print('NOELEM')
#    print(m.noelem)
#    print('TRIPLEIEN')
#    print(m.tripleien)
    
    return m
















