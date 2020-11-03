
import numpy as np
import nurbs as nb
import scipy.sparse as sps
import matplotlib.pyplot as plt
import scipy.sparse.linalg as splalg
import CalculusToolBox as ctb

#import ctypes 

import vtktools as vtk
import os as os


##### Si travail sous LINUX, plutôt activer cette fonction (avec le import ctypes), et commenter les fonctions tout en bas qui la remplacent
##gcc -shared file.c -fPIC -o output.so
#""" Method for evaluating the basis functions and their derivatives imported from C code """
#_foolib = ctypes.cdll.LoadLibrary('./c_library/c_routines.so')
#dersbasisfuns = _foolib._vectorInput_dersbasisfuns
#array_type = np.ctypeslib.ndpointer(dtype= ctypes.c_double, ndim=1, flags='C_CONTIGUOUS')
#dersbasisfuns.restype = None  
#dersbasisfuns.argtypes = [ctypes.c_int,array_type,array_type,ctypes.c_int, ctypes.c_int, ctypes.c_int, array_type  ]
#
#def derbasisfuns_c(p,U,u,nb_u_values,span,nders):
#    ders = np.zeros((nders+1)*(p+1)*(nb_u_values), dtype = ctypes.c_double) 
#    dersbasisfuns(p,U,u,nb_u_values,span,nders,ders)
#    ders.resize(((nb_u_values*(nders+1)),p+1))
#    N = ders[::2,:]
#    dN = ders[1::2,:]
#    return N,dN 


class Mesh:
    def __init__(self,ctrlPts,pp,xxsi):
        self.ctrlPts=ctrlPts    # Inhomogeneous Control Points Coordinates and weights
        self.pp=pp              # degree [ppu,ppv]
        self.xxsi=xxsi          # knot vector p-uplet (XXsiu,XXsiv)
        self.ien=0              # NURBS Connectivity p-uplet (IENu,IENv)
        self.noelem=0           # Connectivity: Control Point number (column) of each element (line)
        self.tripleien=0        # Correspondance between 2D elements and 1D elements
        self.iperM=0            # Sparse matrix for coincident control points
        
        
        self.wdetJmes = 0     # Dictionary containing the value det(Jacobian)*Gauss_Weights*ElementMeasure/4 
        self.mes = 0         # Dictionary containing the measure of the elements in the ordre of listeltot
        
        self.phi = 0         # Dictionary of basis functions evaluated at gauss points 
        self.dphidx = 0      # Dictionary containing the derivative of Basis function in x direction
        self.dphidy = 0      # Dictionary containing the derivative of Basis function in y direction
        
        self.phi_xi = 0     # Dictionary of basis functions in xi direction evaluated at univariate gauss point 
        self.phi_eta = 0    # Dictionary of basis functions in eta direction evaluated at univariate gauss point
        self.dphi_xi = 0
        self.dphi_eta = 0

        
    def Copy(self):
        mesh = Mesh(self.ctrlPts.copy(),self.pp.copy(),self.xxsi.copy())
        mesh.ien = self.ien.copy()
        mesh.noelem = self.noelem.copy()
        mesh.tripleien = self.tripleien.copy()
        mesh.iperM = self.iperM.copy()
        return mesh
        
    def Get_nunv(self):
        return self.ctrlPts.shape[1:]
    def Get_Btot(self):
        P=np.c_[self.ctrlPts[0].ravel(order='F'),self.ctrlPts[1].ravel(order='F'),self.ctrlPts[2].ravel(order='F')]            
        if self.ctrlPts.shape[0]==4:
            P=np.c_[P,self.ctrlPts[3].ravel(order='F')]
        return P.T
    def Get_listeltot(self):
        return np.arange(self.ien[0].shape[1]*self.ien[1].shape[1])   ### erreur IEN ici
    def Get_listBtot(self):
        return np.arange(self.Get_Btot().shape[1])      ### erreur Btot ici
    def Get_nenunv(self):
        return self.pp+1
    def Get_spacedim(self):
        """dimension of the space in which the structure is. Not necessarily equal to the structure dimension"""
        return self.ctrlPts.shape[0]-1
    def Get_dim(self):
        """structure dimension"""
        return len(self.xxsi)
    def Get_ndof(self):
        """ Number of degrees of freedom"""
        nbf_uv =  self.Get_nunv()
        nbf =  nbf_uv[0]*nbf_uv[1]
        return self.Get_dim()*nbf 
    def Mes_Elts(self):
        self.mes = np.outer(nb.MesXsi(self.xxsi[1],self.pp[1]),nb.MesXsi(self.xxsi[0],self.pp[0])).ravel()
    def IsRational(self):
        """ Returns True if the B-spline is rational"""
        if self.ctrlPts.shape[0] == 3:
            return (self.ctrlPts[2]!=1).any()
        if self.ctrlPts.shape[0] == 4:
            return (self.ctrlPts[3]!=1).any()
    def DegElevation(self,ppnew):
        ''' INPUT target degree, example: ppnew=[2,3]
        '''
        t = ppnew - self.pp
        # to homogeneous coordinates
        self.ctrlPts[:-1,:,:] = self.ctrlPts[:-1,:,:]*self.ctrlPts[-1,:,:]
        # NURBS represents a surface
        dim_sp = self.Get_spacedim()        # dimension de l'espace physique (2 si on utilise x,y)
        dim = self.Get_dim()        # dimension de l'espace physique (2 si on utilise x,y)
        if dim==2: # 2D
            num1 = self.ctrlPts.shape[1]       # nombre de points de controle selon u (nu)
            num2 = self.ctrlPts.shape[2]       # nombre de points de controle selon v (nv)
            # Degree elevate along the v direction
            if t[1] != 0:
                coefs = self.ctrlPts.reshape(((dim_sp+1)*num1,num2),order="F")  # fait un tableau 2D pour pouvoir utiliser bspdegelev
                self.ctrlPts,self.xxsi[1] = nb.bspdegelev(self.pp[1],coefs,self.xxsi[1],t[1])
                num2 = self.ctrlPts.shape[1]                             # nouvelle valeur de nv (on a ajouté des pts de controle)
                self.ctrlPts = self.ctrlPts.reshape(((dim_sp+1),num1,num2),order="F") # on remet sous la forme de matrice 3D
            # Degree elevate along the u direction
            if t[0] != 0:
                coefs = np.transpose(self.ctrlPts,(0,2,1))   # pour pouvoir utiliser bspdegelev, on met la dimension sur laquelle on travaille (u) au bon endroit
                coefs = coefs.reshape(((dim_sp+1)*num2,num1),order="F")
                self.ctrlPts,self.xxsi[0] = nb.bspdegelev(self.pp[0],coefs,self.xxsi[0],t[0])
                num1 = self.ctrlPts.shape[1]
                self.ctrlPts = self.ctrlPts.reshape(((dim_sp+1),num2,num1),order="F")
                self.ctrlPts = np.transpose(self.ctrlPts,(0,2,1)) # on remet les dimensions dans le bon ordre
        else:
            print('3D not yet implemented!')
        self.pp=ppnew
        # back to real coordinates
        self.ctrlPts[:-1,:,:] = self.ctrlPts[:-1,:,:]/self.ctrlPts[-1,:,:]
        
    def KnotInsertion(self,u_refinement):
        '''
        INPUT
        ubar=dict()
        ubar[0] = ubaru
        ubar[1] = ubarv
        '''
        # to homogeneous coordinates
        self.ctrlPts[:-1,:,:] = self.ctrlPts[:-1,:,:]*self.ctrlPts[-1,:,:]
        # NURBS represents a surface
        dim_sp = self.Get_spacedim()        # dimension de l'espace physique (2 si on utilise x,y)
        dim = self.Get_dim()        # dimension de l'espace physique (2 si on utilise x,y)
        if dim==2: # 2Dnum1 = ctrlPts.shape[1] # nombre de points de controle selon u (nu)
            num1 = self.ctrlPts.shape[1] # nombre de points de controle selon u (nu)
            num2 = self.ctrlPts.shape[2] # nombre de points de controle selon v (nv)
            nu = np.size(u_refinement[0])
            nv = np.size(u_refinement[1])
            # Degree elevate along the v direction
            if nv != 0:
                coefs = self.ctrlPts.reshape(((dim_sp+1)*num1,num2),order="F") # fait un tableau 2D pour pouvoir utiliser bspdegelev
                self.ctrlPts,self.xxsi[1] = nb.bspkntins(self.pp[1],coefs,self.xxsi[1],u_refinement[1])
                num2 = self.ctrlPts.shape[1] # nouvelle valeur de nv (on a ajouté des pts de controle)
                self.ctrlPts = self.ctrlPts.reshape(((dim_sp+1),num1,num2),order="F") # on remet sous la forme de matrice 3D
                # Degree elevate along the u direction
            if nu != 0:
                coefs = np.transpose(self.ctrlPts,(0,2,1)) # pour pouvoir utiliser bspdegelev, on met la dimension sur laquelle on travaille (u) au bon endroit
                coefs = coefs.reshape(((dim_sp+1)*num2,num1),order="F")
                self.ctrlPts, self.xxsi[0] = nb.bspkntins(self.pp[0], coefs, self.xxsi[0], u_refinement[0])
                num1 = self.ctrlPts.shape[1]
                self.ctrlPts = self.ctrlPts.reshape(((dim_sp+1),num2,num1),order="F")
                self.ctrlPts = np.transpose(self.ctrlPts,(0,2,1)) # on remets les dimensions dans le bon ordre
        # back to real coordinates
        self.ctrlPts[:-1,:,:] = self.ctrlPts[:-1,:,:]/self.ctrlPts[-1,:,:]
    
    def HrefUbar(self,n): # A concaténer avec code de knot insertion précédent ?
        """n[0] : en combien on coupe chaque elt de xsi de taille non nulle / n[1] : idem eta"""
        xsi_unique = np.unique(self.xxsi[0])
        eta_unique = np.unique(self.xxsi[1])
        eltXsi_size = xsi_unique[1:]-xsi_unique[:-1]
        eltEta_size = eta_unique[1:]-eta_unique[:-1]
        discru = 1/n[0]*np.arange(1,n[0])
        discrv = 1/n[1]*np.arange(1,n[1])
        ubar=dict()
        ubar[0] = (xsi_unique[:-1]+np.outer(discru,eltXsi_size)).T.ravel()
        ubar[1] = (eta_unique[:-1]+np.outer(discrv,eltEta_size)).T.ravel()
#        self.KnotInsertion(ubar)
        return ubar
    
    def BezierUbar(self):
        ubar = dict()
        ubar[0] = nb.Xsi_Bezier(self.xxsi[0],self.pp[0])
        ubar[1] = nb.Xsi_Bezier(self.xxsi[1],self.pp[1])
        return ubar
    

    def Href_matrix(self,u_ref):
        """Renvoie la matrice C qui fait C.T * B.T = Bnew.T
        avec B la matrice dim*(nu*nv) des points de contôles HOMOGENES
             Bnew la nouvelle matrice dim*(nouveau nb de pts de ctrl) HOMOGENES
             * la multiplication matricielle"""
        C_xsi,xsi = nb.Href_matrix(self.pp[0],self.xxsi[0],u_ref[0])
        C_eta,eta = nb.Href_matrix(self.pp[1],self.xxsi[1],u_ref[1])
        C = sps.kron(C_eta,C_xsi)
        return C

    def Pref_matrix(self,t):
        """Renvoie la matrice C qui fait C.T * B.T = Bnew.T
        avec B la matrice dim*(nu*nv) des points de contôles HOMOGENES
             Bnew la nouvelle matrice dim*(nouveau nb de pts de ctrl) HOMOGENES pour le degré souhaité
             * la multiplication matricielle"""
        C_xsi,xsi,p_xsi = nb.Pref_matrix(self.xxsi[0],self.pp[0],t[0])
        C_eta,eta,p_eta = nb.Pref_matrix(self.xxsi[1],self.pp[1],t[1])
        C = sps.kron(C_eta,C_xsi)
        return C
    
    def Matrix_N2ToL2(self,type_Q):
        """BL.T = C.T*BN.T"""
        C_xsi = nb.Matrix_N2ToL2(self.xxsi[0],self.pp[0])
        C_eta = nb.Matrix_N2ToL2(self.xxsi[1],self.pp[1])
        C = sps.kron(C_eta,C_xsi)
        # Pour passer de Q9 à Q8
        if type_Q=='Q8':
            mBezier = self.Copy()
            mBezier.KnotInsertion(mBezier.BezierUbar())
            mBezier.Connectivity()
            mBezier.Mes_Elts()
            ind_to_delete = mBezier.noelem[np.where(mBezier.mes !=0),4]
            listBtot = mBezier.Get_listBtot()
            n = len(listBtot)
            listBLtot = np.delete(listBtot,ind_to_delete)
            m = len(listBLtot)
            M = sps.csc_matrix((np.ones_like(listBLtot),(np.arange(len(listBLtot)),listBLtot)),shape=(m,n))
            C = C.dot(M.T)
        return C
    
    def ToLagrange2(self,type_Q):
        """returns nodes coordinates P_L and elements connectivity table E_L for FE"""
        if self.pp[0]!=2 or self.pp[1]!=2:
            print('Functions degrees must be 2 for ToLagrange2 to work')
        # P_N = self.Get_Btot() # Pts de contrôle B spline
        mesh = self.Copy()
        mesh.KnotInsertion(self.BezierUbar())
        # Rq : papier de Marie dit qu'opérateur pas calculé en entier (+voir papier 32 pour algo). Partie suivante à modifier si mieux sans tout calculer
        # Rq2 : Comme pour le raffinement, il vaudrait pê mieux une fonction qui garde en mémoire les points qui ne bougent pas au cours de l'optim
        P_B = mesh.Get_Btot() # pts de ctrl Bernstein
        dim_sp = mesh.Get_spacedim()
        P_L = nb.Bezier2ToLagrange2(P_B[:-1].reshape((dim_sp*mesh.Get_nunv()[1],-1))); P_L = P_L.reshape((dim_sp,-1)) # Pour dir. XSI
        P_L = nb.Bezier2ToLagrange2(P_L.reshape((dim_sp*mesh.Get_nunv()[0],-1),order='F')).reshape((dim_sp,-1),order='F') # Pour dir. ETA
        ### Calcul de la table de connectivité pour élts (de taille non nulle / si pb, supprimer les points confondus au préalable)
        mesh.Connectivity()
        mesh.Mes_Elts()
        ### VERSION Q8
        if type_Q == 'Q8':
            ### Passage à Q8
            a_enlever = mesh.noelem[np.where(mesh.mes)][:,4] # vecteur des noeuds centre Q9 ###!
            num_ddl = mesh.Get_listBtot() ###!
            num_ddl = np.delete(num_ddl,a_enlever) ###!
            P_L = np.delete(P_L,a_enlever,axis=1) ###!
    #        E_L = np.zeros((mesh.Get_listeltot().size,8)) #version array
            E_L = dict() # version dict
            indices = np.array([0,2,8,6,1,5,7,3])
            e=0
            for elt in mesh.Get_listeltot():
                if mesh.mes[elt]!=0:
    #                E_L[e]=mesh.noelem[elt,indices] #version array
                    nds = np.array([],dtype=np.int)
                    for ind in indices:
                        num, = np.where(num_ddl==mesh.noelem[elt,ind])
                        nds = np.r_[nds,num]
                    E_L[e]=np.r_[16,nds] #version dict / 16 pour Q8
                    e+=1
    #            else:                                 #version array
    #                E_L = np.delete(E_L,-1,axis=0)    #version array
        ### VERSION Q9
        elif type_Q == 'Q9':
            E_L = dict() # version dict
            indices = np.array([0,2,8,6,1,5,7,3,4])
            e=0
            for elt in mesh.Get_listeltot():
                if mesh.mes[elt]!=0:
    #                E_L[e]=mesh.noelem[elt,indices] #version array
                    E_L[e]=np.r_[10,mesh.noelem[elt,indices]] #version dict / 10 pour Q9
                    e+=1
    #            else:                                 #version array
    #                E_L = np.delete(E_L,-1,axis=0)    #version array

        return P_L,E_L

    def Connectivity(self):
#        import pdb; pdb.set_trace()
        dim_sp = self.Get_spacedim()        # dimension de l'espace physique (2 si on utilise x,y)
        dim = self.Get_dim()        # dimension de la structure
        self.ien=dict()
        for i in range(dim):
            self.ien[i] = nb.nubsconnect(self.pp[i],self.Get_nunv()[i])
        if dim==2:
            ''' connect_2D(IENu,IENv,nu) '''
            nu=self.Get_nunv()[0]
            # 4 boucles for : marche avec input : nelx,nely,nenx,neny,nx   ### Attention : ancienne version pas avec indices Python qui partent de 0
            #    NOELEM = np.zeros((nely*nelx,neny*nenx),dtype='int')
            #    for j in range(nely):   #1:nely
            #        for i in range(nelx):   #1:nelx
            #            for jj in range(neny):   #1:neny
            #                for ii in range (nenx):  #1:nenx
            #                    NOELEM[i+j*nelx,ii+jj*nenx]=ii+1+nx*jj+i+j*nx
            decal = np.fliplr(self.ien[1].T)*nu
            comptage = np.fliplr(self.ien[0].T)
            tramev = np.ones_like(decal)
            trameu = np.ones_like(comptage)
            self.noelem = np.kron(tramev,comptage)+np.kron(decal,trameu)
            ''' IEN_2D(nelx,nely,listelx,listely) '''
            listelx=np.arange(self.ien[0].shape[1])
            listely=np.arange(self.ien[1].shape[1])
            xx,yy = np.meshgrid(listelx,listely)
            self.tripleien = np.array([xx.ravel(),yy.ravel()]).T
            '''IPERm'''
            Btot=self.Get_Btot()
#            MasterPts,indices = np.unique(Btot[:-1,:].T,axis=0,return_inverse = True) #####tol
#            MasterPts,indices = np.unique(ctb.round_to_n(Btot[:-1,:],12).T,axis=0,return_inverse = True) ##### remplacé par ctb.UniqueStable
            indices,final_length = ctb.UniqueStable(Btot[:-1,:].T)
#            print('TEST')
#            print(indices)
#            IPERm  = sps.csc_matrix((np.ones(len(indices)), (np.arange(len(indices)),indices)), shape = (len(indices),MasterPts.shape[0] ))
            IPERm  = sps.csc_matrix((np.ones(len(indices)), (np.arange(len(indices)),indices)), shape = (len(indices),final_length ))

            self.iperM = sps.kron(np.eye(2),IPERm) # 2 ddl par point de contrôle
        else: 
            print('not yet implemented in 3D!')
    def CopyConnectivity(self,mesh):
        self.ien = mesh.ien.copy()
        self.iperM = mesh.iperM.copy()
        self.noelem = mesh.noelem.copy()
        self.tripleien = mesh.tripleien.copy()
######################### Code Ali
    def GaussIntegration(self,nbg=0):
        """ Build the integration scheme """
        # Gauss quadrature points and weights  for domain elements 
        if nbg==0:
            nbg_xi  = self.pp[0]+1 
            nbg_eta = self.pp[1]+1  
        else: # si le nombre de points de Gauss est précisé
            nbg_xi = nbg
            nbg_eta = nbg
        Gauss_xi  =  nb.GaussLegendre(nbg_xi) # positions pts Gauss dans -1,1 et poids, pour XI
        Gauss_eta = nb.GaussLegendre(nbg_eta) # idem pour ETA
        w_g = np.kron(Gauss_eta[1], Gauss_xi[1])
        ones_xi  = np.ones((nbg_xi,1)) # Useful for kronecker product ( similar to tile and repeat) 
        ones_eta = np.ones((nbg_eta,1))
 
 
        
        self.phi = dict()
        self.dphidx = dict()
        self.dphidy = dict()
        self.wdetJmes = dict() 
        self.mes = dict()
        
        listel_tot = self.Get_listeltot()
        P = self.Get_Btot() 
        
        self.phi_xi = dict()
        self.phi_eta = dict()
        self.dphi_xi = dict()
        self.dphi_eta = dict()
        i_xi = 0 ; i_eta = 0
        
 
        isRational = self.IsRational()
        
        if isRational == True :

            for e in listel_tot: 
                ne_xi  = self.tripleien[e,0]
                ni_xi  = self.ien[0][0,ne_xi]
                ne_eta = self.tripleien[e,1]
                ni_eta = self.ien[1][0,ne_eta]
                
                xi_min  = self.xxsi[0][ni_xi] ; xi_max  = self.xxsi[0][ni_xi+1] # bornes de l'elt e selon XI
                eta_min = self.xxsi[1][ni_eta]; eta_max = self.xxsi[1][ni_eta+1] # idem ETA
                # Treating elements of non zero measures only 
                mes_xi = xi_max-xi_min
                mes_eta = eta_max-eta_min
                self.mes[e] = mes_xi*mes_eta
                if self.mes[e] != 0: ### SI ELT DE TAILLE NON NULLE
                    
                    # Mapping to the knot span 
                    xi_p   = xi_min  + 0.5*(Gauss_xi[0]+1)*mes_xi  # position des points de Gauss dans esp isoparam
                    eta_p  = eta_min + 0.5*(Gauss_eta[0]+1)*mes_eta

                    # Basis functions and dertivatives evaluated on Gauss points 
                    Nxi  , dNxidxi    = derbasisfuns_c(self.pp[0],self.xxsi[0],xi_p,nbg_xi,ni_xi,1)
                    Neta , dNetadeta  = derbasisfuns_c(self.pp[1],self.xxsi[1],eta_p,nbg_eta,ni_eta,1)
                    
                    
                    r_Nxi       = np.kron(ones_eta,Nxi)
                    r_dNxidxi   = np.kron(ones_eta,dNxidxi)
                    r_Neta      = np.kron(Neta, ones_xi )
                    r_dNetadeta = np.kron(dNetadeta, ones_xi)
                    
                    Neta_Dot_Nxi      = np.einsum('nk,nl->nkl',r_Neta,r_Nxi).reshape(r_Neta.shape[0],-1) # k et l pour un kronecker : nb lignes = nb de points de Gauss en 2D / nb colonnes = nb de fonctions 2D non nulles sur l'élt
                    Neta_Dot_dNxidxi  = np.einsum('nk,nl->nkl',r_Neta,r_dNxidxi).reshape(r_Neta.shape[0],-1)
                    dNetadeta_Dot_Nxi = np.einsum('nk,nl->nkl',r_dNetadeta,r_Nxi).reshape(r_dNetadeta.shape[0],-1) 
      
                    
                    num_vector     = Neta_Dot_Nxi*P[2,self.noelem[e,:]]
#                    print('numvector ',num_vector)
                    num_vector_xi  = Neta_Dot_dNxidxi*P[2,self.noelem[e,:]]
                    num_vector_eta = dNetadeta_Dot_Nxi*P[2,self.noelem[e,:]]
                    
                    denom     = np.sum(num_vector, axis = -1 )
                    denom_xi  = np.sum(num_vector_xi, axis = -1)
                    denom_eta = np.sum( num_vector_eta, axis = -1)
            
                    denom_g           =  sps.diags(denom)
                    inv_squ_denom_g   =  sps.diags(1./denom**2)
                    
                    dNdxi  = inv_squ_denom_g.dot  ( denom_g.dot(num_vector_xi)  - sps.diags(denom_xi).dot(num_vector)  ) 
                    dNdeta = inv_squ_denom_g.dot  ( denom_g.dot(num_vector_eta) - sps.diags(denom_eta).dot(num_vector) ) 
                    
                    
                    # Jacobian elements 
                    dxdxi  = dNdxi.dot(P[0,self.noelem[e,:]])
                    dxdeta = dNdeta.dot(P[0,self.noelem[e,:]])
                    dydxi  = dNdxi.dot(P[1,self.noelem[e,:]])
                    dydeta = dNdeta.dot(P[1,self.noelem[e,:]])
                    # Deterinant of Jacobian 
                    detJ   = dxdxi*dydeta - dydxi*dxdeta 
                  
  
                    
                    self.phi[e]  = num_vector/(np.array([denom]).T) ##### modif
                    self.dphidx[e]   = sps.diags(dydeta/detJ).dot(dNdxi) + sps.diags(-dydxi/detJ).dot(dNdeta)
                    self.dphidy[e]   = sps.diags(-dxdeta/detJ).dot(dNdxi) + sps.diags(dxdxi/detJ).dot(dNdeta)
                    self.wdetJmes[e] = sps.diags(w_g*np.abs(detJ)*self.mes[e]/4)
                    
                     
                    
                    # Saving univariate basis functions  ### Ajout code Ali
                    if eta_min == self.xxsi[1][0]:
                        self.phi_xi[i_xi] = Nxi
                        self.dphi_xi[i_xi] = dNxidxi
                        i_xi += 1 
                    if xi_min ==  self.xxsi[0][0]:
                        self.phi_eta[i_eta] = Neta
                        self.dphi_eta[i_eta] = dNetadeta   
                        i_eta +=1 


                else:
                    if mes_xi==0:           ################# Pb p=3 dans Neumann / mars 2019
                        i_xi +=1    #################
                    if mes_eta==0:
                        i_eta +=1   #################



                    
#                    # Saving univariate basis functions  ### Ajout code Ali
#                    if eta_min == self.xxsi[1][0]:
#                        self.phi_xi[i_xi] = Nxi
#                        self.dphi_xi[i_xi] = dNxidxi
#                        i_xi += 1 
#                    if xi_min ==  self.xxsi[0][0]:
#                        self.phi_eta[i_eta] = Neta
#                        self.dphi_eta[i_eta] = dNetadeta   
#                        i_eta +=1 

                        

                        
        if isRational == False : 
            
            for e in listel_tot: 
                ne_xi  = self.tripleien[e,0]
                ni_xi  = self.ien[0][0,ne_xi]
                ne_eta = self.tripleien[e,1]
                ni_eta = self.ien[1][0,ne_eta]
                
                xi_min  = self.xxsi[0][ni_xi] ; xi_max  = self.xxsi[0][ni_xi+1]
                eta_min = self.xxsi[1][ni_eta]; eta_max = self.xxsi[1][ni_eta+1]
                # Treating elements of non zero measures only
                mes_xi = xi_max-xi_min
                mes_eta = eta_max-eta_min
                self.mes[e] = mes_xi*mes_eta
                if self.mes[e] != 0:
                    # Mapping to the knot span 
                    xi_p   = xi_min  + 0.5*(Gauss_xi[0]+1)*mes_xi
                    eta_p  = eta_min + 0.5*(Gauss_eta[0]+1)*mes_eta
                    # Basis functions and dertivatives evaluated on Gauss points 
                    Nxi  , dNxidxi    = derbasisfuns_c(self.pp[0],self.xxsi[0],xi_p,nbg_xi,ni_xi,1)
                    Neta , dNetadeta  = derbasisfuns_c(self.pp[1],self.xxsi[1],eta_p,nbg_eta,ni_eta,1)
                    
                    
                    r_Nxi       = np.kron(ones_eta,Nxi)
                    r_dNxidxi   = np.kron(ones_eta,dNxidxi)
                    r_Neta      = np.kron(Neta, ones_xi )
                    r_dNetadeta = np.kron(dNetadeta, ones_xi)
                    
                    Neta_Dot_Nxi      = np.einsum('nk,nl->nkl',r_Neta,r_Nxi).reshape(r_Neta.shape[0],-1)
                    dNdxi  = np.einsum('nk,nl->nkl',r_Neta,r_dNxidxi).reshape(r_Neta.shape[0],-1)
                    dNdeta = np.einsum('nk,nl->nkl',r_dNetadeta,r_Nxi).reshape(r_dNetadeta.shape[0],-1) 
        
                    
                    # Jacobian elements 
                    dxdxi  = dNdxi.dot(P[0,self.noelem[e,:]])
                    dxdeta = dNdeta.dot(P[0,self.noelem[e,:]])
                    dydxi  = dNdxi.dot(P[1,self.noelem[e,:]])
                    dydeta = dNdeta.dot(P[1,self.noelem[e,:]])
                    # Determinant of Jacobian 
                    detJ   = dxdxi*dydeta - dydxi*dxdeta 
                  
                    self.phi[e]  = Neta_Dot_Nxi
                    self.dphidx[e]   = sps.diags(dydeta/detJ).dot(dNdxi) + sps.diags(-dydxi/detJ).dot(dNdeta)
                    self.dphidy[e]   = sps.diags(-dxdeta/detJ).dot(dNdxi) + sps.diags(dxdxi/detJ).dot(dNdeta)
                    self.wdetJmes[e] = sps.diags(w_g*np.abs(detJ)*self.mes[e]/4)
                    

                    
                    
                    # Saving univariate basis functions  ### Ajout code Ali
                    if eta_min == self.xxsi[1][0]:
                        self.phi_xi[i_xi] = Nxi
                        self.dphi_xi[i_xi] = dNxidxi
                        i_xi += 1 
                    if xi_min ==  self.xxsi[0][0]:
                        self.phi_eta[i_eta] = Neta
                        self.dphi_eta[i_eta] = dNetadeta   
                        i_eta +=1 

                else:
                    if mes_xi==0:           ################# Pb p=3 dans Neumann / mars 2019
                        i_xi +=1    #################
                    if mes_eta==0:
                        i_eta +=1   #################
      

                
    def Stiffness(self, hooke, JCP):
        """
        Returns stiffness matrix 
        Int ( Epsilon* Sigma) = Sum_(On elements of ) Int ( B'HB)
        """
        listel_tot = self.Get_listeltot()
        nbf_elem_uv = self.Get_nenunv()
        nbf_elem = nbf_elem_uv[0]*nbf_elem_uv[1]
        
        n_elems = listel_tot[-1] + 1
        nbf_uv =  self.Get_nunv()
        nbf =  nbf_uv[0]*nbf_uv[1]
        dim = 2 
        ndof = dim*nbf 
 
        
        
        # Indices and values for the sparse stiffness matrix K  
        nnz  = (dim*nbf_elem)**2*n_elems
        indexI = np.zeros(nnz)
        indexJ = np.zeros(nnz)
        nnz_values = np.zeros(nnz)
        Ke = np.zeros((dim*nbf_elem,dim*nbf_elem)) # Elementary stiffness matrix 
        
        sp_count = 0 # Index couter for the sparse values
        print(JCP)
        
        #listel_loc=np.array([322,321,320,296,272,248,247,223,222,198,199,200,176,224,225,249,273,274,298,297,250,295,271,345,346])
        for e in JCP :
        #for e in listel_loc :            
             if self.mes[e] !=0 : 
    
                 # Dot product of derivatives of basis functions 
                         
                 dNdx_dNdx = self.dphidx[e].T.dot(self.wdetJmes[e].dot(self.dphidx[e]))
                 dNdx_dNdy = self.dphidx[e].T.dot(self.wdetJmes[e].dot(self.dphidy[e]))
                 dNdy_dNdx = self.dphidy[e].T.dot(self.wdetJmes[e].dot(self.dphidx[e]))
                 dNdy_dNdy = self.dphidy[e].T.dot(self.wdetJmes[e].dot(self.dphidy[e]))
                
                 # 4 blocs of the elementary stiffness matrix 
                
                 # Bloc 0,0  ### ligne 3 np.sqrt(2)/2 ### patch pour la convention notation Voigt
                 Ke[:nbf_elem,:nbf_elem] =  hooke[0,0]*dNdx_dNdx + \
                                           np.sqrt(2)/2*hooke[2,0]*dNdy_dNdx + \
                                           np.sqrt(2)/2*hooke[0,2]*dNdx_dNdy + \
                                           0.5*hooke[2,2]*dNdy_dNdy
                 # Bloc 0,1
                 Ke[:nbf_elem,nbf_elem:] =  hooke[0,1]*dNdx_dNdy +\
                                           np.sqrt(2)/2* hooke[2,1]*dNdy_dNdy +\
                                           np.sqrt(2)/2*hooke[0,2]*dNdx_dNdx +\
                                           0.5*hooke[2,2]*dNdy_dNdx
                 # Bloc 1,0
                 Ke[nbf_elem:,:nbf_elem] =  hooke[1,0]*dNdy_dNdx +\
                                           np.sqrt(2)/2* hooke[2,0]*dNdx_dNdx +\
                                           np.sqrt(2)/2*hooke[1,2]*dNdy_dNdy +\
                                           0.5*hooke[2,2]*dNdx_dNdy
                 # Bloc 1,1 
                 Ke[nbf_elem:,nbf_elem:] =  hooke[1,1]*dNdy_dNdy +\
                                           np.sqrt(2)/2*hooke[2,1]*dNdx_dNdy +\
                                           np.sqrt(2)/2* hooke[1,2]*dNdy_dNdx +\
                                           0.5* hooke[2,2]*dNdx_dNdx
                
                
                
#                 # Bloc 0,0 ### pour la notation choisie par Ali
#                 Ke[:nbf_elem,:nbf_elem] =  hooke[0,0]*dNdx_dNdx + \
#                                           hooke[2,0]*dNdy_dNdx + \
#                                           hooke[0,2]*dNdx_dNdy + \
#                                           hooke[2,2]*dNdy_dNdy 
#                 # Bloc 0,1
#                 Ke[:nbf_elem,nbf_elem:] =  hooke[0,1]*dNdx_dNdy +\
#                                           hooke[2,1]*dNdy_dNdy +\
#                                           hooke[0,2]*dNdx_dNdx +\
#                                           hooke[2,2]*dNdy_dNdx 
#                 # Bloc 1,0
#                 Ke[nbf_elem:,:nbf_elem] =  hooke[1,0]*dNdy_dNdx +\
#                                           hooke[2,0]*dNdx_dNdx +\
#                                           hooke[1,2]*dNdy_dNdy +\
#                                           hooke[2,2]*dNdx_dNdy 
#                 # Bloc 1,1 
#                 Ke[nbf_elem:,nbf_elem:] =  hooke[1,1]*dNdy_dNdy +\
#                                           hooke[2,1]*dNdx_dNdy +\
#                                           hooke[1,2]*dNdy_dNdx +\
#                                           hooke[2,2]*dNdx_dNdx 
                                   
 
#                 import pdb; pdb.set_trace()                  
                 rep=np.r_[self.noelem[e,:],self.noelem[e,:]+nbf]
                 [repi,repj]=np.meshgrid(rep,rep)  
                 repi = repi.ravel()
                 repj = repj.ravel()
                 indexI[sp_count+np.arange(len(repi))] = repi
                 indexJ[sp_count+np.arange(len(repj))] = repj
                 nnz_values[sp_count+np.arange(len(repj))] = Ke.ravel()        
                 sp_count+=len(repj)
                 
        # Sparse stiffness matrix 
        K  = sps.csc_matrix(( nnz_values, (indexI,indexJ)), shape = (ndof,ndof ))
        return  K 
#    def SelectMeshNodes(self,n=-1):
#        Btot = self.Get_Btot()
#        plt.plot(Btot[0,:],Btot[1,:],'ko')
#        plt.title('Select '+str(n)+' points... and press enter')
#        pts1=np.array(plt.ginput(n))
#        plt.close()
#        dx=np.kron(np.ones(pts1.shape[0]),Btot[[0],:].T) - np.kron(np.ones((Btot.shape[1],1)),pts1[:,0])
#        dy=np.kron(np.ones(pts1.shape[0]),Btot[[1],:].T) - np.kron(np.ones((Btot.shape[1],1)),pts1[:,1])
#        dl = np.sqrt(dx**2+dy**2) ### 09/01/19 : modif pour plusieurs points confondus
#        nset=np.argmin(dl,axis=0) ### 09/01/19 : modif pour plusieurs points confondus
#        nset = np.argwhere(dl.ravel()==dl.ravel()[nset]).ravel() ### 09/01/19 : modif pour plusieurs points confondus
#        plt.plot(Btot[0,:],Btot[1,:],'ko')
#        plt.plot(Btot[0,nset],Btot[1,nset],'ro')    
#        return nset,np.array([nset,nset+Btot.shape[1]])
    def SelectMeshNodes(self,n=-1):
        Btot = self.Get_Btot()
        plt.plot(Btot[0,:],Btot[1,:],'ko')
        plt.title('Select '+str(n)+' points... and press enter')
        pts1=np.array(plt.ginput(n))
        plt.close()
        dx=np.kron(np.ones(pts1.shape[0]),Btot[[0],:].T) - np.kron(np.ones((Btot.shape[1],1)),pts1[:,0])
        dy=np.kron(np.ones(pts1.shape[0]),Btot[[1],:].T) - np.kron(np.ones((Btot.shape[1],1)),pts1[:,1])
        nset=np.argmin(np.sqrt(dx**2+dy**2),axis=0)
        plt.plot(Btot[0,:],Btot[1,:],'ko')
        plt.plot(Btot[0,nset],Btot[1,nset],'ro')    
    #    return nset,np.array([nset*2,nset*2+1])
        return nset,np.array([nset,nset+Btot.shape[1]])      ### ordre inconnues changé

        
    def SelectMeshLine(self):
        Btot = self.Get_Btot()
        plt.plot(Btot[0,:],Btot[1,:],'ko')
        plt.title('Select 2 points of a line... and press enter')
        pts1=np.array(plt.ginput(2))
        plt.close()
        n1=np.argmin(np.linalg.norm(Btot[0:2,:].T-pts1[0,:],axis=1))        ### modif 0:2
        n2=np.argmin(np.linalg.norm(Btot[0:2,:].T-pts1[1,:],axis=1))        ### modif 0:2
        v=np.diff(Btot[0:2,[n1,n2]],axis=1)[:,0]        ### modif 0:2
        nv=np.linalg.norm(v)
        v=v/nv
        n=np.array([v[1],-v[0]])
        c=n.dot(Btot[0:2,n1])        ### modif 0:2
        rep,=np.where(abs(Btot[0:2,:].T.dot(n)-c)<1e-8)   ### modif 0:2 on ne prend que les 2 premières lignes de Btot (pour cas où on prend un poids (NURBS))
        c1=v.dot(Btot[0:2,n1])        ### modif 0:2
        c2=v.dot(Btot[0:2,n2])        ### modif 0:2
        nrep=Btot[0:2,rep]        ### modif 0:2
        rep2,=np.where(((nrep.T.dot(v)-c1)*(nrep.T.dot(v)-c2))<nv*1e-3)
        nset=rep[rep2]
        plt.plot(Btot[0,:],Btot[1,:],'ko')
        plt.plot(Btot[0,nset],Btot[1,nset],'ro')    
        return nset,np.array([nset,nset+Btot.shape[1]])       
    def CoorMeshNode(self,coorNode):
        '''coordNode : [xNode,yNode]
        returns the indices of the node dof that is pointed
        if no node corresponds, returns an empty vector
        For 2D or 3D problems'''
        Btot = self.Get_Btot()
#        nset = np.where(np.all(Btot[:-1,:].T==coorNode,axis=1))[0][0] # si plusieurs points de controle au meme endroit, voir code IPERm
        nset, = np.where(np.all(Btot[:-1,:].T==coorNode,axis=1)) # si plusieurs points de controle au meme endroit, voir code IPERm
#        return nset,np.array([[nset],[nset+Btot.shape[1]]])    ### matrice ???
        return nset,np.array([nset,nset+Btot.shape[1]])    ### matrice ???
    def CoorMeshLine(self,lineCoeff):
        '''lineCoeff : [a,b,c] so that ax+by+c=0 is the line equation
        returns the indices of the nodes dof that are pointed
        if no node corresponds, returns an empty vector
        Only for 2D problems'''
        Btot = self.Get_Btot()
        if lineCoeff[1]==0:
#            nset = np.where(Btot[0,:]==-lineCoeff[2]/lineCoeff[0])[0] #####tol
            nset = np.where(np.abs(Btot[0,:]+lineCoeff[2]/lineCoeff[0])<1e-8*np.max(np.abs(Btot[0,:])))[0] ### remplacement de == par <1e-8 tolerance
        else:
            XBtot = Btot[0,:]
            YBtot = Btot[1,:]
            Yline = (-lineCoeff[0]*XBtot - lineCoeff[2])/lineCoeff[1]
            nset = np.where(YBtot==Yline)[0] #####tol
#            print('')
#            print('CoorMeshLine')
#            print(nset)
            nset = np.where(np.abs(YBtot-Yline)<((1e-8)*np.max(np.abs(YBtot))))[0]  ### remplacement de == par <1e-8
#            print(nset)
#            print('')
        return nset,np.array([nset,nset+Btot.shape[1]])
    
    def SelectEdge(self,set1):
        '''input : chosen points on an edge (typically for BC)
        output : {xi},{eta},elts_integration,pts_1D : one scalar and one knot vector that defines a 1D curve, + indices of knot spans for integration
        Before applying BC on a segment, it can be necessary to use Bezier decomposition so that the segment edges have C0 continuity.'''
#        # A remplacer par un fonction Python plus rapide si ça existe
#        import pdb;pdb.set_trace() #???????????????????????????????????????????
        nu = self.Get_nunv()[0]
        nv = self.Get_nunv()[1]
        ctrlNet = self.Get_listBtot().reshape(nv,-1)
#        coorEdge = np.zeros((len(set1),2))
#        for ipt in range(len(set1)):
#            coorEdge[ipt,0]= np.where(ctrlNet==set1[ipt])[0][0]
#            coorEdge[ipt,1]=np.where(ctrlNet==set1[ipt])[1][0]
#        # Fin du paragraphe à remplacer éventuellement
        tmp = np.where(np.in1d(ctrlNet,set1,assume_unique=True).reshape(self.Get_nunv()[1],-1)) # si cette ligne marche, virer le tmp et adapter la suite
        coorEdge = np.c_[tmp[0],tmp[1]]
        if all(coorEdge[:,0]==coorEdge[0,0]): # si tous les points sur la même ligne de ctrlNet = à eta constant
            if coorEdge[0,0]!=0 and coorEdge[0,0]!=(nv-1):
                print("Eta-constant line, but not an edge") ### plutôt mettre une erreur
            pts1D = set1 % nu # reste % de la division euclidienne par nu pour avoir les points 1D selon la courbe de knot vector xi
            # on prend comme elts d'intégration ceux qui contiennent toutes les fonctions choisies uniquement
            elts_integration = np.where(np.all(np.logical_and(self.ien[0]>=min(pts1D),self.ien[0] <= max(pts1D)),axis=0)) ### A MODIF : elts de taille nulle !
            return self.xxsi[0],np.array([coorEdge[0,0]/(nv-1)]),elts_integration[0] # 0 car elts=(array([...],dtype),)
        elif all(coorEdge[:,1]==coorEdge[0,1]): # si tous les points sur la même colonne de ctrlNet = à xi constant
            if coorEdge[0,1]!=0 and coorEdge[0,1]!=(nu-1):
                print("Xi-constant line, but not an edge") ### plutôt mettre une erreur
            pts1D = set1 // nu # quotient // de la division euclidienne par nu pour avoir les points 1D selon la courbe de knot vector eta
            # on prend comme elts d'intégration ceux qui contiennent toutes les fonctions choisies uniquement
            elts_integration = np.where(np.all(np.logical_and(self.ien[1]>=min(pts1D),self.ien[1] <= max(pts1D)),axis=0)) ### A MODIF : elts de taille nulle !
            return np.array([coorEdge[0,1]/(nu-1)]),self.xxsi[1],elts_integration[0] # 0 car elts=(array([...],dtype),)
        else:
            print("Not a xi-constant nor an eta-constant line") ###Erreur à afficher
            return [],[],[]
    def Neumann(self,rep1,f0):
        '''input : dof corresponding to chosen points on an edge that are on a segment (array[[2,3,4][2+n,3+n,4+n]] for segment [pt2,pt4]), and f0=[fx0,fy0] load
        output : F vector, right-hand side for KU=F
        Computes the load vector for uniformly distributed load f0'''
        set1 = rep1[0]
        xi,eta,elts_int = self.SelectEdge(set1)
#        print(xi,eta,elts_int)
        
        if len(xi)==1 and len(eta)==1:
            print("Can't apply uniformly distributed load on a point") ###Erreur à afficher
            
        F = np.zeros(self.Get_ndof()) ### ou initialiser de la taille de 2*set1 pour ne pas remplir tout F ? (et éventuellement pouvoir imposer des CL de Dirichlet avant Neumann)

        if len(eta)==1: 
            # On travaille maintenant sur une courbe 1D de knot vector xi
            p = self.pp[0]
            xG,wG  =  nb.GaussLegendre(p+1)
            e=0 # compteur
            for elt in elts_int: # boucle sur les élts d'intégration
                xi_min = xi[elt+p]
                xi_max = xi[elt+p+1]
                if abs(xi_max-xi_min)!=0: #si elt de taille non nulle
                    Fe = np.zeros(2*(p+1)) # longueur = nb de fonctions non nulles sur l'élt
                    N  = self.phi_xi[elt]
                    dN = self.dphi_xi[elt]
                    for pg in range(p+1): # boucle sur les points de Gauss
                        wg = wG[pg]
#                        print(dN[pg].shape,self.Get_Btot()[0,set1[e:(e+p+1)]].shape)
#                        print('set1')
#                        print(set1,set1[e:(e+p+1)])
                        Jx = np.sum(dN[pg]*self.Get_Btot()[0,set1[e:(e+p+1)]])
                        Jy = np.sum(dN[pg]*self.Get_Btot()[1,set1[e:(e+p+1)]])
                        Js = (Jx**2+Jy**2)**0.5 ### ?
                        Nboundary_matrix = np.kron(np.eye(2),N[pg])
                        Fe += Js*wg*np.dot(np.transpose(Nboundary_matrix),f0)*abs(xi_max-xi_min)/2. ###?
                F[rep1[:,e:(e+p+1)].ravel()]+=Fe                
                e+=1

        if len(xi)==1:
            # On travaille maintenant sur une courbe 1D de knot vector eta
            p = self.pp[1]
            xG,wG  =  nb.GaussLegendre(p+1)
            e=0 # compteur
            for elt in elts_int: # boucle sur les élts d'intégration
                eta_min = eta[elt+p]
                eta_max = eta[elt+p+1]
                if abs(eta_max-eta_min)!=0: #si elt de taille non nulle
                    Fe = np.zeros(2*(p+1)) # longueur = nb de fonctions non nulles sur l'élt
                    N  = self.phi_eta[elt]
                    dN = self.dphi_eta[elt]
                    for pg in range(p+1): # boucle sur les points de Gauss
                        wg = wG[pg]
                        Jx = np.sum(dN[pg]*self.Get_Btot()[0,set1[e:(e+p+1)]])
                        Jy = np.sum(dN[pg]*self.Get_Btot()[1,set1[e:(e+p+1)]])
                        Js = (Jx**2+Jy**2)**0.5 ### ?
                        Nboundary_matrix = np.kron(np.eye(2),N[pg])
                        Fe += Js*wg*np.dot(np.transpose(Nboundary_matrix),f0)*abs(eta_max-eta_min)/2. ### ? 
                F[rep1[:,e:(e+p+1)].ravel()]+=Fe                
                e+=1
                
        return F
    
    def Neumann_PlateHole(self,rep1,vec_nor): ##############################EN COURS _ utile pour test poutre infinie plaque trouée
        '''input : dof corresponding to chosen points on an edge that are on a segment (array[[2,3,4][2+n,3+n,4+n]] for segment [pt2,pt4]), and f0=[fx0,fy0] load
        vec_nor : normale sortante (vecteur de norme 1)
        output : F vector, right-hand side for KU=F
        Computes the load vector for infinite beam with hole (for plate with hole)'''
        set1 = rep1[0]
        xi,eta,elts_int = self.SelectEdge(set1)
#        print(xi,eta,elts_int)
        
        if len(xi)==1 and len(eta)==1:
            print("Can't apply distributed load on a point") ###Erreur à afficher
            
        F = np.zeros(self.Get_ndof()) ### ou initialiser de la taille de 2*set1 pour ne pas remplir tout F ? (et éventuellement pouvoir imposer des CL de Dirichlet avant Neumann)
        nu,nv = self.Get_nunv()
        

        if len(eta)==1: 
            # On travaille maintenant sur une courbe 1D de knot vector xi
            p = self.pp[0]
            xG,wG  =  nb.GaussLegendre(p+1)
            e=0 # compteur
            for elt in elts_int: # boucle sur les élts d'intégration
                xi_min = xi[elt+p]
                xi_max = xi[elt+p+1]
                if abs(xi_max-xi_min)!=0: #si elt de taille non nulle
                    Fe = np.zeros(2*(p+1)) # longueur = nb de fonctions non nulles sur l'élt
                    N  = self.phi_xi[elt]
                    dN = self.dphi_xi[elt]
                    for pg in range(p+1): # boucle sur les points de Gauss
                        ### Déterminer les coordonnées des pg dans le domaine physique
                        # self.ien[0][:,elt] donne les numéros des fonctions associées
                        pts = int((nv-1)*nu*eta) + np.flip(self.ien[0][:,elt],axis=0)
                        # Trouver les coord des points de numéros trouvés juste avant (grâce à eta)
                        pts = self.Get_Btot()[:-1,pts].T
                        # N[pg] contient les valeurs des fonctions au pg
                        Xg = N[pg].dot(pts)
                        ### Déterminer f = (s_xx, s_xy) ou (s_xy, s_yy) aux pg
                        s_xx,s_yy,s_xy = EvaluateExactStress(Xg[0],Xg[1])
                        sigma2D = np.array([[s_xx,s_xy],[s_xy,s_yy]])
                        f0 = sigma2D.dot(vec_nor)
                        wg = wG[pg]
                        Jx = np.sum(dN[pg]*self.Get_Btot()[0,set1[e:(e+p+1)]])
                        Jy = np.sum(dN[pg]*self.Get_Btot()[1,set1[e:(e+p+1)]])
                        Js = (Jx**2+Jy**2)**0.5 ### ?
                        Nboundary_matrix = np.kron(np.eye(2),N[pg])
                        Fe += Js*wg*np.dot(np.transpose(Nboundary_matrix),f0)*abs(xi_max-xi_min)/2. ###?
                F[rep1[:,e:(e+p+1)].ravel()]+=Fe                
                e+=1

        if len(xi)==1:
            # On travaille maintenant sur une courbe 1D de knot vector eta
            p = self.pp[1]
            xG,wG  =  nb.GaussLegendre(p+1)
            e=0 # compteur
            for elt in elts_int: # boucle sur les élts d'intégration
                eta_min = eta[elt+p]
                eta_max = eta[elt+p+1]
                if abs(eta_max-eta_min)!=0: #si elt de taille non nulle
                    Fe = np.zeros(2*(p+1)) # longueur = nb de fonctions non nulles sur l'élt
                    N  = self.phi_eta[elt]
                    dN = self.dphi_eta[elt]
                    for pg in range(p+1): # boucle sur les points de Gauss
                        ### Déterminer les coordonnées des pg dans le domaine physique
                        # self.ien[0][:,elt] donne les numéros des fonctions associées
                        pts = nu*np.flip(self.ien[1][:,elt],axis=0) +int((nv-1)*xi)
                        # Trouver les coord des points de numéros trouvés juste avant (grâce à eta)
                        pts = self.Get_Btot()[:-1,pts].T
                        # N[pg] contient les valeurs des fonctions au pg
                        Xg = N[pg].dot(pts)
                        ### Déterminer f = (s_xx, s_xy) ou (s_xy, s_yy) aux pg
                        s_xx,s_yy,s_xy = EvaluateExactStress(Xg[0],Xg[1])
                        sigma2D = np.array([[s_xx,s_xy],[s_xy,s_yy]])
                        f0 = sigma2D.dot(vec_nor)
                        wg = wG[pg]
                        Jx = np.sum(dN[pg]*self.Get_Btot()[0,set1[e:(e+p+1)]])
                        Jy = np.sum(dN[pg]*self.Get_Btot()[1,set1[e:(e+p+1)]])
                        Js = (Jx**2+Jy**2)**0.5 ### ?
                        Nboundary_matrix = np.kron(np.eye(2),N[pg])
                        Fe += Js*wg*np.dot(np.transpose(Nboundary_matrix),f0)*abs(eta_max-eta_min)/2. ### ? 
                F[rep1[:,e:(e+p+1)].ravel()]+=Fe                
                e+=1
                
        return F
    
    
    def Surface(self):
        """
        Returns geometry surface
        Int dS
        """
        if self.mes==0:
            self.GaussIntegration() ### si on sort le calcul de mes de GaussIntegration, à modifier ici
        elif len(self.mes) != (self.Get_listeltot().size):
            self.GaussIntegration() ### si on sort le calcul de mes de GaussIntegration, à modifier ici
        listel_tot = self.Get_listeltot()        
        S = 0
#        import pdb; pdb.set_trace()
        for e in listel_tot :
             if self.mes[e] !=0 : 
                 Se = np.sum(self.wdetJmes[e])
             S += Se
        return  S
 
    def Draw(self,Color='',LineType='-',Legend='',ctrlpts=True,cpMarker='o',cpColor='k',cpMarkersize=4.,elem=False,data=False,plot=True):
        """Draws the edges of the geometrical domain"""
        nu,nv = self.Get_nunv() # peut poser pb si Get_nunv renvoie un array au lieu d'un tuple
        B = self.Get_Btot()
        NURBS = self.IsRational()
        neval = 50
        XSI = self.xxsi[0]
        ETA = self.xxsi[1]
        # Bord eta=0
        X1 = nb.drawline(XSI,self.pp[0],B[:,:nu],neval,NURBS)
        # Bord eta=1
        X2 = nb.drawline(XSI,self.pp[0],B[:,-nu:],neval,NURBS)
        # bord xsi=0
        xi0 = nu*np.arange(nv)
        X3 = nb.drawline(ETA,self.pp[1],B[:,xi0],neval,NURBS)
        # bord xsi=1
        xi1 = xi0+nu-1
        X4 = nb.drawline(ETA,self.pp[1],B[:,xi1],neval,NURBS)
        # affichage
        if plot:
            if len(Color)<2 and len(cpColor)<2:
                #plt.figure()
                plt.plot(X1[0],X1[1],Color+LineType,label=Legend)
                plt.plot(X2[0],X2[1],Color+LineType)
                plt.plot(X3[0],X3[1],Color+LineType)
                plt.plot(X4[0],X4[1],Color+LineType)
                if ctrlpts:
                    plt.plot(self.Get_Btot()[0],self.Get_Btot()[1],cpMarker,color=cpColor,markersize=cpMarkersize)
            else:
                #plt.figure()
                plt.plot(X1[0],X1[1],LineType,color=Color,label=Legend)
                plt.plot(X2[0],X2[1],LineType,color=Color)
                plt.plot(X3[0],X3[1],LineType,color=Color)
                plt.plot(X4[0],X4[1],LineType,color=Color)
                if ctrlpts:
                    plt.plot(self.Get_Btot()[0],self.Get_Btot()[1],cpMarker,color=cpColor,markersize=cpMarkersize)
                
            
    #        if elem:
    #            XSI = 
    #            for eta in np.unique(self.xxsi[1])[1:-1]: #traits à eta cst, sauf bords, et on évite les elts nuls
    #                xi_eval = np.linspace(XSI[0],XSI[-1],neval)
    #                x_eval = np.zeros((self.Get_Btot().shape[0]-1,neval))
    #                for x in range(neval):
    #                    XiSpan =nb.findKnotSpan(xi_eval[x],XSI,self.pp[0])
    #            print("Affichage des elts à coder")
                
            plt.axis('equal')
        
        if data:
            X = np.c_[X1,X4,np.fliplr(X2),np.fliplr(X3)]
            return X

    def VTKPlot(self,tol,D,neval,U,filename,timestep=0):
        '''input : D, IEN(x,y),xxsi(x,y)...
        U déplacement, neval nb eval points
        '''
        nfune=self.Get_nenunv()[0]*self.Get_nenunv()[1]   # nb de fonctions non nulles sur un elt (ie nb de pts de contrôle associés)
        nddle=nfune*2    # nb de ddl
        [xpt,ypt]=np.meshgrid(np.linspace(0,1,neval),np.linspace(0,1,neval)) # grille de nds d'évaluation dans l'espace isoparamétrique
        xp=xpt.ravel().copy() # mise sous forme de vecteurs pour que l'indice corresponde au numero du nd d'évaluation
        yp=ypt.ravel().copy()
        xpt[:,0]+=tol # tolerance ajoutée pour traiter les cas où on évalue aux nds du bord du domaine isoparam
        xpt[:,-1]-=tol
        ypt[0,:]+=tol
        ypt[-1,:]-=tol
        xpt=xpt.ravel()
        ypt=ypt.ravel()    
        Up=np.zeros(len(xp)*3) # déplacement X,Y,Z (ici Z=0)
        Sigp=np.zeros(len(xp)*3) # contrainte X,Y,XY
        Sigex=np.zeros(len(xp)*3) ##### A COMMENTER : sol exacte plaque
        Epsp=np.zeros(len(xp)*3) # deformation X,Y,XY
        N=np.zeros((len(xp),3)) ###?
        for ip in range(len(xp)): # boucle sur les nds d'évaluation
            xsipg=xp[ip]
            etapg=yp[ip]
            nix=np.where(self.xxsi[0]<xpt[ip])[0][-1] ###
            niy=np.where(self.xxsi[1]<ypt[ip])[0][-1] ###
            DMx = nb.derbasisfuns(nix,self.pp[0],self.xxsi[0],1,xsipg)
            DMy = nb.derbasisfuns(niy,self.pp[1],self.xxsi[1],1,etapg)        
            Mx = DMx[0,:]
            My = DMy[0,:]
            dMx = DMx[1,:]
            dMy = DMy[1,:]
            
            # fonctions B-spline et derivees 2D    
            #-------------------------------------
            
            Mxy2D = np.zeros(nfune)
            Mxy2Ddeta = np.zeros(nfune)
            Mxy2Ddxsi = np.zeros(nfune)
            for ii in range(self.Get_nenunv()[1]):  #1:neny
                for jj in range(self.Get_nenunv()[0]):  #1:nenx
                    Mxy2D[jj+ii*self.Get_nenunv()[0]] = My[ii]*Mx[jj]
                    Mxy2Ddeta[jj+ii*self.Get_nenunv()[0]] = dMy[ii]*Mx[jj]
                    Mxy2Ddxsi[jj+ii*self.Get_nenunv()[0]] = My[ii]*dMx[jj]
            
            nex=np.where(self.ien[0][0,:]==nix)[0][-1]
            ney=np.where(self.ien[1][0,:]==niy)[0][-1]
            ne=nex+ney*self.ien[0].shape[1] ###
            
            # fonctions NURBS et derivees 2D
            #------------------------
            # denominateur
            denom = 0
            denomdeta = 0
            denomdxsi = 0
            for pp in range(nfune):  #1:nfune
               denom = denom + Mxy2D[pp]*self.Get_Btot()[-1,self.noelem[ne,pp]]
               denomdeta = denomdeta + Mxy2Ddeta[pp]*self.Get_Btot()[-1,self.noelem[ne,pp]]
               denomdxsi = denomdxsi + Mxy2Ddxsi[pp]*self.Get_Btot()[-1,self.noelem[ne,pp]]
            # fonctions et derivées
            Nxy2D = np.zeros(nfune)
            Nxy2Ddeta = np.zeros(nfune)
            Nxy2Ddxsi = np.zeros(nfune)
            for pp in range(nfune):  #1:nfune
               Nxy2D[pp] = (Mxy2D[pp]*self.Get_Btot()[-1,self.noelem[ne,pp]])/denom
               Nxy2Ddeta[pp] = (((Mxy2Ddeta[pp]*self.Get_Btot()[-1,self.noelem[ne,pp]]))*denom-(Mxy2D[pp]*self.Get_Btot()[-1,self.noelem[ne,pp]])*denomdeta)/(denom**2)
               Nxy2Ddxsi[pp] = (((Mxy2Ddxsi[pp]*self.Get_Btot()[-1,self.noelem[ne,pp]]))*denom-(Mxy2D[pp]*self.Get_Btot()[-1,self.noelem[ne,pp]])*denomdxsi)/(denom**2)
            # jacobien
            dxdeta = 0
            dxdxsi = 0
            dydeta = 0
            dydxsi = 0
            for pp in range(nfune):  #1:nfune
                dxdeta = dxdeta + Nxy2Ddeta[pp]*self.Get_Btot()[0,self.noelem[ne,pp]]
                dxdxsi = dxdxsi + Nxy2Ddxsi[pp]*self.Get_Btot()[0,self.noelem[ne,pp]]
                dydeta = dydeta + Nxy2Ddeta[pp]*self.Get_Btot()[1,self.noelem[ne,pp]]
                dydxsi = dydxsi + Nxy2Ddxsi[pp]*self.Get_Btot()[1,self.noelem[ne,pp]]
            Jxsi = dxdxsi*dydeta-dydxsi*dxdeta
            # jacobien inverse
            dxsidx = 1/Jxsi * dydeta
            dxsidy = - 1/Jxsi * dxdeta
            detadx = - 1/Jxsi * dydxsi
            detady = 1/Jxsi * dxdxsi
            
            # calcul des deformations
            BB = np.zeros((3,nddle)) 
            for pp in range(nfune):  #1:nfune   ### ordre inconnues changé
                  BB[0,pp] = Nxy2Ddxsi[pp]*dxsidx + Nxy2Ddeta[pp]*detadx                      #dN2Ddx
                  BB[1,pp+nfune] = Nxy2Ddxsi[pp]*dxsidy + Nxy2Ddeta[pp]*detady                    #dN2Ddy
                  BB[2,pp] = 1/np.sqrt(2)*(Nxy2Ddxsi[pp]*dxsidy + Nxy2Ddeta[pp]*detady)       #1/sqrt(2)*dN2Ddy
                  BB[2,pp+nfune] = 1/np.sqrt(2)*(Nxy2Ddxsi[pp]*dxsidx + Nxy2Ddeta[pp]*detadx)     #1/sqrt(2)*dN2Ddx
                    

            NN = np.zeros((2,nddle))      
            NN[0,np.arange(nfune)] = Nxy2D
            NN[1,np.arange(nfune)+nfune] = Nxy2D      
            Uex=U[(self.noelem[ne,np.arange(nfune)])]  ### modif inconnues       
            Uey=U[(self.noelem[ne,np.arange(nfune)])+self.Get_nunv()[0]*self.Get_nunv()[1]]
            Ue=np.r_[Uex,Uey]
            Epsp[3*ip:3*ip+3]=BB.dot(Ue)
            Epsp[3*ip+2] = Epsp[3*ip+2]/np.sqrt(2)  # dû à la convention, pour avoir juste eps_xy
            Sigp[3*ip:3*ip+3]=D.dot(BB.dot(Ue))
            Sigp[3*ip+2] = Sigp[3*ip+2]/np.sqrt(2)  # dû à la convention, pour avoir juste sig_xy
            Up[3*ip:3*ip+2]=NN.dot(Ue)
            Bex=self.Get_Btot()[0,self.noelem[ne,np.arange(nfune)]]        
            Bey=self.Get_Btot()[1,self.noelem[ne,np.arange(nfune)]]
            Be=np.r_[Bex,Bey]
            N[ip,:2]=NN.dot(Be)
        e=dict()
        NE=np.array([neval-1,neval-1]) # ne pas enlever les -1
        for ix in range(NE[0]):
            for iy in range(NE[1]):
                p1=ix*(NE[1]+1)+iy
                p4=ix*(NE[1]+1)+iy+1
                p2=ix*(NE[1]+1)+iy+NE[1]+1
                p3=ix*(NE[1]+1)+iy+NE[1]+2
                e[ix*NE[1]+iy]=np.array([3,p1,p2,p3,p4])
        # Export  VTK
        nnode=N.shape[0]
        new_node=N.ravel()
        new_u=Up
        new_conn=np.array([], dtype='int')
        new_offs=np.array([], dtype='int')
        new_type=np.array([], dtype='int')
        nelem=len(e)
        coffs=0
        for je in range(nelem):
            coffs=coffs+4
            new_type=np.append(new_type,9)
            new_conn=np.append(new_conn,e[je][1:])
            new_offs=np.append(new_offs,coffs)
        vtkfile=vtk.VTUWriter(nnode,nelem,new_node,new_conn,new_offs,new_type)
        s_xx,s_yy,s_xy = EvaluateExactStress(new_node[3*np.arange(nnode)],new_node[3*np.arange(nnode)+1]) ##### A COMMENTER : sol exacte plaque
        Sigex = np.c_[s_xx,s_yy,s_xy].ravel() ##### A COMMENTER : sol exacte plaque
        vtkfile.addPointData('displ',3,new_u)
        # Strain
        vtkfile.addPointData('strain',3,Epsp)
        # Stress
        vtkfile.addPointData('stress',3,Sigp)
        # Exact Stress ##### A COMMENTER : sol exacte plaque
        vtkfile.addPointData('strex',3,Sigex) ##### A COMMENTER : sol exacte plaque
        # Write the VTU file in the VTK dir
        rep=filename.rfind('/')+1
        if rep==0:
            dir0=''
        else:
            dir0=filename[:rep]
        if not os.path.isdir('vtk/'+dir0):
            os.mkdir('vtk/'+dir0)
        vtkfile.write('vtk/'+filename+'_0_'+str(timestep))
        
    def VTKMesh(self,tol,D,neval,U,filename,timestep=0):
        nfune=self.Get_nenunv()[0]*self.Get_nenunv()[1]
        nddle=nfune*2
        NE=np.array([neval-1,len(np.unique(self.xxsi[0]))])
        [ypt,xpt]=np.meshgrid(np.linspace(0,1,neval),np.unique(self.xxsi[0]))
        xp=xpt.ravel().copy()
        yp=ypt.ravel().copy()
        xpt[0,:]+=tol
        xpt[-1,:]-=tol
        ypt[:,0]+=tol
        ypt[:,-1]-=tol
        xpt=xpt.ravel()
        ypt=ypt.ravel()    
        Up=np.zeros(len(xp)*3)
        N=np.zeros((len(xp),3))
        for ip in range(len(xp)):
            xsipg=xp[ip]
            etapg=yp[ip]
            nix=np.where(self.xxsi[0]<xpt[ip])[0][-1] ###+1
            niy=np.where(self.xxsi[1]<ypt[ip])[0][-1] ###+1
            Mx = nb.derbasisfuns(nix,self.pp[0],self.xxsi[0],0,xsipg)[0]
            My = nb.derbasisfuns(niy,self.pp[1],self.xxsi[1],0,etapg)[0]      
            #Mx = DMx[0,:]
            
            # fonctions B-spline et derivees 2D
            #-------------------------------------
            
            Mxy2D = np.zeros(nfune)
            for ii in range(self.Get_nenunv()[1]):  #1:neny
                for jj in range(self.Get_nenunv()[0]):  #1:nenx
                    Mxy2D[jj+ii*self.Get_nenunv()[0]] = My[ii]*Mx[jj]
            
            nex=np.where(self.ien[0][0,:]==nix)[0][-1]
            ney=np.where(self.ien[1][0,:]==niy)[0][-1]
            ne=nex+ney*self.ien[0].shape[1] ###+1
            
            # fonctions NURBS et derivees 2D
            #------------------------
            # denominateur
            denom = 0
            for pp in range(nfune):  #1:nfune
               denom = denom + Mxy2D[pp]*self.Get_Btot()[-1,self.noelem[ne,pp]]
            # fonctions et derivées
            Nxy2D = np.zeros(nfune)
            for pp in range(nfune):  #1:nfune
                Nxy2D[pp] = (Mxy2D[pp]*self.Get_Btot()[-1,self.noelem[ne,pp]])/denom                
                  
            NN = np.zeros((2,nddle))
            NN[0,np.arange(nfune)] = Nxy2D
            NN[1,np.arange(nfune)+nfune] = Nxy2D           
            
            Uex=U[(self.noelem[ne,np.arange(nfune)])]  ### modif inconnues       
            Uey=U[(self.noelem[ne,np.arange(nfune)])+self.Get_nunv()[0]*self.Get_nunv()[1]]
            Ue=np.r_[Uex,Uey]
            Up[3*ip:3*ip+2]=NN.dot(Ue)
            Bex=self.Get_Btot()[0,self.noelem[ne,np.arange(nfune)]]  ###       
            Bey=self.Get_Btot()[1,self.noelem[ne,np.arange(nfune)]]
            Be=np.r_[Bex,Bey]
            N[ip,:2]=NN.dot(Be)
        e=dict()
        for ix in range(NE[1]):
            for iy in range(NE[0]):
                p1=ix*(NE[0]+1)+iy
                p2=ix*(NE[0]+1)+iy+1
                e[ix*NE[0]+iy]=np.array([p1,p2])
        # Export  VTK
        nnode=N.shape[0]
        new_node=N.ravel()
        new_u=Up
        new_conn=np.array([], dtype='int')
        new_offs=np.array([], dtype='int')
        new_type=np.array([], dtype='int')
        nelem=len(e)
        coffs=0
        for je in range(nelem):
            coffs=coffs+2
            new_type=np.append(new_type,3)
            new_conn=np.append(new_conn,e[je])
            new_offs=np.append(new_offs,coffs)
        vtkfile=vtk.VTUWriter(nnode,nelem,new_node,new_conn,new_offs,new_type)
        vtkfile.addPointData('displ',3,new_u)
        # Write the VTU file in the VTK dir
        rep=filename.rfind('/')+1
        if rep==0:
            dir0=''
        else:
            dir0=filename[:rep]
        if not os.path.isdir('vtk/'+dir0):
            os.mkdir('vtk/'+dir0)
        vtkfile.write('vtk/'+filename+'_0_'+str(timestep))
        
        NE=np.array([neval-1,len(np.unique(self.xxsi[1]))])
        [xpt,ypt]=np.meshgrid(np.linspace(0,1,neval),np.unique(self.xxsi[1]))
        xp=xpt.ravel().copy()
        yp=ypt.ravel().copy()
        xpt[:,0]+=tol
        xpt[:,-1]-=tol
        ypt[0,:]+=tol
        ypt[-1,:]-=tol
        xpt=xpt.ravel()
        ypt=ypt.ravel()    
        Up=np.zeros(len(xp)*3)
        N=np.zeros((len(xp),3))
        for ip in range(len(xp)):
            xsipg=xp[ip]
            etapg=yp[ip]
            nix=np.where(self.xxsi[0]<xpt[ip])[0][-1] ###+1
            niy=np.where(self.xxsi[1]<ypt[ip])[0][-1] ###+1
            Mx = nb.derbasisfuns(nix,self.pp[0],self.xxsi[0],0,xsipg)[0]
            My = nb.derbasisfuns(niy,self.pp[1],self.xxsi[1],0,etapg)[0]      
            #Mx = DMx[0,:]
    
            
            # fonctions B-spline et derivees 2D
            #-------------------------------------
            
            Mxy2D = np.zeros(nfune)
            for ii in range(self.Get_nenunv()[1]):  #1:neny
                for jj in range(self.Get_nenunv()[0]):  #1:nenx
                    Mxy2D[jj+ii*self.Get_nenunv()[0]] = My[ii]*Mx[jj]
            
            nex=np.where(self.ien[0][0,:]==nix)[0][-1]
            ney=np.where(self.ien[1][0,:]==niy)[0][-1]
            ne=nex+ney*self.ien[0].shape[1] ###+1
            
            # fonctions NURBS et derivees 2D
            #------------------------
            # denominateur
            denom = 0
            for pp in range(nfune):  #1:nfune
               denom = denom + Mxy2D[pp]*self.Get_Btot()[-1,self.noelem[ne,pp]]
            # fonctions et derivées
            Nxy2D = np.zeros(nfune)
            for pp in range(nfune):  #1:nfune
                Nxy2D[pp] = (Mxy2D[pp]*self.Get_Btot()[-1,self.noelem[ne,pp]])/denom                
                  
            NN = np.zeros((2,nddle))
            NN[0,np.arange(nfune)] = Nxy2D
            NN[1,np.arange(nfune)+nfune] = Nxy2D           

            Uex=U[(self.noelem[ne,np.arange(nfune)])]  ### modif inconnues       
            Uey=U[(self.noelem[ne,np.arange(nfune)])+self.Get_nunv()[0]*self.Get_nunv()[1]]
            Ue=np.r_[Uex,Uey]
            Up[3*ip:3*ip+2]=NN.dot(Ue)
            Bex=self.Get_Btot()[0,self.noelem[ne,np.arange(nfune)]]  ###       
            Bey=self.Get_Btot()[1,self.noelem[ne,np.arange(nfune)]]
            Be=np.r_[Bex,Bey]
            N[ip,:2]=NN.dot(Be)
        e=dict()
        
        for ix in range(NE[1]):   ###?NE ???
            for iy in range(NE[0]):
                p1=ix*(NE[0]+1)+iy
                p2=ix*(NE[0]+1)+iy+1
                e[ix*NE[0]+iy]=np.array([p1,p2])
        # Export  VTK
    
        nnode=N.shape[0]
        new_node=N.ravel()
        new_u=Up
        new_conn=np.array([], dtype='int')
        new_offs=np.array([], dtype='int')
        new_type=np.array([], dtype='int')
        nelem=len(e)
        coffs=0
        for je in range(nelem):
            coffs=coffs+2
            new_type=np.append(new_type,3)
            new_conn=np.append(new_conn,e[je])
            new_offs=np.append(new_offs,coffs)
        vtkfile=vtk.VTUWriter(nnode,nelem,new_node,new_conn,new_offs,new_type)
        vtkfile.addPointData('displ',3,new_u)
        # Write the VTU file in the VTK dir
        rep=filename.rfind('/')+1
        if rep==0:
            dir0=''
        else:
            dir0=filename[:rep]
        if not os.path.isdir('vtk/'+dir0):
            os.mkdir('vtk/'+dir0)
        vtkfile.write('vtk/'+filename+'_1_'+str(timestep))
    
    def VTKFull(self,tol,D,neval,U,filename):
        self.VTKPlot(tol,D,neval,U,filename)
        filename=filename+'msh'
        self.VTKMesh(tol,D,neval,U,filename)
        PVDFile(filename,2,1)
    
    def Epsilon(self,U,nbg=0):
        """Déformation aux points de Gauss (notation Voigt avec sqrt(2)*eps_xy)
        input : déplacement U aux points de contrôle"""
        self.GaussIntegration(nbg)
        eps = dict()
        for elt in self.Get_listeltot():
            if self.mes[elt]!=0: ####################
                zeros = np.zeros_like(self.dphidx[elt])
                Be = np.c_[np.r_[self.dphidx[elt],zeros,np.sqrt(2)/2*self.dphidy[elt]],np.r_[zeros,self.dphidy[elt],np.sqrt(2)/2*self.dphidx[elt]]]
                Ue = np.r_[U[self.noelem[elt]],U[self.noelem[elt]+self.Get_ndof()//2]]
                eps[elt] = np.reshape(Be.dot(Ue),(3,-1)).T            
        return eps
            
        
    def Erreur_Infinite_Beam(self,U,D,nbg=9):
        """Calcule l'erreur en RdC pour le cas de la plaque trouée avec chargement de poutre infinie
        où on connaît le résultat analytique sigma
        U chp de déplacement
        D Hooke"""
#        plt.figure() ###plot
        eps = self.Epsilon(U,nbg)
        E = 0
        normalisation = 0
        for elt in self.Get_listeltot():   # Boucle sur les élts
            if self.mes[elt]!=0: ######################
            ### Calcul du champ de déformation epsilon aux pg e
            ### Calcul de sigma exact s_ex aux pts de Gauss
                # Calcul de sigma calculé aux points de Gauss
            ### Déterminer les coordonnées des pg dans le domaine physique          
                pts = self.noelem[elt]   # self.ien[0][:,elt] donne les numéros des fonctions associées            
                pts = self.Get_Btot()[:-1,pts].T   # Trouver les coord des points de numéros trouvés juste avant (grâce à eta)            
                N = self.phi[elt]   # N[pg] contient les valeurs des fonctions au pg
                Xg = N.dot(pts)
                ### Déterminer f = (s_xx, s_xy) ou (s_xy, s_yy) aux pg
                s_xx,s_yy,s_xy = EvaluateExactStress(Xg[:,0],Xg[:,1])
    #            plt.plot(Xg[:,0],Xg[:,1],'x')   ###plot
                s_ex = np.c_[s_xx,s_yy,np.sqrt(2)*s_xy]
            ### Calcul de la déformation associée e_ex = D\s_ex
                e_ex = (np.linalg.solve(D,s_ex.T)).T
            ### intégration numérique (e-e_ex)D(e-e_ex)
                err_e = eps[elt]-e_ex
                integrande = np.einsum('ik,kj,ij->i',err_e,D,err_e) # = np.diag(err_e.dot(D).dot(err_e.T))
                int_norm = np.einsum('ik,kj,ij->i',s_ex,np.linalg.inv(D),s_ex)   ###sD_1s
                E += np.sum(integrande*self.wdetJmes[elt].diagonal())
                normalisation += np.sum(int_norm*self.wdetJmes[elt].diagonal())
        return E/normalisation

    def Erreur_Curved_Beam(self,U,a,b,U0,D,E,nbg=9):
        """Calcule l'erreur en RdC pour le cas de la plaque trouée avec chargement de poutre infinie
        où on connaît le résultat analytique sigma
        U chp de déplacement
        D Hooke"""
#        plt.figure() ###plot
#        import pdb; pdb.set_trace()
        eps = self.Epsilon(U,nbg)
#        print(eps)
        Err = 0
        normalisation = 0
        for elt in self.Get_listeltot():   # Boucle sur les élts
            if self.mes[elt]!=0: ######################
            ### Calcul du champ de déformation epsilon aux pg e
            ### Calcul de sigma exact s_ex aux pts de Gauss
                # Calcul de sigma calculé aux points de Gauss
            ### Déterminer les coordonnées des pg dans le domaine physique          
                pts = self.noelem[elt]   # self.ien[0][:,elt] donne les numéros des fonctions associées            
                pts = self.Get_Btot()[:-1,pts].T   # Trouver les coord des points de numéros trouvés juste avant (grâce à eta)            
                N = self.phi[elt]   # N[pg] contient les valeurs des fonctions au pg
#                print(N)
                Xg = N.dot(pts)
#                plt.plot(Xg[:,0],Xg[:,1],'*')
                ### Déterminer f = (s_xx, s_xy) ou (s_xy, s_yy) aux pg
                e_ex = ExactSolCurvedBeam(Xg[:,0],Xg[:,1],a,b,E,U0,D)
                e_ex = e_ex.T
#                print(e_ex)
            ### intégration numérique (e-e_ex)D(e-e_ex)
                err_e = eps[elt]-e_ex
#                print(err_e)
                integrande = np.einsum('ik,kj,ij->i',err_e,D,err_e) # = np.diag(err_e.dot(D).dot(err_e.T))
                int_norm = np.einsum('ik,kj,ij->i',e_ex,D,e_ex)   ###sD_1s
                Err += np.sum(integrande*self.wdetJmes[elt].diagonal())
                normalisation += np.sum(int_norm*self.wdetJmes[elt].diagonal())
        return Err/normalisation

    def PgPhysique(self,nbg):
        PG = dict()
        for elt in self.Get_listeltot():   # Boucle sur les élts        
            pts = self.noelem[elt]   # self.ien[0][:,elt] donne les numéros des fonctions associées            
            pts = self.Get_Btot()[:-1,pts].T   # Trouver les coord des points de numéros trouvés juste avant (grâce à eta)            
            N = self.phi[elt]   # N[pg] contient les valeurs des fonctions au pg
            Xg = N.dot(pts)
            PG[elt]=Xg
        return PG


def PVDFile(fileName,npart,nstep):
    rep=fileName.rfind('/')+1
    if rep==0:
        dir0=''
    else:
        dir0=fileName[:rep]
    if not os.path.isdir('vtk/'+dir0):
        os.mkdir('vtk/'+dir0)
    vtk.PVDFile('vtk/'+fileName,'vtu',npart,nstep)



############################################################################################################
#%% Classe résolution Pb méca
############################################################################################################

class Mechanalysis:
    def __init__(self,mesh,D):
        self.mesh=mesh          # Mesh object
        self.D=D                # Relation de comportement (matrice 3*3)

        self.rep = []           # ddls restant après déplacement imposé

        self.K = []
        self.U = np.zeros(self.mesh.Get_ndof()) # U réduit
        self.F = np.zeros(self.mesh.Get_ndof())
        self.U0 = np.zeros(self.mesh.Get_ndof()) # contient les déplacements imposés


    def Stiffness(self,JCP):
        """Calcul de la matrice K"""
        if np.all(self.mesh.ien==0) or np.all(self.mesh.tripleien==0) or np.all(self.mesh.noelem==0) or np.sum(self.mesh.iperM)==0:   #####bof
            self.mesh.Connectivity()
        self.mesh.GaussIntegration()  
        self.K = self.mesh.Stiffness(self.D,JCP) 
        
    def Add_Load(self,F0,rep,n,type_F='uniform'):
        """Permet d'ajouter des effort à un vecteur F d'efforts existant (pouvant contenir déjà des efforts)"""
#        self.F = self.mesh.Neumann(rep3,np.array([-F0,0]))+m_opt.Neumann(rep4,np.array([0,F0])) #### plaque trouée : rep3 gauche, rep4 haut
        if type_F == 'uniform':
            self.F += self.mesh.Neumann(rep,F0*n)
        elif type_F == 'infinite beam':
            self.F += self.mesh.Neumann_PlateHole(rep,n)
    
    def Add_ImposedU(self,u0,rep):
        """Permet d'ajouter un déplacement imposé sur les ddl rep"""
        self.U0[rep] = u0 # on impose le deplacement
        self.rep = np.delete(self.rep,np.where(np.in1d(self.rep,rep,assume_unique=True))) # màj des ddl à garder
                        
    def SolveUReducedPb(self):
        """Résol KU=F dans le cas où certains points de contrôle sont confondus"""
        pbR = Mechanalysis(self.mesh,self.D)
        pbR.K = self.mesh.iperM.T.dot(self.K.dot(self.mesh.iperM))
        pbR.U = np.zeros(self.mesh.iperM.shape[1]) # U réduit
        UU = np.zeros(self.mesh.Get_ndof());UU[self.rep] = 1. # UU sert juste à sommer certaines lignes dans iperM
        pbR.rep = np.where(self.mesh.iperM.T.dot(UU))[0] # rep_tilde = un équivalent de rep, mais pour le pb "réduit"
        pbR.F = self.mesh.iperM.T.dot(self.F)
        pbR.SolveU()
        self.U = self.mesh.iperM.dot(pbR.U)
        
    def SolveU(self):
        """Résol KU=F dans le cas où les points de contrôle ne sont pas confondus"""
        repk = np.ix_(self.rep,self.rep)
        
####################################ASUPPR
        ordre = sps.csgraph.reverse_cuthill_mckee(self.K[repk]).astype('int64')
#        ordre = np.random.permutation(self.K[repk].shape[0])
        rep_ordre = np.ix_(ordre,ordre)
        KLU1=splalg.splu(self.K[repk][rep_ordre])
        TMP = np.zeros_like(self.U[self.rep])
        TMP[ordre]=KLU1.solve(self.F[self.rep][ordre])
        self.U[self.rep] = TMP
##############################################              
        
#        KLU2=splalg.splu(self.K[repk])
#        self.U[self.rep]=KLU2.solve(self.F[self.rep])


    def Solve(self):
        """Résolution KU=F dans le cas général"""
        if any(self.U0!=0):
            self.F = self.F-self.K.dot(self.U0)
            
        if np.max(np.sum(self.mesh.iperM,axis=0))>1:
            self.SolveUReducedPb()
        else:
            self.SolveU()
        
        if any(self.U0!=0):
            self.U += self.U0
    
    def Compliance(self):
        return self.F.T.dot(self.U)


###############################################################################
#import scipy.sparse as sps
#import numpy.linalg as npl
#AAA = sps.diags(np.array([1,2,3,4,0.1])); AAA = AAA.tocsc(); AAA[2,3]=1; AAA[2,0]=2; AAA[2,1]=3
#AAA[4,1]=5;AAA[4,0]=5
#bbb = np.ones(5)
#aaa = AAA.todense()
#print(npl.solve(aaa,bbb))
#KLU = splalg.splu(AAA)
#print(KLU.solve(bbb))
#
#ordre = np.array([3,1,0,2,4])
#ORDRE = np.ix_(ordre,ordre)
#print(KLU.solve(bbb)[ordre])
#print(npl.cond(aaa))
#print(npl.solve(aaa[ORDRE],bbb[ordre]))
#KLU = splalg.splu(AAA[ORDRE])
#print(KLU.solve(bbb[ordre]))









#%%
        
def EvaluateExactStress(x,y):
   # Pour evaluer les contraintes exactes 
   # Passage en coordonnées cartésiennes 
   # Attention : s_xy est la valeur de la contrainte NON multipliée par sqrt(2)

    z = x+1.j*y
    r = np.abs(z)
    t = np.angle(z)
    R=1; Tx = 10 
    srr = 0.5*Tx*(1-R**2/r**2) + 0.5*Tx*(1+3*R**4/r**4-4*R**2/r**2)*np.cos(2*t)
    stt = 0.5*Tx*(1+R**2/r**2) - 0.5*Tx*(1+3*R**4/r**4)*np.cos(2*t)
    srt = -0.5*Tx*(1-3*R**4/r**4+2*R**2/r**2)*np.sin(2*t)
    sxx = np.cos(t)**2*srr -2*np.cos(t)*np.sin(t)*srt+np.sin(t)**2*stt
    sxy = np.sin(t)*np.cos(t)*srr +(np.cos(t)**2-np.sin(t)**2)*srt-np.sin(t)*np.cos(t)*stt
    syy = np.sin(t)**2*srr + 2*np.cos(t)*np.sin(t)*srt + np.cos(t)**2*stt
    return sxx,syy,sxy



def ExactSolCurvedBeam(x,y,a,b,E,U0,D):
#    print(x)
#    print(y)
    
    # Calcule la déformation exacte aux pts x,y pour la poutre courbe
#    import pdb; pdb.set_trace()
    N = a**2-b**2+(a**2+b**2)*np.log(b/a)
    P = -E*N*U0/(np.pi*(a**2+b**2))

    z = x+1.j*y
    r = np.abs(z)
    t = np.angle(z)
        
    srr = P/N*(r+((a**2*b**2)/r**3)-((a**2+b**2)/r))*np.sin(t)
    stt = P/N*(3*r-((a**2*b**2)/r**3)-((a**2+b**2)/r))*np.sin(t)
    srt = -P/N*(r+((a**2*b**2)/r**3)-((a**2+b**2)/r))*np.cos(t) 
    
#    print(srr)
#    print(stt)
#    print(srt)
    
    sxx = np.cos(t)**2*srr -2*np.cos(t)*np.sin(t)*srt+np.sin(t)**2*stt
    sxy = np.sin(t)*np.cos(t)*srr +(np.cos(t)**2-np.sin(t)**2)*srt-np.sin(t)*np.cos(t)*stt
    syy = np.sin(t)**2*srr + 2*np.cos(t)*np.sin(t)*srt + np.cos(t)**2*stt
    
    sig_ex = np.c_[sxx, syy, np.sqrt(2)*sxy].T  
#    print(sig_ex)    
    epsi_ex = np.linalg.solve(D,sig_ex)
    
    return epsi_ex







### TESTS A SUPPR

#def fct1(y):
#    sxx,syy,sxy = EvaluateExactStress(4,y)
#    return sxx
#
#def fct2(x):
#    sxx,syy,sxy = EvaluateExactStress(x,4)
#    return syy
#
#
#def RectanglesGauche(f,a,b,N):
#    x = np.linspace(a,b,N+1) # xk
#    xk1 = x[:-1] # pour la somme des xk à gauche
#   # xk = x[1:] # pour la somme des xk à droite
#    I = ((b-a)/N)*(np.sum(fct1(xk1)))       # rectangle gauche
#    return I
#
#print(RectanglesGauche(fct1,0,4,1000))
#print(RectanglesGauche(fct2,0,4,1000))


#%%
#m=Mesh(ctrlPts,np.array([ppu,ppv]),(XXsiu,XXsiv))
#nu,nv=m.Get_nunv()
#ne=m.Get_nenunv()


def derbasisfuncVectorInput(p,U,u,nb_u_values,span,nders):
    ders_matrix = np.zeros(((nders+1)*(nb_u_values),p+1))
    for i in range(nb_u_values):
        ders = nb.derbasisfuns(span,p,U,nders,u[i])
        ders_matrix[2*i:2*(i+1),:] = ders 
    return ders_matrix 
def derbasisfuns_c(p,U,u,nb_u_values,span,nders):
    ders = derbasisfuncVectorInput(p,U,u,nb_u_values,span,nders)
    N = ders[::2,:]
    dN = ders[1::2,:]
    return N,dN 

























