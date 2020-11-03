
import numpy as np
import scipy.special as spe
import scipy.sparse as sps

#%% Semble Ok. Bien penser à mettre les knots à ajouter sous forme d'un vecteur np.array (même s'il n'y en a qu'un)
def bspkntins(d,c,k,u):
    ''' Function Name:  
    #   bspkntins - Insert knots into a univariate B-Spline. 
    # Calling Sequence:  
    #   [ic,ik] = bspkntins(d,c,k,u) 
    # Parameters: 
    #   d	: Degree of the B-Spline. 
    #   c	: Control points, matrix of size (dim,nc). 
    #   k	: Knot sequence, row vector of size nk. 
    #   u	: Row vector of knots to be inserted, size nu 
    #   ic	: Control points of the new B-Spline, of size (dim,nc+nu) 
    #   ik	: Knot vector of the new B-Spline, of size (nk+nu) 
    # Description: 
    #   Insert knots into a univariate B-Spline. This function provides an 
    #   interface to a toolbox 'C' routine. '''
    mc,nc = c.shape
    nu = len(u) 
    nk = len(k) 
                                                         #  
                                                         # int bspkntins(int d, double *c, int mc, int nc, double *k, int nk, 
                                                         #               double *u, int nu, double *ic, double *ik) 
                                                         # { 
                                                         #   int ierr = 0; 
                                                         #   int a, b, r, l, i, j, m, n, s, q, ind; 
                                                         #   double alfa; 
                                                         # 
                                                         #   double **ctrl  = vec2mat(c, mc, nc); 
    ic = np.zeros((mc,nc+nu))                            #   double **ictrl = vec2mat(ic, mc, nc+nu); 
    ik = np.zeros(nk+nu) 
                                                         # 
    n = c.shape[1] - 1                                   #   n = nc - 1; 
    r = len(u) - 1                                       #   r = nu - 1; 
                                                         # 
    m = n + d + 1                                        #   m = n + d + 1; 
    a = findKnotSpan(u[0], k, d)                         #   a = findspan(n, d, u[0], k); ###
    b = findKnotSpan(u[r], k, d)                         #   b = findspan(n, d, u[r], k); ###
    b+=1                                                 #   ++b; 
                                                         # 
    for q in range(mc):                                  #   for (q = 0; q < mc; q++)  { 
        for j in range(a-d+1):
            ic[q,j] = c[q,j]                              #     for (j = 0; j <= a-d; j++) ictrl[j][q] = ctrl[j][q]; 
        for j in range(b-1,n+1):
            ic[q,j+r+1] = c[q,j]                          #     for (j = b-1; j <= n; j++) ictrl[j+r+1][q] = ctrl[j][q]; 
                                                         #   } 
    
    for j in range(a+1):
        ik[j] = k[j]                                     #   for (j = 0; j <= a; j++)   ik[j] = k[j]; 
    for j in range(b+d,m+1):
        ik[j+r+1] = k[j]                                 #   for (j = b+d; j <= m; j++) ik[j+r+1] = k[j]; 
                                                         # 
    i = b + d - 1                                        #   i = b + d - 1; 
    s = b + d + r                                        #   s = b + d + r; 
    
    
    
    for j in range(r,-1,-1):                             #   for (j = r; j >= 0; j--) { 
        while (u[j] <= k[i] and i > a):                   #     while (u[j] <= k[i] && i > a) { 
            for q in range(mc):                           #       for (q = 0; q < mc; q++) 
                ic[q,s-d-1] = c[q,i-d-1]                  #         ictrl[s-d-1][q] = ctrl[i-d-1][q]; 
                                                    
            ik[s] = k[i]                                  #       ik[s] = k[i]; 
            s -= 1                                        #       --s; 
            i -= 1                                        #       --i; 
                                                         #     } 

        for q in range(mc):                               #     for (q = 0; q < mc; q++) 
            ic[q,s-d-1] = ic[q,s-d]                       #       ictrl[s-d-1][q] = ictrl[s-d][q]; 
     
        for l in range(1,d+1):                            #     for (l = 1; l <= d; l++)  { 
            ind = s - d + l                               #       ind = s - d + l; 
            alfa = ik[s+l] - u[j]                         #       alfa = ik[s+l] - u[j]; 
            if abs(alfa) == 0:                            #       if (fabs(alfa) == 0.0) 
                for q in range(mc):                       #         for (q = 0; q < mc; q++) 
                    ic[q,ind-1] = ic[q,ind]               #           ictrl[ind-1][q] = ictrl[ind][q]; 
            else:                                         #       else  { 
                alfa = alfa/(ik[s+l] - k[i-d+l])          #         alfa /= (ik[s+l] - k[i-d+l]); 
                for q in range(mc):                       #         for (q = 0; q < mc; q++) 
                    tmp = (1.-alfa)*ic[q,ind] 
                    ic[q,ind-1] = alfa*ic[q,ind-1] + tmp  #           ictrl[ind-1][q] = alfa*ictrl[ind-1][q]+(1.0-alfa)*ictrl[ind][q]; 
                                                         #       } 
                                                         #     } 
       # 
        ik[s] = u[j]                                      #     ik[s] = u[j]; 
        s -= 1                                            #     --s; 
                                                         #   } 
                                                         # 
                                                         #   freevec2mat(ctrl); 
                                                         #   freevec2mat(ictrl); 
                                                         # 
                                                         #   return ierr; 
                                                         # } 
    return ic,ik



#%% Ok
    
def GaussLegendre(n):
#     [nodes,weigths]=GaussLegendre(n)
#
# Generates the abscissa and weights for a Gauss-Legendre quadrature.
# Reference:  Numerical Recipes in Fortran 77, Cornell press.

    xg = np.zeros(n)                                           # Preallocations.
    wg = xg.copy()
    m = (n+1)/2
    #import pdb; pdb.set_trace()
    for ii in range(int(m)): # for ii=1:m
        z = np.cos(np.pi*(ii+1-0.25)/(n+0.5))                        # Initial estimate.
        z1 = z+1
        while np.abs(z-z1)>np.finfo(np.float).eps:
            p1 = 1
            p2 = 0
            for jj in range(n): #for jj = 1:n
                p3 = p2
                p2 = p1
                p1 = ((2*jj+1)*z*p2-(jj)*p3)/(jj+1)      # The Legendre polynomial.
            
            pp = n*(z*p1-p2)/(z**2-1)                        # The L.P. derivative.
            z1 = z
            z = z1-p1/pp
        
        xg[ii] = -z                                   # Build up the abscissas.
        xg[-1-ii] = z
        wg[ii] = 2/((1-z**2)*(pp**2))                     # Build up the weights.
        wg[-1-ii] = wg[ii]
    
    
    return xg,wg
    

#%% Semble OK
    
def nubsconnect(p,n):
    nel = n-p
    nen = p+1
#    IEN = np.zeros((nen,nel),dtype='int')
#    for i in range(nen):
#        for j in range(nel):
#            IEN[i,j]=p+j+1-i
    tmp1,tmp2 = np.meshgrid(np.arange(nel),np.arange(nen))
    IEN = np.flipud(tmp1+tmp2)
    return IEN#,nen,nel

#%% Size of elements of XSI knot vector

def MesXsi(xsi,p):
    return xsi[p+1:-p]-xsi[p:-p-1]


#%% semble OK / attention à la forme de listel en entrée (ici, codé pour listel commençant à 1 comme Matlab)

def nubsmatKrigidite(n,E,S,IEN,XXsi,listel,nen,B,listB,p):
    K = np.zeros((n,n))
    #[xpg,apg] = fonction pour points de gauss
    xpg,apg = GaussLegendre(p+1)
    for ne in listel:
        ni = int(IEN[0,ne]) ### attention numerotation indices
        #print('ni = '+str(ni))
#        print('Xxsi = '+str(Xxsi))
        xsii = XXsi[ni]
        xsii1 = XXsi[ni+1]
#        print('xi, xi1 = '+str(xsii)+' et '+str(xsii1))
        if (xsii1-xsii)>0:
            Ke = np.zeros((nen,nen))
            for pg in range(p+1):
                xsipg = xsii + (xpg[pg]+1)*(xsii1-xsii)/2
                DN = derbasisfuns(ni,p,XXsi,1,xsipg)
                dN = DN[1,:]
                #print('dN = '+str(dN))
                jac = 0
                for k in range(nen):
#                    import pdb; pdb.set_trace()
                    jac = jac + dN[k]*B[0,ni-p+k]
                dxsidx = 1/jac
                dxsidxsitilde = (xsii1-xsii)/2
                #import pdb; pdb.set_trace()
                Ke += apg[pg]*E*S*np.outer(dN,dN)*dxsidx*dxsidxsitilde
            im = np.zeros(nen)
            for kk in range(nen):
                #import pdb; pdb.set_trace()
                no = IEN[nen-kk-1,ne]
                ###
                ###
                im[kk]=no
            for i in range(nen):
                for j in range(nen):
                    #import pdb; pdb.set_trace()
                    K[int(im[i]),int(im[j])] += Ke[i,j]
    return K


#%% derbasisfuns _ semble OK

# This file contains the basis functions and derivatives routine. 
# The algorithm is taken from Piegl, Les. "The NURBS Book". Springer-Verlag: 
# Berlin 1995; p. 72-73.
# 
# dersbasisfuns.f file created by T.Elguedj on 29/08/2009
# Last edition by T. Elguedj 22/02/2010
# 
# 
# The routine consumes a knot index, parameter value, and a knot
#  vector and returns a vector containing all nonzero 1D b-spline shape
#  functions evaluated at that parameter as well as their derivatives.
#       
#       subroutine dersbasisfuns(i,pl,ml,u,nders,u_knotl,ders)
#       
#       IMPLICIT NONE
# 
#    --------------VARIABLE DECLARATIONS--------------------------------
#   knot span, degree of curve, number of control points, counters
#       integer i, pl, ml, j, r, k, j1, j2, s1, s2, rk, pk,nders
#   parameter value, vector of knots, derivative matrix
#       real*8 u, u_knotl(pl+ml+1), ders(nders+1,pl+1), ndu(pl+1,pl+1),d
# 
#       real*8 left(pl+1), right(pl+1), saved, temp, a(2,pl+1)
# 
#     -------------------------------------------------------------------

def derbasisfuns(i,pl,U,nders,u):
    '''
    # i = numéro de la fonction à calculer (sortie de findspan)
    # pl = degrés de la nurbs
    # u = endroit ou l'on veut la fonction
    # nders = numéro de la dérivée désirée
    # U = vecteur de noeud de la fonction'''
    
#    import pdb; pdb.set_trace()
    u_knotl=U.copy()
    left = np.zeros((pl+1))
    right = np.zeros((pl+1))
    ndu = np.zeros((pl+1,pl+1))
    ders = np.zeros((nders+1,pl+1))
    ndu[0,0] = 1
    for j in range(pl):   #1:pl
        left[j+1] = u - u_knotl[i-j]   ### rq Ali : i-j au lieu de i-j-1
        right[j+1] = u_knotl[i+j+1] - u ### rq : i+j+1 au lieu de i+j
        saved = 0
        for r in range(j+1):   #0:j-1
            ndu[j+1,r] = right[r+1] + left[j-r+1]
            temp = ndu[r,j]/ndu[j+1,r]
            ndu[r,j+1] = saved + right[r+1]*temp
            saved = left[j-r+1]*temp
        ndu[j+1,j+1] = saved
#    print('checkpoint1 : '+str(ndu))
      
                                    # load basis functions
    for j in range(pl+1):   #0:pl
        ders[0,j] = ndu[j,pl]
#    print('checkpoint2 : '+str(ders))

                                # compute derivatives
    for r in range(pl+1): # 0:pl              # loop over function index
        s1 = 0
        s2 = 1                # alternate rows in array a
        a = np.zeros((nders+1,nders+1))
        a[0,0] = 1
                    # loop to compute kth derivative
        for k in range(nders):   # 1:nders
            d = 0
            rk = r-(k+1)
            pk = pl-(k+1)
            if (r >= (k+1)):
                a[s2,0] = a[s1,0]/ndu[pk+1,rk]
                d = a[s2,0]*ndu[rk,pk]
            if (rk >= -1):
                j1 = 1
            else: 
                j1 = -rk
            if ((r-1) <= pk): 
                j2 = k
            else: 
                j2 = pl-r
            for j in np.arange(j1,j2+0.1):   #j1:j2
                j = int(j)
                a[s2,j] = (a[s1,j] - a[s1,j-1])/ndu[pk+1,rk+j]
                d = d + a[s2,j]*ndu[rk+j,pk]
            if (r <= pk): 
                a[s2,k+1] = -a[s1,k]/ndu[pk+1,r]
                d = d + a[s2,k+1]*ndu[r,pk]
            ders[k+1,r] = d
            j = s1
            s1 = s2
            s2 = j            # switch rows

      
    #     Multiply through by the correct factors
     
    r = pl
    for k in range(nders):   # 1:nders
        for j in range(pl+1):   # 0:pl
            ders[k+1,j] = ders[k+1,j]*r
        r = r*(pl-(k+1))

    return ders

#%% Semble ok

def bspdegelev(d,c,k,t):
    ''' 
    # Function Name:  
    #   bspdegevel - Degree elevate a univariate B-Spline. 
    # Calling Sequence: 
    #   [ic,ik] = bspdegelev(d,c,k,t) 
    # Parameters: 
    #   d	: Degree of the B-Spline. 
    #   c	: Control points, matrix of size (dim,nc). 
    #   k	: Knot sequence, row vector of size nk. 
    #   t	: Raise the B-Spline degree t times. 
    #   ic	: Control points of the new B-Spline. 
    #   ik	: Knot vector of the new B-Spline. 
    # Description: 
    #   Degree elevate a univariate B-Spline. This function provides an 
    #   interface to a toolbox 'C' routine. 
    '''
    mc,nc = c.shape 
                                                          # 
                                                          # int bspdegelev(int d, double *c, int mc, int nc, double *k, int nk, 
                                                          #                int t, int *nh, double *ic, double *ik) 
                                                          # { 
                                                          #   int row,col 
                                                          # 
                                                          #   int ierr = 0; 
                                                          #   int i, j, q, s, m, ph, ph2, mpi, mh, r, a, b, cind, oldr, mul; 
                                                          #   int n, lbz, rbz, save, tr, kj, first, kind, last, bet, ii; 
                                                          #   double inv, ua, ub, numer, den, alf, gam; 
                                                          #   double **bezalfs, **bpts, **ebpts, **Nextbpts, *alfs; 
                                                          # 
    #init ic                                                      #   double **ctrl  = vec2mat(c, mc, nc); 
    ic = np.zeros((mc,nc*(t+1)))                                  #   double **ictrl = vec2mat(ic, mc, nc*(t+1)); 
    ik = np.zeros((t+1)*k.shape[0])
                                                         # 
    n = nc - 1                                               #   n = nc - 1; 
                                                              # 
    bezalfs = np.zeros((d+1,d+t+1))                              #   bezalfs = matrix(d+1,d+t+1); 
    bpts = np.zeros((mc,d+1))                                     #   bpts = matrix(mc,d+1); 
    ebpts = np.zeros((mc,d+t+1))                                 #   ebpts = matrix(mc,d+t+1); 
    Nextbpts = np.zeros((mc,d+1))                                 #   Nextbpts = matrix(mc,d+1); 
    alfs = np.zeros((d,1))                                        #   alfs = (double *) mxMalloc(d*sizeof(double)); 
                                                              # 
    m = n + d + 1                                            #   m = n + d + 1; 
    ph = d + t                                               #   ph = d + t; 
    ph2 = int(ph/2)                                      #   ph2 = ph / 2; 
                                                              # 
                                                              #   // compute bezier degree elevation coefficeients 
    bezalfs[0,0] = 1.                                         #   bezalfs[0][0] = bezalfs[ph][d] = 1.0; 
    bezalfs[d,ph] = 1.                                   # 
     
    for i in np.arange(1,ph2+1):   #1:ph2                                               #   for (i = 1; i <= ph2; i++) { 
        inv = 1/bincoeff(ph,i)                                #     inv = 1.0 / bincoeff(ph,i); 
        mpi = min(d,i)                                        #     mpi = min(d,i); 
                                                              # 
        for j in np.arange(max(0,i-t),mpi+1):   #max(0,i-t):mpi      #     for (j = max(0,i-t); j <= mpi; j++) 
            bezalfs[j,i] = inv*bincoeff(d,j)*bincoeff(t,i-j)  #       bezalfs[i][j] = inv * bincoeff(d,j) * bincoeff(t,i-j); 
                                                              # 
    for i in np.arange(ph2+1,ph):   #ph2+1:ph-1                                          #   for (i = ph2+1; i <= ph-1; i++) { 
        mpi = min(d,i)                                        #     mpi = min(d, i); 
        for j in np.arange(max(0,i-t),mpi+1):   #max(0,i-t):mpi                                   #     for (j = max(0,i-t); j <= mpi; j++) 
            bezalfs[j,i] = bezalfs[d-j,ph-i]         #       bezalfs[i][j] = bezalfs[ph-i][d-j]; 
                                                              # 
    mh = ph                                                  #   mh = ph;       
    kind = ph+1                                              #   kind = ph+1; 
    r = -1                                                   #   r = -1; 
    a = d                                                    #   a = d; 
    b = d+1                                                  #   b = d+1; 
    cind = 1                                                 #   cind = 1; 
    ua = k[0]                                                #   ua = k[0];  
                                                              # 
    for ii in range(mc):  #0:mc-1                                             #   for (ii = 0; ii < mc; ii++) 
        ic[ii,0] = c[ii,0]                                #     ictrl[0][ii] = ctrl[0][ii]; 
    for i in range(ph+1): #0:ph                                                #   for (i = 0; i <= ph; i++) 
        ik[i] = ua                                          #     ik[i] = ua; 
                                                              #   // initialise first bezier seg 
    for i in range(d+1): #0:d                                                 #   for (i = 0; i <= d; i++) 
        for ii in range(mc): #0:mc-1                                          #     for (ii = 0; ii < mc; ii++) 
            bpts[ii,i] = c[ii,i]                       #       bpts[i][ii] = ctrl[i][ii]; 
                                                              #   // big loop thru knot vector 
    while b < m  :                                             #   while (b < m)  { 
        i = b                                                 #     i = b; 
        while b < m and k[b] == k[b+1]  :                      #     while (b < m && k[b] == k[b+1]) 
            b = b + 1                                          #       b++; 
        mul = b - i + 1                                       #     mul = b - i + 1; 
        mh += mul + t                                     #     mh += mul + t; 
        ub = k[b]                                           #     ub = k[b]; 
        oldr = r                                              #     oldr = r; 
        r = d - mul                                           #     r = d - mul; 
                                                              # 
                                                              #     // insert knot u(b) r times 
        if oldr > 0:                                           #     if (oldr > 0) 
#            lbz = np.floor((oldr+2)/2)   #####25/01/2019                          #       lbz = (oldr+2) / 2; 
            lbz = (oldr+2)//2                           #       lbz = (oldr+2) / 2; 
        else :                                                  #     else 
            lbz = 1                                            #       lbz = 1; 
        
        if r > 0 :                                              #     if (r > 0) 
#            rbz = ph - np.floor((r+1)/2)    #####25/01/2019                        #       rbz = ph - (r+1)/2; 
            rbz = ph - (r+1)//2                          #       rbz = ph - (r+1)/2; 
        else :                                                  #     else 
            rbz = ph                                           #       rbz = ph; 
        
        if r > 0 :                                             #     if (r > 0) { 
                                                              #       // insert knot to get bezier segment 
            numer = ub - ua                                    #       numer = ub - ua; 
            for q in np.arange(d,mul,-1):    #d:-1:mul+1                                    #       for (q = d; q > mul; q--) 
                alfs[q-mul-1] = numer / (k[a+q]-ua)             #         alfs[q-mul-1] = numer / (k[a+q]-ua); 
           
            for j in np.arange(1,r+1):   #1:r                                           #       for (j = 1; j <= r; j++)  { 
                save = r - j                                    #         save = r - j; 
                s = mul + j                                     #         s = mul + j; 
                                                              # 
                for q in np.arange(d,s-1,-1): #d:-1:s                                     #         for (q = d; q >= s; q--) 
                    for ii in range(mc): #0:mc-1                                 #           for (ii = 0; ii < mc; ii++) 
                        tmp1 = alfs[q-s]*bpts[ii,q]  
#                        tmp2 = (1-alfs(q-s))*bpts(ii,q-1)  #####24/01/2019
                        tmp2 = (1-alfs[q-s])*bpts[ii,q-1]  
                        bpts[ii,q] = tmp1 + tmp2              #             bpts[q][ii] = alfs[q-s]*bpts[q][ii]+(1.0-alfs[q-s])*bpts[q-1][ii]; 
              
                for ii in range(mc): #0:mc-1                                    #         for (ii = 0; ii < mc; ii++) 
                    Nextbpts[ii,save] = bpts[ii,d]       #           Nextbpts[save][ii] = bpts[d][ii]; 
                                                              #     // end of insert knot 
                                                              # 
                                                              #     // degree elevate bezier 
        for i in np.arange(lbz,ph+1):  #lbz:ph                                           #     for (i = lbz; i <= ph; i++)  { 
            for ii in range(mc): #0:mc-1                                       #       for (ii = 0; ii < mc; ii++) 
                ebpts[ii,i] = 0                             #         ebpts[i][ii] = 0.0; 
            mpi = min(d, i)                                    #       mpi = min(d, i); 
            for j in np.arange(max(0,i-t),mpi+1): #max(0,i-t):mpi                                #       for (j = max(0,i-t); j <= mpi; j++) 
                for ii in range(mc): #0:mc-1                                    #         for (ii = 0; ii < mc; ii++) 
                    tmp1 = ebpts[ii,i]  
                    tmp2 = bezalfs[j,i]*bpts[ii,j] 
                    ebpts[ii,i] = tmp1 + tmp2                #           ebpts[i][ii] = ebpts[i][ii] + bezalfs[i][j]*bpts[j][ii]; 
                                                              #     // end of degree elevating bezier 
                                                              # 
        if oldr > 1 :                                           #     if (oldr > 1)  { 
                                                              #       // must remove knot u=k[a] oldr times 
            first = kind - 2                                                    #       first = kind - 2; 
            last = kind                                        #       last = kind; 
            den = ub - ua                                      #       den = ub - ua; 
            bet = np.floor((ub-ik[kind-1]) / den)                   #       bet = (ub-ik[kind-1]) / den; 
                                                              # 
                                                              #       // knot removal loop 
            for tr in np.arange(1,oldr):  #1:oldr-1                                     #       for (tr = 1; tr < oldr; tr++)  { 
                i = first                                       #         i = first; 
                j = last                                        #         j = last; 
                kj = j - kind + 1                               #         kj = j - kind + 1; 
                while j-i > tr :                                  #         while (j - i > tr)  { 
                                                              #           // loop and compute the new control points 
                                                              #           // for one removal step 
                    if i < cind  :                                 #           if (i < cind)  { 
                        alf = (ub-ik[i])/(ua-ik[i])           #             alf = (ub-ik[i])/(ua-ik[i]); 
                        for ii in range(mc):  #0:mc-1                              #             for (ii = 0; ii < mc; ii++) 
                            tmp1 = alf*ic[ii,i] 
                            tmp2 = (1-alf)*ic[ii,i-1]  
                            ic[ii,i] = tmp1 + tmp2             #               ictrl[i][ii] = alf * ictrl[i][ii] + (1.0-alf) * ictrl[i-1][ii]; 
                    if j >= lbz :                                   #           if (j >= lbz)  { 
                        if j-tr <= kind-ph+oldr :                   #             if (j-tr <= kind-ph+oldr) { 
                            gam = (ub-ik[j-tr]) / den            #               gam = (ub-ik[j-tr]) / den; 
                            for ii in range(mc):  #0:mc-1                           #               for (ii = 0; ii < mc; ii++) 
                                tmp1 = gam*ebpts[ii,kj]  
                                tmp2 = (1-gam)*ebpts[ii,kj+1]  
                                ebpts[ii,kj] = tmp1 + tmp2      #                 ebpts[kj][ii] = gam*ebpts[kj][ii] + (1.0-gam)*ebpts[kj+1][ii]; 
                        else :                                      #             else  { 
                            for ii in range(mc):  #0:mc-1                           #               for (ii = 0; ii < mc; ii++) 
                                tmp1 = bet*ebpts[ii,kj]                                      
                                tmp2 = (1-bet)*ebpts[ii,kj+1]                                      
                                ebpts[ii,kj] = tmp1 + tmp2      #                 ebpts[kj][ii] = bet*ebpts[kj][ii] + (1.0-bet)*ebpts[kj+1][ii]; 
                    i += 1                                    #           i++; 
                    j -= 1                                    #           j--; 
                    kj -= 1                                  #           kj--; 
                                                              # 
                first -= 1                               #         first--; 
                last += 1                                 #         last++; 
                                                              #     // end of removing knot n=k[a] 
                                                              # 
                                                              #     // load the knot ua 
        if a != d :                                             #     if (a != d) 
            for i in range(ph-oldr):   #0:ph-oldr-1                                   #       for (i = 0; i < ph-oldr; i++)  { 
                ik[kind] = ua                                 #         ik[kind] = ua; 
                kind += 1                                 #         kind++; 
                                                          # 
                                                          #     // load ctrl pts into ic 
        for j in np.arange(lbz,rbz+1):   #lbz:rbz                                       #     for (j = lbz; j <= rbz; j++)  { 
            for ii in range(mc):   #0:mc-1                                    #       for (ii = 0; ii < mc; ii++) 
                ic[ii,cind] = ebpts[ii,j]            #         ictrl[cind][ii] = ebpts[j][ii]; 
            cind += 1                                 #       cind++; 
                                                          # 
        if b < m :                                           #     if (b < m)  { 
                                                          #       // setup for next pass thru loop 
            for j in range(r): #0:r-1                                      #       for (j = 0; j < r; j++) 
                for ii in range(mc): #0:mc-1                                 #         for (ii = 0; ii < mc; ii++) 
                    bpts[ii,j] = Nextbpts[ii,j]       #           bpts[j][ii] = Nextbpts[j][ii]; 
            for j in np.arange(r,d+1):  #r:d                                        #       for (j = r; j <= d; j++) 
                for ii in range(mc):  #0:mc-1                                 #         for (ii = 0; ii < mc; ii++) 
                    bpts[ii,j] = c[ii,b-d+j]          #           bpts[j][ii] = ctrl[b-d+j][ii]; 
            a = b                                           #       a = b; 
            b += 1                                         #       b++; 
            ua = ub                                         #       ua = ub; 
                                                          #     } 
        else:                                                #     else 
                                                      #       // end knot 
            for i in range(ph+1):   #0:ph                                       #       for (i = 0; i <= ph; i++) 
                ik[kind+i] = ub                            #         ik[kind+i] = ub; 
    # End big while loop                                      #   // end while loop 
                                                              # 
                                                              #   *nh = mh - ph - 1; 
                                                              # 
                                                              #   freevec2mat(ctrl); 
                                                              #   freevec2mat(ictrl); 
                                                              #   freematrix(bezalfs); 
                                                              #   freematrix(bpts); 
                                                              #   freematrix(ebpts); 
                                                              #   freematrix(Nextbpts); 
                                                              #   mxFree(alfs); 
                                                              # 
                                                              #   return(ierr); 
                                                              # } 
                                                              
    # ajout dû au fait qu'on a initialisé trop grand (car difficile d'estimer la taille de ic et ik avant, dépend entre autres de la multiplicité des knots)
    # on enleve les 0 à la fin du knot vector ik
    ik = np.trim_zeros(ik,'b')
    
    # on tronque la matrice des points de contrôle où il faut (revient à enlever les 0, mais si la courbe finit avec un point en (0,0), on n'enlève pas celui-là)
    n = len(ik)-(d+t)-1
    ic = ic[:,0:n]

    return ic,ik
#%% Ok                                                            
def bincoeff(n,k):
    #  Computes the binomial coefficient. 
    # 
    #      ( n )      n! 
    #      (   ) = -------- 
    #      ( k )   k!(n-k)! 
    # 
    #  b = bincoeff(n,k) 
    # 
    #  Algorithm from 'Numerical Recipes in C, 2nd Edition' pg215. 
     
                                                              # double bincoeff(int n, int k) 
                                                              # { 
    b = np.floor(0.5+np.exp(factln(n)-factln(k)-factln(n-k)));      #   return floor(0.5+exp(factln(n)-factln(k)-factln(n-k))); 
     
    return b
     
def factln(n):
    # computes ln(n!) 
    if n <= 1:
        f = 0
        return f
    
    f = spe.gammaln(n+1) #log(factorial(n));</pre>
    
    return f


#%%

def findKnotSpan(u,U,p):
    """
    Finds the knots space of a given knot parameter u in
    the knot vector U corresponding to the degree p 
    """
    m = np.size(U)
    if u==U[m-p-1]:
        k=m-p-2
    else :
        k=np.max(np.where(u>=U)) 
    return k   ###?-1


#%%
    
def Href_matrix(p,XSI,xsi):
    """Spline degree p (int), knot vector XSI (array), knots to be added xsi (array)
    Returns the matrix C for knot insertion so that C.T*B.T = Bnew.T, with B a dim*n matrix (dim dimension of the pb, n number of control points), Bnew new control points
    and returns XSI the new knot vector"""
    n = len(XSI)-p-1
    C = sps.eye(n)
    for jj in range(xsi.size):
        k = findKnotSpan(xsi[jj],XSI,p)
        alpha = np.zeros(n+jj+1)
        alpha[:(k-p+1)]=1
        alpha[(k-p+1):(k+1)]=(xsi[jj]-XSI[(k-p+1):(k+1)])/(XSI[(k+1):(k+p+1)]-XSI[(k-p+1):(k+1)])
        Cj = np.zeros((n+jj,1)); Cj[-1,0]=1-alpha[-1] # dernière colonne de Cj
        Cj = sps.hstack([sps.diags([alpha[:-1],1-alpha[1:-1]],[0,1]),Cj])
#        Cj = np.c_[np.diag(alpha[:-1])+np.diag((1-alpha[1:-1]),1),Cj]
        C = C.dot(Cj)
        XSI = np.r_[XSI[:(k+1)],xsi[jj],XSI[(k+1):]]
    return C,XSI

##TEST
XSI = np.array([0.,0.,0.,1.,1.,1.])
xsi = np.array([0.25,0.5,0.75])
#xsi = np.array([0.1,0.7,0.7,0.95])
B = np.array([[1.,1.,0.],[0.,1.,1.]])
#B = np.array([[1.,1.,0.],[0.,1.,1.],[1.,1.,1.]])
p=2
# Pour vérif
Btest,Xtest = bspkntins(p,B,XSI,xsi)
# Fonction matricielle
C,X = Href_matrix(p,XSI,xsi)
BHT = C.T.dot(B.T)

print(Xtest)
print(X)
#print(Btest.T,BHT)
print(Btest.T-BHT)
print(np.max(Btest.T-BHT))




#%%
    
def Pref_Bezier(p,XSI,t):
    """Spline degree p (int), knot vector XSI with maximum multiplicity for all knots (array),number of degree elevations t
    Returns the matrix M for degree elevation of Bezier fonctions so that M*B.T = Bnew.T, with B a dim*n matrix (dim dimension of the pb, n number of control points), Bnew new control points
    and returns XSI the new knot vector"""
    M = sps.eye(len(XSI)-p-1)
    for deg in range(t):
        Nelt = (len(XSI)-2)//p-1
        alpha = 1/(p+1)*np.arange(p+1)
        i_elt = np.r_[np.arange(p+1),np.arange(1,p+1)]
        j_elt = np.r_[np.arange(p+1),np.arange(p)]
        val_elt = np.r_[1-alpha,alpha[1:]]
        I = np.kron(np.ones(Nelt),i_elt)+np.kron(np.ones(2*p+1),np.arange(Nelt))*(p+1)
        J = np.kron(np.ones(Nelt),j_elt)+np.kron(np.ones(2*p+1),np.arange(Nelt))*p
        VAL = np.kron(np.ones(Nelt),val_elt)
        I = np.r_[I,Nelt*(p+1)]
        J = np.r_[J,Nelt*p]
        VAL = np.r_[VAL,1]
        M = (sps.csc_matrix((VAL,(I,J)),shape=(Nelt*(p+1)+1,Nelt*p+1))).dot(M)
        XSI = np.sort(np.r_[XSI,np.unique(XSI)])
        p+=1
    return M, XSI, p

##TEST
#XSI = np.array([0.,0.,0.,0.2,0.2,0.8,0.8,1.,1.,1.])
#p=2
#t=2
#B = np.array([[0.,1.,1.,2.5,1.,0.5,0.],[0.,0.5,0.8,1.,1.5,2.,3.]])
#CC,KK=bspdegelev(p,B,XSI,t)
#M,K,pp = Pref_Bezier(p,XSI,t)
#CCC = M.dot(B.T)
#print(CCC-CC.T)
#print(np.max(CCC-CC.T))

#%%

def Xsi_Bezier(XSI,p):
    """returns a vector of the knots that are to be added for Bézier decomposition"""
    XSI_unique,count = np.unique(XSI,return_counts=True)
    count = p-count
    count[[0,-1]]=0
    xsi = np.repeat(XSI_unique,count)
    return xsi


#%% TEST DEGREE ELEVATION
#XSI = np.array([0.,0.,0.,0.2,0.2,0.8,1.,1.,1.])
#p=2
#t=1
#B = np.array([[0.,1.,1.,2.5,1.,0.5],[0.,0.5,0.8,1.,1.5,2.]])
#
#B_degelev,X_degelev = bspdegelev(p,B,XSI,t) # pour validation
#
#Cins,XSIbarre = Href_matrix(p,XSI,Xsi_Bezier(XSI,p)) # décompo Bézier
#Bbarre = B.dot(Cins) # calcul nouveaux pts de controle
#
#D,XSItilde,ptilde = Pref_Bezier(p,XSIbarre,t) # élevation degré Bézier
#Btilde = (D.dot(Bbarre.T)).T # nvx pts de ctrl
#
#XSI_fin = np.sort(np.r_[XSI,np.unique(XSI)]) # knots qu'on doit avoir à la fin
#Crem,XSItilde2 = Href_matrix(ptilde,XSI_fin,Xsi_Bezier(XSI_fin,ptilde)) # décompo Bézier
#B_fin = Btilde.dot(np.linalg.pinv(Crem))


##########

def Pref_matrix(XSI,p,t):
    """Returns C so that Brefined.T = C.T B.T, ; XSI_fin the final knot vector and ptilde the final degree
    Returns the matrix C for degree elevation so that C.T*B.T = Bnew.T, with B a dim*n matrix (dim dimension of the pb, n number of control points), Bnew new control points
    and returns XSI the new knot vector"""
    Cins,XSIbarre = Href_matrix(p,XSI,Xsi_Bezier(XSI,p)) # décompo Bézier
    D,XSItilde,ptilde = Pref_Bezier(p,XSIbarre,t) # élevation degré Bézier
    XSI_fin = np.sort(np.r_[XSI,np.repeat(np.unique(XSI),t)]) # knots qu'on doit avoir à la fin
    Crem,XSItilde2 = Href_matrix(ptilde,XSI_fin,Xsi_Bezier(XSI_fin,ptilde)) # décompo Bézier
    C = Cins.dot(D.T).dot(sps.csr_matrix(np.linalg.pinv(Crem.todense()))) # Brefined.T = C.T B.T
    return C,XSI_fin,ptilde



#%% Bezier to Lagrange 1D

def Bezier2ToLagrange2(P_Bezier):
    """ Input : Control points P_Bezier (size dim*nu) for Bernstein polynoms. Degree = 2
    WARNING : P_Bezier should not contain weights (and they are all supposed to be equal to 1)
    Returns P_Lagrange the control points for Lagrange polynoms. Degree = 2"""
    P_Lagrange = np.zeros_like(P_Bezier)
    even = np.arange(0,P_Bezier.shape[1]+1,2)
    odd = np.arange(1,P_Bezier.shape[1],2)
    # Rq : en degré 2 et C0 aux nds/polynômes de Lagrange, il y a toujours un nombre impair de points de contrôle.
    P_Lagrange[:,even] = P_Bezier[:,even]
    P_Lagrange[:,odd] = 1/2*P_Bezier[:,odd]+1/4*P_Bezier[:,odd-1]+1/4*P_Bezier[:,odd+1]
    return P_Lagrange

def Matrix_B2ToL2(n): ### on n'a besoin que de la taille de P_Bezier
    """n le nombre de points de contrôle
    PL.T = M.T.dot(PB.T)"""
    d0 = np.r_[np.tile(np.array([1,1/2]),n//2),1]
    d1 = np.tile(np.array([0,1/4]),n//2)
    d_1 = np.tile(np.array([1/4,0]),n//2)
    return sps.diags([d0,d1,d_1],[0,1,-1]).T
    

    
#### Test
#import matplotlib.pyplot as plt
#XSI = np.array([0.,0.,0.,1.,1.,1.])
#P = np.array([[0., 0., 1.],[0., 1., 1.],[1,1,1]])
##PL = Bezier2ToLagrange2(P[:-1])
#PL = (Matrix_B2ToL2(P[:-1]).dot(P[:-1].T)).T
#xB = drawline(XSI,2,P,20,rational=False)
#plt.figure()
#plt.plot(P[0],P[1],'o',label='Bezier ctrl pts',markersize=8)
#plt.plot(PL[0],PL[1],'o',label='Lagrange ctrl pts',markersize=6)
#plt.plot(xB[0],xB[1])
#plt.legend()

#%%

def Matrix_N2ToL2(XSI,p):
    """PL.T = M.T.dot(PN.T)"""
    C_NB,xsiNew = Href_matrix(p,XSI,Xsi_Bezier(XSI,p))
    C_BL = Matrix_B2ToL2(len(xsiNew)-p-1)
    return C_NB.dot(C_BL)

####TEST
#import matplotlib.pyplot as plt
#x0 = np.array([[-1,-1,(1-np.sqrt(2)),0]]).T
#y0 = np.array([[0,(np.sqrt(2)-1),1,1]]).T
#P = np.c_[x0,y0,np.ones_like(x0)].T
#XSI = np.array([0.,0.,0.,0.5,1.,1.,1.])
#p=2
#M = Matrix_N2ToL2(XSI,p)
#PL = (M.T.dot(P.T)).T
#xB = drawline(XSI,2,P,20,rational=False)
#plt.figure()
#plt.plot(P[0],P[1],'o',label='B spline ctrl pts',markersize=8)
#plt.plot(PL[0],PL[1],'o',label='Lagrange ctrl pts',markersize=6)
#plt.plot(xB[0],xB[1])
#plt.legend()




#%%
def drawline(Xsi,p,B,neval,rational=True):
    """Draws a 1D NURBS curve with neval evaluation points. Xsi knot vector. p degree. B control points ((d+1)*nb_ctrlpts)."""
    xsieval = np.linspace(Xsi[0],Xsi[-1],neval)
    xeval = np.zeros((B.shape[0]-1,neval))
    IEN = nubsconnect(p,B.shape[1])
    for ii in range(neval): # boucle sur les points d'évaluation
        elt = findKnotSpan(xsieval[ii],Xsi,p)
        n_fct = np.flip(IEN[:,elt-p],axis=0)
        Ni_xsi = derbasisfuns(elt,p,Xsi,0,xsieval[ii])
        if rational: # si NURBS avec poids différents de 1
            denom = np.sum(Ni_xsi*B[-1,n_fct])
            Ni_xsi = (Ni_xsi*B[-1,n_fct])/denom
        for jj in range(IEN.shape[0]):
            xeval[:,ii] += Ni_xsi[0,jj]*B[:2,n_fct[jj]]
    return xeval






    
    
### Test
#print((1+1/np.sqrt(2))/2)
#B = np.array([[0.,0.,1.],[0.,1.,1.],[1.,1.,1.]])
#B2 = np.array([[0.,0.,1.],[0.,1.,1.],[1.,np.sqrt(2)/2,1.]])
#Xsi = np.array([0.,0.,0.,1.,1.,1.])
#p = 2
#neval = 20
#XX = drawline(Xsi,p,B,neval,rational=True)
#XX2 = drawline(Xsi,p,B2,neval,rational=True)   
#import matplotlib.pyplot as plt
#plt.figure()
#plt.plot(XX[0],XX[1])
#plt.plot(XX2[0],XX2[1])
#plt.plot(B[0],B[1])
#plt.axis('equal')
#B2homogene = B2.copy()
#B2homogene[:-1,:]=B2homogene[:-1,:]*B2homogene[-1,:]
#Ctl,xi=bspkntins(p,B2homogene,Xsi,np.array([0.5]))
#Ctl[:-1,:]=Ctl[:-1,:]/Ctl[-1,:]
#XX3 = drawline(xi,p,Ctl,neval,rational=True)   
#plt.plot(XX3[0],XX3[1],'x')

