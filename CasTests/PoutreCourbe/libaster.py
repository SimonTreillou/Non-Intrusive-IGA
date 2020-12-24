#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from numpy import linalg
import scipy.sparse as sp
from code_aster.Cata.Legacy.Syntax import _F
from code_aster.Cata.Commands import CREA_RESU
from code_aster.Cata.Commands import LIRE_MAILLAGE
from code_aster.Cata.Commands import AFFE_MODELE
from code_aster.Cata.Commands import DEBUT
from code_aster.Cata.Commands import DEFI_MATERIAU
from code_aster.Cata.Commands import AFFE_MATERIAU
from code_aster.Cata.Commands import AFFE_CHAR_CINE
from code_aster.Cata.Commands import AFFE_CHAR_MECA
from code_aster.Cata.Commands import CALC_MATR_ELEM
from code_aster.Cata.Commands import NUME_DDL
from code_aster.Cata.Commands import ASSE_MATRICE
from code_aster.Cata.Commands import CALC_CHAR_CINE
from code_aster.Cata.Commands import CALC_VECT_ELEM
from code_aster.Cata.Commands import ASSE_VECTEUR
from code_aster.Cata.Commands import FACTORISER
from code_aster.Cata.Commands import RESOUDRE
from code_aster.Cata.Commands import CALC_CHAMP
from code_aster.Cata.Commands import IMPR_RESU
from code_aster.Cata.Commands import DETRUIRE
from code_aster.Cata.Commands import CREA_CHAMP
#import reac_noda as rn

AsterDim = 100
o = [None]*AsterDim
AsterCount = iter(range(AsterDim))


o2 = [None]*AsterDim
AsterCount2 = iter(range(AsterDim))


def Global(E=100000.,Nu=0.3,fx=0.,fy=100.):

	### Lecture du maillage
	asMeshG = LIRE_MAILLAGE(FORMAT = 'MED', 
			       UNITE = 20,						# Unité logique du fichier de maillage
			       NOM_MED = 'global',					# Nom du maillage au sein du fichier
			       INFO = 1,
			       )

	# Nombre de noeuds physiques du maillage
	nbNoeudG = asMeshG.sdj.DIME.get()[0]
	dimG = 2 # Dimension du problème

	### Affectation des modèles
	modG = AFFE_MODELE(MAILLAGE = asMeshG,
			  AFFE = (
				  _F(TOUT = 'OUI',				# Modèle de la structure
				    PHENOMENE = 'MECANIQUE',
				    MODELISATION = 'C_PLAN',
				    ),
				  )
				  )

	### Définition des matériaux
	matG = DEFI_MATERIAU(ELAS= _F(E = E,
					NU = Nu,
				       ),			
					)

	### Affectation des matériaux
	MatG  = AFFE_MATERIAU(MAILLAGE = asMeshG,
			     AFFE = (_F(TOUT = 'OUI',
				       MATER = matG,
				       ),
				     ),
				     )

	### Affectation des conditions limites
	# Encastrement
	# GROUP_MA => group of edges
	FixG = AFFE_CHAR_CINE(MODELE = modG,
			     MECA_IMPO = (_F(GROUP_MA = 'Wd',
					    DX = 0., DY = 0.
					    ),
					  )
			     )
	# Effort imposé
	FdG = AFFE_CHAR_MECA(MODELE = modG,
			    FORCE_CONTOUR= _F(GROUP_MA='Fd',FX=fx,FY=fy),
			    #FORCE_ARETE = _F(GROUP_MA='Fd',FX=0,FY=10),
		           #PRES_REP = _F(GROUP_MA='Fd',PRES=10),
		           )

	# Calcul des matrices de rigidité élémentaires
	matElemG = CALC_MATR_ELEM(OPTION='RIGI_MECA', MODELE=modG, CHAM_MATER=MatG)

	# Calcul de la numérotation
	numDDLG = NUME_DDL(MATR_RIGI=matElemG, );

	# Assemblage de la matrice de rigidité
	matAssG = ASSE_MATRICE(MATR_ELEM=matElemG, NUME_DDL=numDDLG, CHAR_CINE=FixG)

	# Calcul du second membre lié aux CL de Dirichlet
	vcineG = CALC_CHAR_CINE(NUME_DDL=numDDLG, CHAR_CINE=FixG,);

	# Calcul du second membre lié aux CL de Neumann
	vecElemG = CALC_VECT_ELEM(OPTION='CHAR_MECA',CHARGE=FdG,CHAM_MATER=MatG)
	vneumG = ASSE_VECTEUR(VECT_ELEM=vecElemG,NUME_DDL=numDDLG)

	# Factorisation de la matrice de rigidité et prise en compte des CL de 
	# Dirichlet éliminées
	matAssG = FACTORISER(reuse=matAssG,MATR_ASSE=matAssG, METHODE='MUMPS',);

	return matAssG, vcineG, vneumG, MatG, modG, numDDLG  
	
	
def Local(E=100000.,Nu=0.3,fx=0.,fy=0.):
	### Lecture du maillage
	asMeshL = LIRE_MAILLAGE(FORMAT = 'MED',
		   UNITE = 22,                           # Unité logique du fichier de maillage
		   NOM_MED = 'fish',                    # Nom du maillage au sein du fichier
		   INFO = 1,
		   )
	# Nombre de noeuds physiques du maillage
	nbNoeudL = asMeshL.sdj.DIME.get()[0]
	dim = 2 # Dimension du problème

	### Affectation des modèles
	modL = AFFE_MODELE(MAILLAGE = asMeshL,
	      AFFE = (
		  _F(TOUT = 'OUI',                # Modèle de la structure
		    PHENOMENE = 'MECANIQUE',
		    MODELISATION = 'C_PLAN',
		    ),
		  )
		  )

	### Définition des matériaux
	matL = DEFI_MATERIAU(ELAS= _F(E = E,
		    NU = Nu,
		       ),            
		    )

	### Affectation des matériaux
	MatL  = AFFE_MATERIAU(MAILLAGE = asMeshL,
		 AFFE = (_F(TOUT = 'OUI',
		       MATER = matL,
		       ),
		     ),
		     )

	# Effort imposé
	FdL = AFFE_CHAR_MECA(MODELE = modL,
		 FORCE_CONTOUR= _F(GROUP_MA='Wd',FX=fx,FY=fy),
		    )

	# Char_cine nul
	blank = AFFE_CHAR_CINE(MODELE=modL,
				MECA_IMPO=_F(GROUP_MA='Wd',DX=0.,DY=0.,)
				)
					     
	# Calcul des matrices de rigidité élémentaires
	matElemL = CALC_MATR_ELEM(OPTION='RIGI_MECA', MODELE=modL, CHAM_MATER=MatL)

	# Calcul de la numérotation
	numDDLL = NUME_DDL(MATR_RIGI=matElemL);

	matAssL = ASSE_MATRICE(MATR_ELEM=matElemL, NUME_DDL=numDDLL, CHAR_CINE=blank,)
	# Calcul du second membre force
	vecElemL = CALC_VECT_ELEM(OPTION='CHAR_MECA',CHARGE=FdL,CHAM_MATER=MatL)
	vneumL = ASSE_VECTEUR(VECT_ELEM=vecElemL,NUME_DDL=numDDLL)

	# Calcul du second membre cine nul
	vblank = CALC_CHAR_CINE(NUME_DDL=numDDLL,CHAR_CINE=blank,);
	
	# Assemblage de la matrice de rigidité
	
	matAssL = FACTORISER(reuse=matAssL,MATR_ASSE=matAssL, METHODE='MUMPS',);
	
	return matAssL, vneumL, MatL, modL, numDDLL

def create_resu(field,model,mat,char_cine=None):
    """
    Create an aster concept of results from an aster field obtained by a solve.
    Input:
        -field: aster displacement field
        -model: aster model
        -mat: aster assigned material
        -char_cine: dirichlet boundary conditions
    Output:
        -aster result
    """
    
    if char_cine:
        o2[AsterIter2] = CREA_RESU(OPERATION = 'AFFE',
                         TYPE_RESU = 'EVOL_ELAS',
                         NOM_CHAM = 'DEPL',
                         EXCIT = _F(CHARGE = char_cine),
                         AFFE = _F(CHAM_GD = field,
                                   MODELE = model,
                                   CHAM_MATER = mat,
                                   INST = 0.,)
                         )
    else:
        o2[AsterIter2] = CREA_RESU(OPERATION = 'AFFE',
                     TYPE_RESU = 'EVOL_ELAS',
                     NOM_CHAM = 'DEPL',
                     AFFE = _F(CHAM_GD = field,
                               MODELE = model,
                               CHAM_MATER = mat,
                               INST = 0.,)
                    )

    return resu


def compute_nodal_reaction_from_field_on_group(field,model,mat,group,charg,char_cine=None):
    """
    Compute nodal reaction from a displacement field on a specific group.
    Input:
        -field: aster displacement field
        -model: aster model,
        -mat: assigned material on a mesh
        -char_cine: dirichlet boundary conditions
        -group: group where the nodal reaction has to be computed
    Output:
        -asterField instance
    """
    ### Create result concept from the displacement field
    #resu = create_resu(field,model,mat,char_cine)

    if char_cine:
        resu =CREA_RESU(OPERATION = 'AFFE',
                         TYPE_RESU = 'EVOL_ELAS',
                         NOM_CHAM = 'DEPL',
                         EXCIT = _F(CHARGE = charg),
                         AFFE = _F(CHAM_GD = field,
                                   MODELE = model,
                                   CHAM_MATER = mat,
                                   INST = 0.,)
                         )
    else:
        resu= CREA_RESU(OPERATION = 'AFFE',
                     TYPE_RESU = 'EVOL_ELAS',
                     NOM_CHAM = 'DEPL',
                     AFFE = _F(CHAM_GD = field,
                               MODELE = model,
                               CHAM_MATER = mat,
                               INST = 0.,)
                    )
    
    resu = CALC_CHAMP(FORCE = 'REAC_NODA',
                     reuse = resu,
                     MODELE=model,
                     CHAM_MATER=mat,
                     EXCIT=_F(CHARGE=charg),
                     TOUT='OUI',
                     RESULTAT = resu,
                    )
    AsterIter = AsterCount.__next__()
    o[AsterIter] = CREA_CHAMP(OPERATION = 'EXTR',
                                NOM_CHAM = 'REAC_NODA',
                                TYPE_CHAM = 'NOEU_DEPL_R',
                                RESULTAT = resu,
                                INST=0.,
                                )
    nodalrea = CREA_CHAMP(OPERATION = 'ASSE',
                                TYPE_CHAM = 'NOEU_DEPL_R',
                                MODELE = model,
                                ASSE = _F(CHAM_GD = o[AsterIter] ,GROUP_MA = group),
                                )
    
    DETRUIRE(CONCEPT=(_F(NOM=nodalrea),_F(NOM=resu)))
    return o[AsterIter]
	
