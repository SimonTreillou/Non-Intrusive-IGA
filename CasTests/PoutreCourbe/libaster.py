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


def Rigidite_Glob():
	DEBUT(PAR_LOT='NON',IMPR_MACRO = 'NON')

	### Lecture du maillage
	asMesh = LIRE_MAILLAGE(FORMAT = 'MED',
			       UNITE = 20,						# Unité logique du fichier de maillage
			       NOM_MED = 'global',					# Nom du maillage au sein du fichier
			       INFO = 1,
			       )

	# Nombre de noeuds physiques du maillage
	nbNoeud = asMesh.sdj.DIME.get()[0]
	dim = 2 # Dimension du problème

	### Affectation des modèles
	mod = AFFE_MODELE(MAILLAGE = asMesh,
			  AFFE = (
				  _F(TOUT = 'OUI',				# Modèle de la structure
				    PHENOMENE = 'MECANIQUE',
				    MODELISATION = 'C_PLAN',
				    ),
				  )
				  )

	### Définition des matériaux
	mat = DEFI_MATERIAU(ELAS= _F(E = 100000.,
					NU = 0.3,
				       ),			
					)

	### Affectation des matériaux
	Mat  = AFFE_MATERIAU(MAILLAGE = asMesh,
			     AFFE = (_F(TOUT = 'OUI',
				       MATER = mat,
				       ),
				     ),
				     )

	### Affectation des conditions limites
	# Encastrement
	# GROUP_MA => group of edges
	Fix = AFFE_CHAR_CINE(MODELE = mod,
			     MECA_IMPO = (_F(GROUP_MA = 'Wd',
					    DX = 0., DY = 0.
					    ),
					  )
			     )
	# Effort imposé
	Fd = AFFE_CHAR_MECA(MODELE = mod,
			    FORCE_CONTOUR= _F(GROUP_MA='Fd',FX=0,FY=10),
			    #FORCE_ARETE = _F(GROUP_MA='Fd',FX=0,FY=10),
		           #PRES_REP = _F(GROUP_MA='Fd',PRES=10),
		           )

	# Calcul des matrices de rigidité élémentaires
	matElem = CALC_MATR_ELEM(OPTION='RIGI_MECA', MODELE=mod, CHAM_MATER=Mat)

	# Calcul de la numérotation
	numDDL = NUME_DDL(MATR_RIGI=matElem, );

	# Assemblage de la matrice de rigidité
	matAss = ASSE_MATRICE(MATR_ELEM=matElem, NUME_DDL=numDDL, CHAR_CINE=Fix)

	# Calcul du second membre lié aux CL de Dirichlet
	vcine = CALC_CHAR_CINE(NUME_DDL=numDDL, CHAR_CINE=Fix,);

	# Calcul du second membre lié aux CL de Neumann
	vecElem = CALC_VECT_ELEM(OPTION='CHAR_MECA',CHARGE=Fd,CHAM_MATER=Mat)
	vneum = ASSE_VECTEUR(VECT_ELEM=vecElem,NUME_DDL=numDDL)

	# Factorisation de la matrice de rigidité et prise en compte des CL de 
	# Dirichlet éliminées
	matAss = FACTORISER(reuse=matAss,MATR_ASSE=matAss, METHODE='MUMPS',);

	# Résolution du problèm Ku=F
	sol = RESOUDRE(MATR=matAss, CHAM_NO=vneum, CHAM_CINE=vcine,)

	# Création du concept de résultat
	Res = CREA_RESU(OPERATION = 'AFFE',
		        TYPE_RESU = 'EVOL_ELAS',
		        NOM_CHAM = 'DEPL',
		        AFFE = _F(CHAM_GD = sol,
		                MODELE = mod,
		                CHAM_MATER = Mat,
		                INST = 1.
		                )
		        )

	# Calcul des champs de réactions nodales
	Res = CALC_CHAMP(reuse = Res,
		         RESULTAT = Res,
		         FORCE = 'REAC_NODA',
		         TOUT = 'OUI',
		         )
		         

	# Sauvegarde au format med
	IMPR_RESU(FORMAT = 'MED',UNITE=80,RESU=_F(RESULTAT=Res))


	### Exemple de passage aster->numpy et numpy->aster

	# Conversion de la matrice de rigidité au format sparse
	matAssPython = matAss.EXTR_MATR(sparse='True')
	matAssPython = sp.coo_matrix((matAssPython[0],(matAssPython[1],matAssPython[2])),shape=(2*nbNoeud,2*nbNoeud)).tocsc()

	# Conversion du champ de déplacement en python (ça s'est facile)
	topo = sol.EXTR_COMP(topo=1).noeud # Numéro des noeux associés aux degrés de liberté
	NO = ['N{}'.format(ii) for ii in sorted(list(set(list(topo))))] # Liste de noeuds
	deplPython = sol.EXTR_COMP().valeurs # Valeur du champ au format numpy
	return matAssPython, deplPython
	
	
def Rigidite_Loc():
	DEBUT(PAR_LOT='NON',IMPR_MACRO = 'NON')

	### Lecture du maillage
	asMesh = LIRE_MAILLAGE(FORMAT = 'MED',
			       UNITE = 20,						# Unité logique du fichier de maillage
			       NOM_MED = 'local',					# Nom du maillage au sein du fichier
			       INFO = 1,
			       )

	# Nombre de noeuds physiques du maillage
	nbNoeud = asMesh.sdj.DIME.get()[0]
	dim = 2 # Dimension du problème

	### Affectation des modèles
	mod = AFFE_MODELE(MAILLAGE = asMesh,
			  AFFE = (
				  _F(TOUT = 'OUI',				# Modèle de la structure
				    PHENOMENE = 'MECANIQUE',
				    MODELISATION = 'C_PLAN',
				    ),
				  )
				  )

	### Définition des matériaux
	mat = DEFI_MATERIAU(ELAS= _F(E = 100000.,
					NU = 0.3,
				       ),			
					)

	### Affectation des matériaux
	Mat  = AFFE_MATERIAU(MAILLAGE = asMesh,
			     AFFE = (_F(TOUT = 'OUI',
				       MATER = mat,
				       ),
				     ),
				     )

	### Affectation des conditions limites
	# Encastrement
	# GROUP_MA => group of edges
	Fix = AFFE_CHAR_CINE(MODELE = mod,
			     MECA_IMPO = (_F(GROUP_MA = 'Wd',
					    DX = 0., DY = 0.
					    ),
					  )
			     )
	# Effort imposé
	Fd = AFFE_CHAR_MECA(MODELE = mod,
			    FORCE_CONTOUR= _F(GROUP_MA='Fd',FX=0,FY=10),
			    #FORCE_ARETE = _F(GROUP_MA='Fd',FX=0,FY=10),
		           #PRES_REP = _F(GROUP_MA='Fd',PRES=10),
		           )

	# Calcul des matrices de rigidité élémentaires
	matElem = CALC_MATR_ELEM(OPTION='RIGI_MECA', MODELE=mod, CHAM_MATER=Mat)

	# Calcul de la numérotation
	numDDL = NUME_DDL(MATR_RIGI=matElem, );

	# Assemblage de la matrice de rigidité
	matAss = ASSE_MATRICE(MATR_ELEM=matElem, NUME_DDL=numDDL, CHAR_CINE=Fix)

	# Calcul du second membre lié aux CL de Dirichlet
	vcine = CALC_CHAR_CINE(NUME_DDL=numDDL, CHAR_CINE=Fix,);

	# Calcul du second membre lié aux CL de Neumann
	vecElem = CALC_VECT_ELEM(OPTION='CHAR_MECA',CHARGE=Fd,CHAM_MATER=Mat)
	vneum = ASSE_VECTEUR(VECT_ELEM=vecElem,NUME_DDL=numDDL)

	# Factorisation de la matrice de rigidité et prise en compte des CL de 
	# Dirichlet éliminées
	matAss = FACTORISER(reuse=matAss,MATR_ASSE=matAss, METHODE='MUMPS',);

	# Résolution du problèm Ku=F
	sol = RESOUDRE(MATR=matAss, CHAM_NO=vneum, CHAM_CINE=vcine,)

	# Création du concept de résultat
	Res = CREA_RESU(OPERATION = 'AFFE',
		        TYPE_RESU = 'EVOL_ELAS',
		        NOM_CHAM = 'DEPL',
		        AFFE = _F(CHAM_GD = sol,
		                MODELE = mod,
		                CHAM_MATER = Mat,
		                INST = 1.
		                )
		        )

	# Calcul des champs de réactions nodales
	Res = CALC_CHAMP(reuse = Res,
		         RESULTAT = Res,
		         FORCE = 'REAC_NODA',
		         TOUT = 'OUI',
		         )
		         

	# Sauvegarde au format med
	IMPR_RESU(FORMAT = 'MED',UNITE=80,RESU=_F(RESULTAT=Res))


	### Exemple de passage aster->numpy et numpy->aster

	# Conversion de la matrice de rigidité au format sparse
	matAssPython = matAss.EXTR_MATR(sparse='True')
	matAssPython = sp.coo_matrix((matAssPython[0],(matAssPython[1],matAssPython[2])),shape=(2*nbNoeud,2*nbNoeud)).tocsc()

	# Conversion du champ de déplacement en python (ça s'est facile)
	topo = sol.EXTR_COMP(topo=1).noeud # Numéro des noeux associés aux degrés de liberté
	NO = ['N{}'.format(ii) for ii in sorted(list(set(list(topo))))] # Liste de noeuds
	deplPython = sol.EXTR_COMP().valeurs # Valeur du champ au format numpy
	return matAssPython, deplPython
	
