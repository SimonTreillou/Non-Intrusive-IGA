def affeField(field,typeCL,Struc,param):
    '''
    cham : champ Aster ou champ de classe ChamPython
    typeCL : type de CL en string (D ou F)
    Struc : Classe Structure sur laquelle affecter le champ python
    param : Classe des parametres
    Retourne un char_meca associe a l'affectation du cham aster chamPython
    '''
    global AsterDim, m, AsterCount, memoryLocation
    ### Si cham est de type chamPython
    if isinstance(field,asterField):
        fieldPython = field.convert2python()					# Transformation en champ Aster
    else:
        fieldPython = field
    ### Conditions limites en déplacement

    if typeCL == 'D':
        if param.dim == 2:

            ddlImpo = [_F(NOEUD=fieldPython.nodes[ii],DX=fieldPython.valField[2*ii],DY=fieldPython.valField[2*ii+1]) for ii in range(len(fieldPython.nodes))]
        elif param.dim == 3:
            ddlImpo = [_F(NOEUD=fieldPython.nodes[ii],DX=fieldPython.valField[3*ii],DY=fieldPython.valField[3*ii+1],DZ=fieldPython.valField[3*ii+2]) for ii in range(len(fieldPython.nodes))]

        AsterIter = AsterCount.next()
        memoryLocation.append(AsterIter)
        m[AsterIter] = AFFE_CHAR_MECA(MODELE = Struc.mod.MODELE,DDL_IMPO = ddlImpo)

        return m[AsterIter]
    ### Conditions limites en effot
    elif typeCL == 'F':
        if param.dim == 2:
            forceNodale = [_F(NOEUD=fieldPython.nodes[ii],FX=fieldPython.valField[2*ii],FY=fieldPython.valField[2*ii+1]) for ii in range(len(fieldPython.nodes))]
        elif param.dim == 3:
            forceNodale = [_F(NOEUD=fieldPython.nodes[ii],FX=fieldPython.valField[3*ii],FY=fieldPython.valField[3*ii+1],FZ=fieldPython.valField[3*ii+2]) for ii in range(len(fieldPython.nodes))]
        AsterIter = AsterCount.next()
        memoryLocation.append(AsterIter)
        m[AsterIter] = AFFE_CHAR_MECA(MODELE = Struc.mod.MODELE,FORCE_NODALE = forceNodale)
        return m[AsterIter]
