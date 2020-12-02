from code_aster.Cata.Legacy.Syntax import _F
from code_aster.Cata.Commands import CREA_RESU

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
        resu = CREA_RESU(OPERATION = 'AFFE',
                         TYPE_RESU = 'EVOL_ELAS',
                         NOM_CHAM = 'DEPL',
                         EXCIT = _F(CHARGE = char_cine),
                         AFFE = _F(CHAM_GD = field,
                                   MODELE = model,
                                   CHAM_MATER = mat,
                                   INST = 0.,)
                         )
    else:
        resu = CREA_RESU(OPERATION = 'AFFE',
                     TYPE_RESU = 'EVOL_ELAS',
                     NOM_CHAM = 'DEPL',
                     AFFE = _F(CHAM_GD = field,
                               MODELE = model,
                               CHAM_MATER = mat,
                               INST = 0.,)
                    )

    return resu


def compute_nodal_reaction_from_field_on_group(field,model,mat,group,char_cine=None):
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
    resu = create_resu(field,model,mat,char_cine)
    resu = CALC_CHAMP(reuse = resu,
                      FORCE = 'REAC_NODA',
                      GROUP_MA = group,
                      RESULTAT = resu,
                      INST=0.)
    toto = CREA_CHAMP(OPERATION = 'EXTR',
                                NOM_CHAM = 'REAC_NODA',
                                TYPE_CHAM = 'NOEU_DEPL_R',
                                RESULTAT = resu,
                                INST=0.
                                )
    nodal_reaction = CREA_CHAMP(OPERATION = 'ASSE',
                                TYPE_CHAM = 'NOEU_DEPL_R',
                                MODELE = model,
                                ASSE = _F(CHAM_GD = toto,GROUP_MA = group),
                                )

    return toto.EXTR_COMP().valeurs
