# Multi-scale and non-invasive multi-model coupling combining Iso-Geometric and Finite Element Analysis

This repository contains the code implemented during our 5th year research project at INSA Toulouse.

## Project description

Modelling and simulating mechanical structures is now a very common mathematical and engineering practice, used in various domains such as aeronautics. It allows industrial engineers to study their products and mechanical behaviors without needs for physical tests that may appear very expensive. Nevertheless, modelling different scales leads to a major issue: using the finest scale in the whole mesh carries high computational cost and memory usage. Moreover, with different models and scales, we cannot modify the whole meshes and operators at each modification. That is why we present in this paper a review and implementation of the **global/local non-invasive multi-model and multi-scale coupling algorithm** using the industrial software **Code_Aster** from EDF focusing on mechanical structures. This method allows us to introduce local models, representing singularities or refined meshes, in global models without modifying them, and with no regards about their implementation. We then investigate a **global-IGA/local-FEM coupling algorithm** and compare results.  Combining IGA and FEM gives the best out of these two methods: a better description of global and regular solutions for IGA, and a better simulation of singularities for FEM. Our results show great benefits and fully describe the mechanical behaviors of our pieces.  More attention could be spent on algorithmic acceleration, non-linear coupling or non-elastic behaviors.

## Code_Aster

Code_Aster is an industrial software for mechanical structures developed by EDF. You can download it at:

https://www.code-aster.org/spip.php?rubrique7

## Salome-Meca 

Salome-Meca is a free software including a CAD module, Code_Aster and post-processing module. It allows to manipulate .med files, for results visualization or meshes manipulation. You can download it at :

https://www.code-aster.org/spip.php?article295

## Test cases 

Two main test cases are implemented. An elastic and simple girder with only three elements, with two different Young modulus. An elastic curved girder with added holes and cracks. 

## How to use our code

Once Code_Aster installated, you have to:
1. change path variable to your current working directory in ```global.comm```
2. create ```.export``` file taking as an example the file ```CasTests/PoutreCourbe/global-temp.export```
3. in a terminal, run ```bash as_run your-export-file.export```

In order to visualize your results, you can use Paravis in Salome-Meca with results files ```resultG.med``` and ```resultL.med``` for global and local solutions.

In order to plot the convergence curves, do:
```bash
python3 plot-convergence.py arg1 arg2
```
with ```arg1: girder [DEFAULT], toy``` and ```arg2: residue [DEFAULT], error```. 
```arg1``` allows you to choose which case you want to display, ```girder``` for the elastic curved girder or ```toy``` for the simple toy case.
```arg2``` allows you to choose which type of convergence you want to display, ```residue``` for the residual of forces on the interface, ```error``` for the stagnation of the displacement field.

## Bug fix

If there is any problem, do not hesitate to open an issue or contact us: [SimonTreil](https://github.com/SimonTreil) or [paul401](https://github.com/paul401).

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
