# Transport optimal pour associer des maladies

Ce dépôt contient le travail de stage de M2 autour de la construction d'une fonction de coût adaptée $c$ entre maladies. 

## Organisation du dépôt 

Le dépôt s'articule autour des fichiers suivant :
- `data`, contenant certaines bases de données (publiques) sur lesquelles repose le code ;
- `Disease_Embeddings`, où l'on essaye de mettre en œuvre l'apprentissage d'une représentation des maladies dans le disque de Poincaré d'après la méthode de >Klimovskaia (2020);
- `Embeddings`, où l'on apprend et l'on utilise une représentation de l'ontologie HPO dans le disque de Poincaré selon la méthode de >Nickel et Kiela (2017) ;
- `HGCN`, où l'on apprend et l'on utilise une représentation de l'ontologie HPO dans le disque de Poincaré selon la méthode de >Chami et al. (2019) ; 
- `OT`, qui contient le processus de simulations de maladies complexes.
