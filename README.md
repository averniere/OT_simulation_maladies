# Transport optimal pour associer des maladies

Ce dépôt contient le travail de stage de M2 autour de la construction d'une fonction de coût adaptée $c$ entre maladies, pour résoudre le problème
$$\min_{P\in U(a,b)} \langle P, C\rangle_F$$, où $U(a,b)=\{P\in\mathbb{R}^{n\times m} : P\mathbb{1}_m=a, P^T\mathbb{1}_n=b\}$
. 

## Utilisation du dépôt
Pour cloner le dépôt, faire dans le terminal
```
git clone https://github.com/averniere/OT_simulation_maladies.git
cd OT_simulation_maladies
```
Pour installer les *packages*, faire dans le terminal
```
pip install -r requirements.txt
```
## Organisation du dépôt 

Le dépôt s'articule autour des dossiers suivant :
1. `data`, contenant certaines bases de données (publiques) sur lesquelles repose le code ;
2. `Disease_Embeddings`, où l'on essaye de mettre en œuvre l'apprentissage d'une représentation des maladies dans le disque de Poincaré d'après la méthode de [Klimovskaia (2020)](https://www.nature.com/articles/s41467-020-16822-4) ;
3. `Embeddings`, où l'on apprend et l'on utilise une représentation de l'ontologie HPO dans le disque de Poincaré selon la méthode de [Nickel et Kiela (2017)](https://arxiv.org/pdf/1705.08039) ;
4. `HGCN`, où l'on apprend et l'on utilise une représentation de l'ontologie HPO dans le disque de Poincaré selon la méthode de [Chami et al. (2019)](https://arxiv.org/pdf/1910.12933) ; 
5. `OT`, qui contient le processus de simulations de maladies complexes.

## Organisation des fichiers
Dossier `Embeddings`:
- `batched_dataset.py`, `lorentz.py`, `model.py`, `main.py`, `poincare.py`, `RSGD.py`, `train.py` sont utiles à l'apprentissage de la représentation de l'ontologie dans le disque de Poincaré.
- `data.py` permet de charger les tables d'annotations de maladies, les interactions PPI, gène-maladie, gène-protéine. `statistiques_descriptives.ipynb` est un notebook présentant les premières statistiques descriptives relatives aux données chargées.
- `data_utils.py`, `OT_utils.py`, `information_content.py`, `tsw.py` contiennent les fonctions utilisées pour calculer les différentes fonctions de coût et plans de transport. Les notebooks `test.ipynb` et `brouillon.ipynb` contiennent les résultats.
- `evaluation.py` contient les fonctions utilisées pour prédire ou non une bonne association (encore en exploration).
- `simulations_add_metrics.py`, `simulations_process.py`, `simulations.py` et le dossier `simuls` reprennent le cadre de simulation de maladies complexes à partir de maladies mendéliennes.

Dossier `HGCN` :
- `decoder.py`, `encoder.py`, `hyp_layer.py`, `main.py`, `model.py`, `poincare.py`, `RiemAdam.py`, `RSGD.py` et `train.py` sont utiles à l'apprentissage de la représentation de l'ontologie dans le disque de Poincaré.
- 

## Où trouver les fonctions Python utiles

Dans le dossier `Embeddings`:
- `information_content.py` : 
    - *compute_information_content* : calculer l'IC des termes HPO à partir d'une base de données de maladies.
- `data_utils.py` :
    - *f_ground_truth* : retourne les couples (i, j) d'indices de maladies OMIM-Orphanet correspondantes.
- `OT_utils.py` : 
    - *basic_cost_matrix* : calculer la matrice de coût associée aux dissimilarités de Jaccard, Hamming ou à la distance euclidienne.
    - *compute_cost_matrix* : calculer la matrice de coût entre barycentres dans le disque de Poincaré.
    - *cost_matrix_hamm* : calculer la matrice de coût associée à la dissimilarité de Hamming avec une pondération quelconque. 
    - *compute_cost_matrix_pseudo_jacc* : calculer la matrice de coût associée à la dissimilarité de Hamming, pondérée par la distance à l'origine des termes HPO dans le disque de Poincaré.
    - *compute_cost_wasserstein2* : calculer la matrice de coût associée aux distances de Wasserstein entre maladies dans le disque de Poincaré. 
    - *compute_transport* : retourne le plan de transport optimal et le coût optimal d'un problème de transport 
- `tsw.py` : 
    - *propagate_terms* : construire une base d'annotations avec propagation ancestrale jusqu'à un niveau souhaité $k$.
