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

Schéma des principaux dossiers : 
```
|--data
|   |--utils
|--Embeddings
|   |--simuls
|   |   |--analyse_resultats.ipynb
|   |--__init__.py
|   |--analyse.ipynb
|   |--batched_dataset.py
|   |--data_utils.py
|   |--data.py
|   |--evaluation.py
|   |--frechetmean.py
|   |--frlc0.py
|   |--information_content.py
|   |--lorentz.py
|   |--main.py
|   |--model.py
|   |--OT.py
|   |--poincare.py
|   |--RSGD.py
|   |--simulations_add_metrics.py
|   |--simulations_process.py
|   |--simulations.py
|   |--statistiques_descriptives.ipynb
|   |--test.ipynb
|   |--train.py
|   |--tsw_node2vec.ipynb
|   |--tsw.py
|--HGCN
|   |--__init__.py
|   |--data_utils.py
|   |--data.py
|   |--decoder.py
|   |--encoder.py
|   |--evaluation.py
|   |--frechetmean.py
|   |--hyp_layer.py
|   |--load_data.py
|   |--main.py
|   |--model.py
|   |--OT.py
|   |--poincare.py
|   |--RiemAdam.py
|   |--RSGD.py
|   |--similarities.py
|   |--simulations.py
|   |--test.ipynb
|   |--train.py
|--.gitignore
|--README.md
|--requirements.txt

```

## Organisation des dossiers
Dossier `Embeddings`:
- `batched_dataset.py`, `lorentz.py`, `model.py`, `main.py`, `poincare.py`, `RSGD.py`, `train.py` sont utiles à l'apprentissage de la représentation de l'ontologie dans le disque de Poincaré.
- `data.py` permet de charger les tables d'annotations de maladies, les interactions PPI, gène-maladie, gène-protéine. `statistiques_descriptives.ipynb` est un notebook présentant les premières statistiques descriptives relatives aux données chargées.
- `data_utils.py`, `OT.py`, `information_content.py`, `tsw.py` contiennent les fonctions utilisées pour calculer les différentes fonctions de coût et plans de transport. Les notebooks `test.ipynb` et `tsw_node2vec.ipynb` contiennent les résultats.
- `evaluation.py` contient les fonctions utilisées pour prédire ou non une bonne association (encore en exploration).
- `simulations_add_metrics.py`, `simulations_process.py`, `simulations.py` et le dossier `simuls` reprennent le cadre de simulation de maladies complexes à partir de maladies mendéliennes.

Dossier `HGCN` :
- `decoder.py`, `encoder.py`, `hyp_layer.py`, `load_data.py`, `main.py`, `model.py`, `poincare.py`, `RiemAdam.py`, `RSGD.py`(inutilisé finalement) et `train.py` sont utiles à l'apprentissage de la représentation de l'ontologie dans le disque de Poincaré.
- `data.py` permet de charger les tables d'annotations de maladies, les interactions PPI, gène-maladie, gène-protéine.
- `data_utils.py`, `OT.py` contiennent les fonctions utilisées pour calculer les différentes fonctions de coût et plans de transport. Le notebook `test.ipynb` contient les résultats.
- `evaluation.py` contient les fonctions utilisées pour prédire ou non une bonne association (encore en exploration).