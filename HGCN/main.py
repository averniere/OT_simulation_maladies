import pandas as pd
import os
import networkx as nx
import numpy as np
import datetime

from dataclasses import dataclass
from load_data import *
from train import train
from data import add_edges, add_corresponding_terms, get_ancestors0

union_diseases = add_corresponding_terms(work_omim, work_orpha, df_orpha_omim)
hpo_cols = [c for c in union_diseases.columns if c.startswith('HP')]
G_hpo_omim = add_edges(union_diseases, G_hpo_work, depths)
# Cas particulier : on relie manuellement le noeud à son parent dans le sens enfant-parent
G_hpo_work.add_edge('HP:6001347', 'HP:0001832')
G_hpo_omim.add_edge('HP:6001347', 'HP:0001832')

ancestors = {hp: get_ancestors0(G_hpo_work, hp) for hp in hpo_cols}

G_hpo_rev = G_hpo_work.reverse()
G_hpo_omim_rev = G_hpo_omim.reverse()

@dataclass
class Args:
    # Entraînement
    val_prop: float = 0.05
    test_prop: float = 0.10
    split_seed: int = 42
    seed : int = 42
    normalize_adj: bool = True
    normalize_feats: bool = True
    epochs : int = 2500
    min_epochs : int = 100
    patience : int = 200
    lr : float = 1e-2
    lr_reduce_freq : int = None
    weight_decay : float = 0.0
    gamma : float = 0.5  # Par combien multiplier le learning_rate si on veut le décroitre avec lr_scheduler
    grad_clip : float = 5.
    eval_freq : int = 5
    save : bool = True
    # Modèle
    dropout : float = 0.3
    c : float = 1.  # Rayon hyperbolique --> essayer avec None pour apprendre la courbure également ?
    r : float = 2.  # Paramètre du decoder
    t : float = 4.  # Paramètre du decoder
    bias : bool = True  # Utiliser un biais
    use_att : bool = False  # Inutilisable en l'état : problème de mémoire
    local_agg : bool = False  # Local aggregation
    act : str = 'relu'  # Fonction d'activation
    num_layers : int = 2  # Nb of hidden layers
    dim : int = 20  # Dimension de l'embedding
    optimizer : str = 'Adam'  # ou 'Adam' (ne marche pas pour le moment)

args = Args()

def get_dir_name(models_dir):
    """Gets a directory to save the model.

    If the directory already exists, then append a new integer to the end of
    it. This method is useful so that we don't overwrite existing models
    when launching new jobs.

    Args:
        models_dir: The directory where all the models are.

    Returns:
        The name of a new directory to save the training logs and model weights.
    """
    if not os.path.exists(models_dir):
        save_dir = os.path.join(models_dir, '0')
        os.makedirs(save_dir)
    else:
        existing_dirs = np.array(
                [
                    d
                    for d in os.listdir(models_dir)
                    if os.path.isdir(os.path.join(models_dir, d))
                    ]
        ).astype(int)
        if len(existing_dirs) > 0:
            dir_id = str(existing_dirs.max() + 1)
        else:
            dir_id = "1"
        save_dir = os.path.join(models_dir, dir_id)
        os.makedirs(save_dir)
    return save_dir


dt = datetime.datetime.now()
date = f"{dt.year}_{dt.month}_{dt.day}"
models_dir = os.path.join("HGCN/logs/", date)
save_dir = get_dir_name(models_dir)

train(args, G_hpo_work, union_diseases, save_dir, ancestors, depths)
