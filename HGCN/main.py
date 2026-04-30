import pandas as pd
import os
import networkx as nx
import ast
import numpy as np
import datetime

from dataclasses import dataclass
from train import train


profils_omim = pd.read_csv("../data/profils_omim.csv.gz", index_col=0)
profils_omim = profils_omim.reset_index()

#current_dir = os.path.dirname(os.path.abspath(__file__))
#data_path = os.path.join(current_dir, "..", "data", "HPOs.csv")

df_hpo = pd.read_csv("../data/HPOs.csv", sep=";")
df_hpo = df_hpo.drop(columns = ["definition", "synonyms"])
df_hpo['parents'] = df_hpo['parents'].apply(ast.literal_eval)

G_hpo = nx.DiGraph()

for i, row in df_hpo.iterrows():
    G_hpo.add_node(row['hp_id'])

for j, row in df_hpo.iterrows():
    for parent_id in row["parents"]:
        if parent_id in G_hpo:
            G_hpo.add_edge(row['hp_id'], parent_id)  # Permet d'avoir la racine au centre

@dataclass
class Args:
    # Entraînement
    val_prop: float = 0.05
    test_prop: float = 0.10
    split_seed: int = 42
    seed : int = 42
    normalize_adj: bool = True
    normalize_feats: bool = True
    epochs : int = 500
    min_epochs : int =100
    patience : int = 50
    lr : float = 0.1
    lr_reduce_freq : int = None
    weight_decay : float = 0.
    gamma : float = 0.5  # Par combien multiplier le learning_rate si on veut le décroitre avec lr_scheduler
    grad_clip : float = None
    eval_freq : int = 1
    save : bool = True
    # Modèle
    dropout : float = 0.
    c : float = 1.  # Rayon hyperbolique --> essayer avec None pour apprendre la courbure également ?
    r : float = 2.  # Paramètre du decoder
    t : float = 1.  # Paramètre du decoder
    bias : bool = True  # Utiliser un biais
    use_att : bool = False 
    local_agg : bool = True  # Local aggregation
    act : str = 'relu'  # Fonction d'activation
    num_layers : int = 2  # Nb of hidden layers
    dim : int = 10  # Dimension de l'embedding
    optimizer : str = 'RSGD'  # ou 'Adam' (ne marche pas pour le moment)

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
models_dir = os.path.join("logs/", date)
save_dir = get_dir_name(models_dir)

train(args, G_hpo, profils_omim, save_dir)