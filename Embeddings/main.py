import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import datetime
import os

from poincare import PoincareManifold
from model import Distance_PE
from RSGD import RiemanianSGD
from batched_dataset import *
from train import train
from data import *
from data_utils import add_corresponding_terms, add_edges
from information_content import *
from OT_utils import *


# TEST : on relie les termes présents dans une même maladie ----------------------------------

union_diseases = add_corresponding_terms(work_omim, work_orpha, df_orpha_omim)
G_hpo_omim = add_edges(union_diseases, G_hpo_work, depths)
# Cas particulier : on relie manuellement le noeud à son parent dans le sens enfant-parent
G_hpo_omim.add_edge('HP:6001347', 'HP:0001832')

# --------------------------------------------------------------------------------------------

objects_omim = list(G_hpo_omim.nodes())
node2id_omim = {n: i for i, n in enumerate(objects_omim)}
edges_omim = np.array([(node2id_omim[u], node2id_omim[v]) for u, v in G_hpo_omim.edges()],dtype=np.int32)
print(f"{len(edges_omim)} arêtes et {len(objects_omim)} noeuds")

objects = list(G_hpo_work.nodes())
node2id = {n: i for i, n in enumerate(objects)}
edges = np.array([(node2id[u], node2id[v]) for u, v in G_hpo_work.edges()],dtype=np.int32)
print(f"{len(edges)} arêtes et {len(objects)} noeuds")

# Voisins dans le graphe raccordé
pos_neighbors = [set() for _ in range(len(objects_omim))]
for u, v in edges_omim:
    pos_neighbors[int(u)].add(int(v))

all_u_pos = []
all_v_pos = []
for u, neighbors in enumerate(pos_neighbors):
    for v in neighbors:
        if v != u:
            all_u_pos.append(u)
            all_v_pos.append(v)

all_u_pos = torch.tensor(all_u_pos, dtype=torch.long)
all_v_pos = torch.tensor(all_v_pos, dtype=torch.long)


# TEST : Fermeture transitive partielle ------------------------------------------------------
def partial_transitive_closure(edges, max_depth=3):
    G = nx.DiGraph()
    G.add_edges_from(edges)
    new_edges = set(map(tuple, edges))
    
    for u in G.nodes():
        # BFS limité à max_depth
        visited = nx.single_source_shortest_path_length(G, u, cutoff=max_depth)
        for v, depth in visited.items():
            if depth > 0:
                new_edges.add((u, v))
    
    return list(new_edges)


# edges_closed = partial_transitive_closure(edges, max_depth=3)
# print(f"Arêtes avant : {len(edges)}, après : {len(edges_closed)}")
# -------------------------------------------------------------------------------------------

DIM = 15
EPOCHS = 1500
LR0 = 0.4
BURN_IN = 100
NNEGS = 50
BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
POS_RATIO = 0  # Pourcentage de pseudo-positifs tirés en plus
P = 1.0
Q = 0.05
WINDOW_SIZE = 10
REFRESH = 50   # Ré-échantillonnage des positifs tous les ...

LR = LR0/32*BATCH_SIZE
print(LR)

print('Modèle')
manifold = PoincareManifold()
model = Distance_PE(
    n=len(objects), 
    dim=DIM, 
    manifold=manifold, 
    sparse=False,  # True à l'origine
    learn_curvature=False, 
    init_curvature=1.,
    weight_decay=0.
    )

optimizer = RiemanianSGD(model.parameters(), lr=LR, manifold=manifold, c=model.c.item())

if model._log_c.requires_grad:
    c_optimizer = torch.optim.Adam([model._log_c], lr=1e-2)
else:
    c_optimizer=None 

print('Données')
data = BatchedDataset(edges, objects, nnegs=NNEGS, batch_size=BATCH_SIZE, pos_neighbors=pos_neighbors, pos_ratio=POS_RATIO)
#data = BatchedDatasetNode2Vec(G_hpo_omim, edges, True, P, Q, BATCH_SIZE, NNEGS, WINDOW_SIZE, REFRESH)
# print('Preprocess (partly) transition probabilities')
# data.preprocess_transition_probs()


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
                [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
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


print("Début de l'entraînement")
losses, norms = train(
    model=model,
    data=data,
    optimizer=optimizer,
    epochs=EPOCHS,
    lr=LR,
    device=DEVICE,
    node2vec=False,
    burnin=BURN_IN,
    save_dir=save_dir,
    objects=objects,
    node2id=node2id,
    edges=edges,
    hyperparams={
        'dim': DIM,
        'epochs': EPOCHS,
        'lr': LR,
        'burnin': BURN_IN,
        'n_neg': NNEGS,
        'num_walks': 10,
        'walk_length': 5,
    },
    patience=50,
    early_stop=0.001,
    c_optimizer=c_optimizer, 
    all_u_pos=all_u_pos,
    all_v_pos=all_v_pos
)
