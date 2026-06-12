import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import pickle
import hashlib
import datetime

from torch.utils.data import TensorDataset

from data import prepare_data, compute_rfa
from poincare import PoincareManifold
from poincare_embeddings import Poincarre_embeddings
from RSGD import RiemanianSGD
from train import train
from load_data import * 

import os
import os.path


DIM = 15
GAMMA = 0.5
LR = 0.1
K_NEIGHBOURS = 10
SIGMA = 0.1
EARLY_STOP = 0.0001
N_PCA = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device : ", DEVICE)
distlocal = 'minkowski'

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

# Chargement des données
work = pd.concat([work_omim, work_orpha]).reset_index()
print(f"Taille du jeu de données : {work.shape}")
omim_to_idx = {v: i for i, v in enumerate(work['database_id'].values) if v.startswith('OMIM')} 
orpha_to_idx = {v: i for i, v in enumerate(work['database_id'].values) if v.startswith('ORPHA')}
correspondances = []
for _, row in df_orpha_omim.iterrows():
        if row['omim_id'] in omim_to_idx and row['orpha_id'] in orpha_to_idx:
            i = omim_to_idx[row['omim_id']]
            j = orpha_to_idx[row['orpha_id']]
            correspondances.append((i, j))

# Préparation des données
print("="*10, "Préparation des données", "="*10)
x, features, labels = prepare_data(work, with_labels=True, normalize=False, n_pca=N_PCA)


def get_cache_path(k_neighbours, sigma, distlocal, sym, connected, n_pca=0):
    """Génère un nom de fichier unique basé sur les paramètres."""
    if n_pca > 0:
        params = f"{k_neighbours}_{sigma}_{distlocal}_{sym}_{connected}_{n_pca}"
    else:
        params = f"{k_neighbours}_{sigma}_{distlocal}_{sym}_{connected}"
    hash_str = hashlib.md5(params.encode()).hexdigest()[:8]
    return f"../cache/rfa_{hash_str}.pkl"


cache_path = get_cache_path(K_NEIGHBOURS, SIGMA, distlocal, False, True, n_pca=N_PCA)

if os.path.exists(cache_path):
    print("Chargement RFA depuis le cache...")
    with open(cache_path, 'rb') as f:
        RFA, D_high = pickle.load(f)
else:
    print("Calcul RFA...")
    RFA, D_high = compute_rfa(
        features, mode='features', k_neighbours=K_NEIGHBOURS, 
        sym=False, connected=True, sigma=SIGMA, distlocal=distlocal,
        correspondances=correspondances, use_knn=True, method_knn='closest_both')
    os.makedirs("../cache", exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump((RFA, D_high), f)
    print(f"RFA sauvegardé dans {cache_path}")

# print("")
# print("RFA min/max/std :", RFA.min().item(), RFA.max().item(), RFA.std().item())
# print("RFA diag mean :", RFA.diagonal().mean().item())  # valeurs dominantes ?

batchsize = 64
if batchsize < 0:
    batchsize = min(512, int(len(RFA)/10))
    print('batchsize = ', batchsize)
lr = batchsize / 16 * LR
print('lr = ', lr)

indices = torch.arange(len(RFA))
if torch.cuda.is_available():
    indices = indices.cuda()
    RFA = RFA.cuda()

dataset = TensorDataset(indices, RFA)

manifold = PoincareManifold()
model = Poincarre_embeddings(
    n=len(dataset), 
    dim=DIM, 
    manifold=manifold, 
    gamma=GAMMA,
    learn_curvature=False, 
    init_curvature=1.,
    Qdist='laplace', 
    lossfn='klSym')

optimizer = RiemanianSGD(model.parameters(), lr=lr, manifold=manifold, c=model.c.item())

args = {
    "epochs": 750, 
    "lr": lr, 
    "burnin": 20, 
    "batchsize": batchsize, 
    "lrm": 0.1, 
    'dim':DIM,
    'gamma': GAMMA,
    }

embeddings, loss, epoch_loss, epoch = train(
    model, 
    dataset, 
    optimizer, 
    args, 
    DEVICE,
    save_dir,
    D_high, 
    labels=labels, 
    earlystop=EARLY_STOP,)


"""
torch.save({
    'model_state_dict': model.state_dict(),
    'embeddings': embeddings,
    'x': x,
    'data': dataset,
    'dist_kNNG': D_high,
    'losses': loss,
    'hyperparams': {
        'dim': DIM,
        'epochs': args["epochs"],
        'lr': args["lr"],
        'burnin': args["burnin"],
        'batchsize': args["batchsize"]
    }
}, 'models/poincare_hpo.pt')
"""
