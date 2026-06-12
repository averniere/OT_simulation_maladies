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
from batched_dataset import BatchedDataset
from train import train
from data import *
from data_utils import add_corresponding_terms, add_edges


# TEST : on relie les termes présents dans une même maladie ----------------------------------

union_diseases = add_corresponding_terms(work_omim, work_orpha, df_orpha_omim)
G_hpo_omim = add_edges(union_diseases, G_hpo_work, depths)

# --------------------------------------------------------------------------------------------

objects = list(G_hpo_work.nodes())
node2id = {n: i for i, n in enumerate(objects)}
edges = np.array([(node2id[u], node2id[v]) for u, v in G_hpo_work.edges()],dtype=np.int64)
print(f"{len(edges)} arêtes et {len(objects)} noeuds")

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


edges_closed = partial_transitive_closure(edges, max_depth=3)
print(f"Arêtes avant : {len(edges)}, après : {len(edges_closed)}")
# -------------------------------------------------------------------------------------------

DIM = 15
EPOCHS = 1500
LR0 = 0.3
BURN_IN = 100
NNEGS = 100
BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

LR = LR0/32*BATCH_SIZE
print(LR)

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

data = BatchedDataset(edges, objects, nnegs=NNEGS, batch_size=BATCH_SIZE)


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


losses, norms = train(
    model=model,
    data=data,
    optimizer=optimizer,
    epochs=EPOCHS,
    lr=LR,
    device=DEVICE,
    burnin=BURN_IN,
    save_dir=save_dir,
    objects=objects,
    node2id=node2id,
    edges=edges_closed,
    hyperparams={
        'dim': DIM,
        'epochs': EPOCHS,
        'lr': LR,
        'burnin': BURN_IN,
        'n_neg': NNEGS,
    },
    patience=50,
    early_stop=0.005,
    c_optimizer=c_optimizer
)


# Diagnostic 3
def visualize_training(model, losses, norm_history, objects, node2id, lr, burnin):

    W = model.weight.detach().cpu().numpy()
    norms = np.linalg.norm(W, axis=1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # ── 1. Disque de Poincaré ─────────────────────────────────────────
    ax = axes[0]
    ax.add_patch(plt.Circle((0,0), 1.0, color='gray', fill=False, lw=1.5, ls='--'))
    sc = ax.scatter(W[:,0], W[:,1], c=norms, cmap='plasma',
                    s=10, alpha=0.7, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label='‖θ‖')
    ax.scatter(W[0,0], W[0,1], c='red', s=10, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.set_title(f'Embeddings finaux\nnorme max={norms.max():.3f}  moy={norms.mean():.3f}')

    # ── 2. Évolution des normes ───────────────────────────────────────
    ax = axes[1]
    ax.plot([h['mean'] for h in norm_history], label='moyenne', color='steelblue')
    ax.plot([h['max']  for h in norm_history], label='max',     color='tomato')
    ax.plot([h['min']  for h in norm_history], label='min',     color='green')
    ax.axvline(burnin, color='orange', ls='--', label=f'fin burn-in ({burnin})')
    ax.set_xlabel('Epoch'); ax.set_ylabel('‖θ‖')
    ax.set_title('Évolution des normes')
    ax.legend(); ax.grid(alpha=0.3)

    # ── 3. Loss ───────────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(losses, color='steelblue', lw=2)
    ax.axvline(burnin, color='orange', ls='--', label=f'fin burn-in ({burnin})')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Courbe de loss')
    ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle(
        f'Run — {len(objects)} nœuds | dim={W.shape[1]} | lr ={lr} |'
        f'{len(losses)} epochs| batch_size ={BATCH_SIZE}',
        fontsize=12, y=1.02
    )
    plt.tight_layout()
    
    from datetime import datetime
    fname = f"plots/poincare_{datetime.now().strftime('%H%M%S')}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Figure sauvegardée : {fname}")
    plt.show()
    
    return fig

visualize_training(model, losses, norms, objects, node2id, lr=LR, burnin=BURN_IN)