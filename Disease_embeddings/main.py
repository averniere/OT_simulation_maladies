import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from torch.utils.data import TensorDataset

from data import prepare_data, compute_rfa
from poincare import PoincareManifold
from poincare_embeddings import Poincarre_embeddings
from RSGD import RiemanianSGD
from train import train

import os
import os.path


DIM=2
GAMMA = 5.0
LR = 0.5
K_NEIGHBOURS = 10
SIGMA = 0.001
EARLY_STOP = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

profils_omim = pd.read_csv("../data/profils_omim.csv.gz", index_col=0)
profils_omim = profils_omim.reset_index()

features, labels = prepare_data(profils_omim, with_labels=True, normalize=False)
RFA, D_high = compute_rfa(features, mode='features', k_neighbours=K_NEIGHBOURS, sym=True, connected = True, sigma=SIGMA, distlocal='minkowski')

print("RFA min/max/std :", RFA.min().item(), RFA.max().item(), RFA.std().item())
print("RFA diag mean :", RFA.diagonal().mean().item())  # valeurs dominantes ?

batchsize = 32
if batchsize<0 : 
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
model = Poincarre_embeddings(n=len(dataset), dim=DIM, manifold=manifold, Qdist='laplace', lossfn='klSym', gamma=GAMMA)

optimizer = RiemanianSGD(model.parameters(), lr=lr, manifold=manifold)

args={"epochs" : 300,"lr": lr, "burnin":10, "batchsize":batchsize, "lrm":0.1}

embeddings, loss, epoch_loss, epoch = train(model, dataset, optimizer, args, device=DEVICE, labels=labels, earlystop=EARLY_STOP)


torch.save({
    'model_state_dict': model.state_dict(),
    'embeddings': embeddings,
    'data': dataset,
    'dist_kNNG' : D_high,
    'losses': loss,
    'hyperparams': {
        'dim': DIM,
        'epochs': args["epochs"],
        'lr': args["lr"],
        'burnin' : args["burnin"],
        'batchsize' : args["batchsize"]
    }
}, 'models/poincare_hpo.pt')

os.rename("models/poincare_hpo.pt", os.path.join("models/", f"omim_{args["epochs"]}_LR{args["lr"]}_BS{args["batchsize"]}_D{DIM}.pt"))

# Diagnostic 3
def visualize_training(losses, W):

    norms = np.linalg.norm(W, axis=1)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    
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

    # Loss
    ax = axes[1]
    ax.plot(losses, color='steelblue', lw=2)
    ax.axvline(args["burnin"], color='orange', ls='--', label=f'fin burn-in ({args["burnin"]})')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Courbe de loss')
    ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle(
        f'Run — dim={W.shape[1]} | lr ={args["lr"]} |'
        f'{len(losses)} epochs| batch_size ={args["batchsize"]}|dim ={DIM}',
        fontsize=12, y=1.02
    )
    plt.tight_layout()
    
    fname = f"plots/omim_{args["epochs"]}_LR{args["lr"]}_BS{args["batchsize"]}_D{DIM}.pt.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Figure sauvegardée : {fname}")
    plt.show()
    
    return fig

visualize_training(loss, embeddings)