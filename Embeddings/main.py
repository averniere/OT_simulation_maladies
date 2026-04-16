import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import ast
import os

from poincare import PoincareManifold
from model import Distance_PE
from RSGD import RiemanianSGD
from batched_dataset import BatchedDataset
from train import train

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "..", "data", "HPOs.csv")

df_hpo = pd.read_csv(data_path, sep=";")
df_hpo = df_hpo.drop(columns = ["definition", "synonyms"])
df_hpo['parents'] = df_hpo['parents'].apply(ast.literal_eval)

G_hpo = nx.DiGraph()

for i, row in df_hpo.iterrows():
    G_hpo.add_node(row['hp_id'])

for j, row in df_hpo.iterrows():
    for parent_id in row["parents"]:
        if parent_id in G_hpo:
            G_hpo.add_edge(parent_id, row['hp_id'])
    
objects = list(G_hpo.nodes())
node2id = {n: i for i, n in enumerate(objects)}
edges = np.array([(node2id[u], node2id[v]) for u, v in G_hpo.edges()],dtype=np.int64)

DIM = 2
EPOCHS = 300
LR = 0.5
BURN_IN = 30
NNEGS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

manifold = PoincareManifold()
model = Distance_PE(n=len(objects), dim=DIM, manifold=manifold, sparse=True)
optimizer = RiemanianSGD(model.parameters(), lr=0.5, manifold=manifold)
data = BatchedDataset(edges, objects, nnegs=NNEGS, batch_size=50)

losses, norms = train(model, data, optimizer,
    epochs = EPOCHS,
    lr = LR,
    device = DEVICE,
    burnin = BURN_IN,
    eval_each = 10,
    progress = True)


torch.save({
    'model_state_dict': model.state_dict(),
    'data': data, 
    'objects': objects,
    'node2id': node2id,
    'edges': edges,
    'losses': losses,
    'norm_history': norms,
    'hyperparams': {
        'dim': DIM,
        'epochs':  EPOCHS,
        'lr': LR,
        'burnin': BURN_IN,
        'n_neg': NNEGS,
    }
}, 'models/poincare_hpo.pt')



# Diagnostic 3
def visualize_training(model, losses, norm_history, objects, node2id, burnin):

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
        f'Run — {len(objects)} nœuds | dim={W.shape[1]} | '
        f'{len(losses)} epochs',
        fontsize=12, y=1.02
    )
    plt.tight_layout()
    
    from datetime import datetime
    fname = f"plots/poincare_{datetime.now().strftime('%H%M%S')}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Figure sauvegardée : {fname}")
    plt.show()
    
    return fig

visualize_training(model, losses, norms, objects, node2id, burnin=10)