import networkx as nx
import pandas as pd
import numpy as np
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
    
objects  = list(G_hpo.nodes())
node2id  = {n: i for i, n in enumerate(objects)}
edges = np.array([(node2id[u], node2id[v]) for u, v in G_hpo.edges()],dtype=np.int64)

manifold = PoincareManifold()
model = Distance_PE(n=len(objects), dim=2, manifold=manifold, sparse=True)
optimizer = RiemanianSGD(model.parameters(), lr=0.3, manifold=manifold)
data = BatchedDataset(edges, objects, nnegs=10, batch_size=50)

losses, embeddings = train( model, data, optimizer,
    epochs    = 100,
    lr        = 0.3,
    burnin    = 10,
    eval_each = 10,
    progress  = True,
)
