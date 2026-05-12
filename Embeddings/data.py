import os
import ast
import pandas as pd
import networkx as nx

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "..", "data", "HPOs.csv")

df_hpo = pd.read_csv(data_path, sep=";")
df_hpo = df_hpo.drop(columns = ["definition", "synonyms"])
# df_hpo['parents'] = df_hpo['parents'].apply(ast.literal_eval)
df_hpo['hp_id'] = df_hpo['hp_id'].str.strip("'\"")
df_hpo['parents'] = df_hpo['parents'].apply(
    lambda lst: [p.strip("'\"") for p in ast.literal_eval(lst)] 
    if isinstance(lst, str) else [p.strip("'\"") for p in lst])

G_hpo = nx.DiGraph()

for i, row in df_hpo.iterrows():
    G_hpo.add_node(row['hp_id'])

for j, row in df_hpo.iterrows():
    for parent_id in row["parents"]:
        if parent_id in G_hpo:
            G_hpo.add_edge(row['hp_id'], parent_id)  # Permet d'avoir la racine au centre
