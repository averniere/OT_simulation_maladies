import pandas as pd
import networkx as nx
import re

from data_utils import build_disease_correspondence, find_gene_correspondence

hp_ids = []
parents_list = []

with open("../data/HPOs.csv", "r") as f:
    next(f)
    for line in f:
        hp_id = line.split(';')[0]
        
        # Extraire uniquement la liste contenant des IDs HP:XXXXXXX
        match = re.search(r"\[([^\]]*'HP:\d{7}'[^\]]*)\]", line)
        if match:
            parents = re.findall(r"HP:\d{7}", match.group(0))
        else:
            parents = []
        
        hp_ids.append(hp_id)
        parents_list.append(parents)

df_hpo = pd.DataFrame({'hp_id': hp_ids, 'parents': parents_list})

G_hpo_work = nx.DiGraph()
for hp_id in hp_ids:
    G_hpo_work.add_node(hp_id)
for hp_id, parents in zip(hp_ids, parents_list):
    for parent_id in parents:
        if parent_id in G_hpo_work:
            G_hpo_work.add_edge(hp_id, parent_id)

G_hpo_work.add_edge('HP:0430046', 'HP:0001382')  # Missing edge
objects_w = list(G_hpo_work.nodes())
node2id_w = {n: i for i, n in enumerate(objects_w)}
root = "HP:0000001"
depths = nx.single_source_shortest_path_length(G_hpo_work.reverse(), source=root)


def read_hpoa(path):
    with open(path, 'r') as f:
        skip = sum(1 for line in f if line.startswith('#'))
    return pd.read_csv(path, sep='\t', skiprows=skip, low_memory=False)


df_hpoa = read_hpoa('../data/phenotype_omim_orpha.hpoa')
df_hpoa['disease_name'] = df_hpoa['disease_name'].str.lower().str.strip().str.replace(r'[\s\-]+', ' ', regex=True)
df_hpoa.tail()

correspondence_exacte = build_disease_correspondence(df_hpoa)
print(f"Correspondances trouvées : {len(correspondence_exacte)}")

# Construction de deux dataframes à partir de df_hpoa
df_pivot = df_hpoa[['database_id', 'hpo_id']].drop_duplicates()
df_pivot['values']=1.
df_pivot = pd.pivot_table(data=df_pivot, values='values', index='database_id', columns='hpo_id', aggfunc='max', fill_value=0)
df_pivot.columns.name = None
df_pivot = df_pivot.reset_index()

df_orpha = df_pivot[df_pivot['database_id'].str.startswith('ORPHA:')]
df_orpha = df_orpha[df_orpha['database_id'].isin(correspondence_exacte['orpha_id'])]

df_omim = df_pivot[df_pivot['database_id'].str.startswith('OMIM:')]
df_omim = df_omim[df_omim['database_id'].isin(correspondence_exacte['omim_id'])]

hpo_cols = [c for c in df_omim.columns if c.startswith('HP:')]

profils_omim = pd.read_csv("../data/profils_omim.csv.gz", index_col=0)
profils_omim = profils_omim.reset_index()
hpo_cols0 = [c for c in profils_omim.columns if c.startswith('HP:')]

# Associations gène-maladie et PPI
genes_to_disease = pd.read_csv("../data/genes_to_disease.txt", sep="\t")
genes_to_disease = genes_to_disease.drop(columns='source')

ppi = pd.read_csv("https://stringdb-downloads.org/download/stream/protein.links.v12.0/9606.protein.links.v12.0.min700.csv.gz", sep="," )

doc = pd.read_csv("https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz", sep="\t")
doc = doc.rename(columns={"preferred_name":"gene_symbol"})
doc = doc.drop(columns="protein_size")

# Base de données de correspondances OMIM-ORPHA en fonction du gène indiqué
df_gene_correspondence = find_gene_correspondence(genes_to_disease, 'disease_id', 'gene_symbol')
df_gene_correspondence0 = df_gene_correspondence.loc[df_gene_correspondence['orpha_id'].isin(df_hpoa['database_id'])]

list_omim = df_gene_correspondence0['omim_id'].unique()
list_orpha = df_gene_correspondence0['orpha_id'].unique()

gene_omim = df_pivot[df_pivot['database_id'].isin(list_omim)]
gene_orpha = df_pivot[df_pivot['database_id'].isin(list_orpha)]
print("gene_omim : ", gene_omim.shape)
print("gene_orpha : ", gene_orpha.shape)