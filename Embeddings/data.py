import pandas as pd
import networkx as nx
import requests, xml.etree.ElementTree as ET
import urllib.request
import re
import gc

from scipy.sparse import csr_matrix

import psutil, os
def mem():
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


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


def read_hpoa(path, usecols=None, dtype=None):
    with open(path, 'r') as f:
        skip = sum(1 for line in f if line.startswith('#'))
    return pd.read_csv(
        path, sep='\t', skiprows=skip, low_memory=False, usecols=usecols
    )

cols_needed = ['database_id', 'hpo_id', 'disease_name', 'frequency']
df_hpoa = read_hpoa('../data/phenotype_omim_orpha.hpoa', usecols=cols_needed)
df_hpoa['disease_name'] = df_hpoa['disease_name'].str.lower().str.strip().str.replace(r'[\s\-]+', ' ', regex=True)


# Construction de deux dataframes à partir de df_hpoa
df_pivot = df_hpoa[['database_id', 'hpo_id']].drop_duplicates()
df_pivot['values']=1.
df_pivot = pd.pivot_table(data=df_pivot, values='values', index='database_id', columns='hpo_id', aggfunc='max', fill_value=0)
df_pivot.columns.name = None
df_pivot = df_pivot.reset_index()

profils_omim = pd.read_csv("../data/profils_omim.csv.gz", index_col=0)
profils_omim = profils_omim.reset_index()
hpo_cols0 = [c for c in profils_omim.columns if c.startswith('HP:')]

# Associations gène-maladie et PPI
genes_to_disease = pd.read_csv("../data/genes_to_disease.txt", sep="\t")
genes_to_disease = genes_to_disease.drop(columns='source')

ppi = pd.read_csv(
    "https://stringdb-downloads.org/download/stream/protein.links.v12.0/9606.protein.links.v12.0.min700.csv.gz", 
    sep=",",
    dtype={"protein1": "str", "protein2": "str", "combined_score": "int16"}
    )


doc = pd.read_csv(
    "https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz", 
    sep="\t",
    usecols=["#string_protein_id", "preferred_name", "annotation"])
doc = doc.rename(columns={"preferred_name":"gene_symbol"})

df0 = pd.merge(genes_to_disease, doc, how='left', on='gene_symbol')
df1 = pd.merge(df0, ppi, how='left', left_on="#string_protein_id", right_on="protein1")
df1 = df1.drop(columns="protein1")
print(df1.dtypes)
print()

# ======================================================================================
# ================== Correspondances issues d'Orphadata ================================
# ======================================================================================
print("1")
url = "https://www.orphadata.com/data/xml/en_product1.xml"
response = requests.get(url)
response.raise_for_status()
tree = ET.fromstring(response.content)
rows = []
for disorder in tree.iter("Disorder"):
    orpha_id = "ORPHA:" + disorder.findtext("OrphaCode")
    for ref in disorder.iter("ExternalReference"):
        if ref.findtext("Source") == "OMIM":
            rows.append({
                "orpha_id": orpha_id,
                "omim_id":  "OMIM:" + ref.findtext("Reference"),
                "mapping_type": ref.findtext("DisorderMappingRelation/Name")
            })

df_orpha_omim = pd.DataFrame(rows)
df_orpha_omim_exact = df_orpha_omim[df_orpha_omim["mapping_type"]=='E (Exact mapping: the two concepts are equivalent)']

list_omim = df_orpha_omim['omim_id'].unique()
list_orpha = df_orpha_omim['orpha_id'].unique()

work_omim = df_pivot[df_pivot['database_id'].isin(list_omim)]
work_orpha = df_pivot[df_pivot['database_id'].isin(list_orpha)]

df1_agg = df1.groupby('disease_id', as_index=False, dropna=True).agg(
    ncbi_gene_id=("ncbi_gene_id", "first"),
    gene_symbol=('gene_symbol', lambda x: list(set(x.dropna()))),
    association_type=("association_type", "first"),
    protein=('#string_protein_id', lambda x: list(set(x.dropna()))),
    annotation=("annotation", "first"),
    protein2=("protein2", lambda x: list(set(x.dropna()))),
    combined_score=("combined_score", lambda x: list(set(x.dropna()))),
    n_proteins=('#string_protein_id', lambda x: len(set(x.dropna()))),
)

df1_omim = pd.merge(work_omim, df1_agg, how='left', left_on='database_id', right_on='disease_id')
#df1_omim = pd.merge(work_omim, df1, how='left', left_on='database_id', right_on='disease_id')

df1_orpha = pd.merge(work_orpha, df1_agg, how='left', left_on='database_id', right_on='disease_id')
#df1_orpha = pd.merge(work_orpha, df1, how='left', left_on='database_id', right_on='disease_id')

del df_pivot
gc.collect()
# ======================================================================================
# ================== Maladies Orphanet depuis Orphadata ================================
# ======================================================================================
print("2")
url = "https://www.orphadata.com/data/xml/en_product4.xml"
local_file = "en_product4.xml"

# Téléchargement
urllib.request.urlretrieve(url, local_file)

# Parsing
tree = ET.parse(local_file)
root = tree.getroot()

rows = []
for disorder in root.iter("Disorder"):
    orpha_code = disorder.findtext("OrphaCode")
    name = disorder.findtext("Name")
    for assoc in disorder.iter("HPODisorderAssociation"):
        rows.append({
            "disease_id": f"ORPHA:{orpha_code}",
            "Disorder": name,
            "HPO_id": assoc.findtext("HPO/HPOId"),
            "HPOTerm": assoc.findtext("HPO/HPOTerm"),
            "Frequency": assoc.findtext("HPOFrequency/Name"),
        })

deprecated={
    'HP:0006887':'HP:0001249',
    'HP:0002275':'HP:0002311',
    'HP:0002370':'HP:0002311',
    'HP:0002438':'HP:0001317',
    'HP:0004059':'HP:0006433',
    'HP:0005365':'HP:0010976',
    'HP:0005435':'HP:0011840',
    'HP:0005807':'HP:0009881',
    'HP:0007543':'HP:0000962',
    'HP:0007680':'HP:0007894',
    'HP:0007850':'HP:0030666',
    'HP:0007898':'HP:0012231',
    'HP:0009062':'HP:0008936',
    'HP:0010064':'HP:0010091',
    'HP:0012178':'HP:0012177',
    'HP:0030050':'HP:0002524',  # Suspect
    'HP:0031014':'HP:0031632',
    'HP:0100786':'HP:0001262',
    'HP:0200065':'HP:0000533',
    'HP:0008014':'HP:0032416',
    'HP:0008715':'HP:0008733',
    'HP:0009773':'HP:0009700',
    'HP:0012509':'HP:0034591',
    'HP:0025237':'HP:6000456',
    'HP:0025239':'HP:0025240',
    'HP:0030220':'HP:0000719',
    'HP:0031297':'HP:0011643',
    'HP:0032172':'HP:0033661',
    'HP:0031530':'HP:0031528',
    'HP:0030638':'HP:0007642',
    'HP:0031349':'HP:0011540',
    'HP:0030639':'HP:0007642',
    'HP:0045080':'HP:0005403'
}

orphadata = pd.DataFrame(rows)
orphadata['HPO_id'] = orphadata['HPO_id'].str.strip()
orphadata['HPO_id'] = orphadata['HPO_id'].replace(deprecated)
orphadata = orphadata.dropna(subset=['HPO_id'])

pivot = orphadata[['disease_id', 'HPO_id']].drop_duplicates()
pivot['values'] = 1.


def pivot_sparse(df, index_col, columns_col, value_col):
    row_cat = df[index_col].astype('category')
    col_cat = df[columns_col].astype('category')
    sp = csr_matrix(
        (df[value_col].values, (row_cat.cat.codes, col_cat.cat.codes)),
        shape=(len(row_cat.cat.categories), len(col_cat.cat.categories))
    )
    out = pd.DataFrame.sparse.from_spmatrix(
        sp, index=row_cat.cat.categories, columns=col_cat.cat.categories
    )
    out.index.name = index_col
    return out.reset_index()

pivot = pd.pivot_table(data=pivot, values='values', index='disease_id', columns='HPO_id', aggfunc='max', fill_value=0)
#pivot = pivot_sparse(pivot, 'disease_id', 'HPO_id', 'values')
pivot.columns.name = None
pivot = pivot.reset_index()

all_columns = work_omim.columns.union(pivot.columns)

pivot_aligned = pivot.reindex(columns=all_columns, fill_value=0)
pivot_aligned = pivot_aligned.drop(columns=['database_id'])
work_omim2 = work_omim.reindex(columns=all_columns, fill_value=0).drop(columns=['disease_id'])

work_orpha2 = pivot_aligned[pivot_aligned['disease_id'].isin(list_orpha)]
work_orpha2 = work_orpha2.rename(columns={'disease_id':'database_id'})

del pivot, pivot_aligned
gc.collect()

# ======================================================================================
# ================== Bases avec les fréquences =========================================
# ======================================================================================
print("3")
def convert_frequency(
    freq, dico={'HP:0040283':0.05,'HP:0040280':1.,'HP:0040282':0.3,'HP:0040285':-1,'HP:0040281':0.8,'HP:0040284':0.01}
    ):
    if pd.isna(freq):
        return 0.
    freq = str(freq).strip()
    if freq.startswith('HP'):
        return dico[freq]
    if re.search(r"[0-9]\/[0-9]", freq):
        num, den = freq.split("/")
        return float(num) / float(den)
    if freq.endswith("%"):
        return float(freq[:-1]) / 100

df_hpoa["value"] = df_hpoa.apply(lambda row: convert_frequency(row.get('frequency')), axis=1)
relevant_ids = set(list_omim) | set(list_orpha)
df_hpoa_filtered = df_hpoa[df_hpoa['database_id'].isin(relevant_ids)]

n_dis = df_hpoa['database_id'].nunique()
n_hpo = df_hpoa['hpo_id'].nunique()

matrix = df_hpoa.pivot_table(
    index="database_id",
    columns="hpo_id",
    values="value",
    aggfunc="first",
    fill_value=0
)

#matrix = pivot_sparse(df_hpoa_filtered, "database_id", "hpo_id", "value")

matrix.columns.name = None
matrix = matrix.reset_index()

work_omimF = matrix[matrix['database_id'].isin(list_omim)]
work_orphaF = matrix[matrix['database_id'].isin(list_orpha)]

dico = {'Very frequent (99-80%)':0.8,
'Frequent (79-30%)':0.3,
'Occasional (29-5%)':0.05,
'Very rare (<4-1%)':0.01,
'Excluded (0%)':-1.,
'Obligate (100%)':1.}
orphadata['value'] = orphadata['Frequency'].replace(dico)

matrix_orpha = orphadata.pivot_table(
    index="disease_id",
    columns="HPO_id",
    values="value",
    aggfunc="first",
    fill_value=0
)
#matrix_orpha = pivot_sparse(orphadata, "disease_id", "HPO_id", "value")
matrix_orpha.columns.name = None
matrix_orpha = matrix_orpha.reset_index()

matrix_orpha = matrix_orpha.reindex(columns=all_columns, fill_value=0).drop(columns=['database_id'])
work_omimF2 = work_omimF.reindex(columns=all_columns, fill_value=0).drop(columns=['disease_id'])

work_orphaF2 = matrix_orpha[matrix_orpha['disease_id'].isin(list_orpha)]
work_orphaF2 = work_orphaF2.rename(columns={'disease_id':'database_id'})

del matrix, matrix_orpha
gc.collect()