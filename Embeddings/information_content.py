import numpy as np
from collections import defaultdict
from tqdm import tqdm


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
    'HP:0200065':'HP:0000533'
}


def get_ancestors0(G, node):
    visited = set()
    queue = list(G.successors(node))
    while queue:
        current = queue.pop()
        if current not in visited:
            visited.add(current)
            queue.extend(G.successors(current))
    return visited


def compute_information_content(df_omim, G_hpo, deprecated=deprecated):
    colnames = [c for c in df_omim.columns if c.startswith('HP:')]
    hp_matrix = df_omim[colnames].values
    ids = df_omim.index.tolist()

    weights = defaultdict(float)
    diseases = defaultdict(set)
    all_diseases = defaultdict(set)
    ancestors = {}

    def get_ancestors(term):
        if term not in ancestors:
            ancestors[term]=get_ancestors0(G_hpo, term)
        return ancestors[term]

    row_idxs, col_idxs = np.where(hp_matrix == 1)
    for row_idx, col_idx in zip(row_idxs, col_idxs):
        disease_id = ids[row_idx]
        term = colnames[col_idx]
        resolved = deprecated.get(term, term)
        
        weights[resolved] += 1
        diseases[resolved].add(disease_id)
        all_diseases[resolved].add(disease_id)
        
        for ancestor in get_ancestors(resolved):
            weights[ancestor] += 1
            all_diseases[ancestor].add(disease_id)

    total = sum(weights.values())
    return {t: w / total for t, w in weights.items()}, diseases, all_diseases

    '''
    for id, row in df_omim.iterrows():
        for term in colnames:
            if row[term]==1:
                resolved = deprecated.get(term, term)
                weights[resolved]+=1
                diseases[resolved].add(id)
                all_diseases[resolved].add(id)
                for ancestor in get_ancestors(resolved):
                    weights[ancestor]+=1
                    all_diseases[ancestor].add(id)
    total = sum(weights.values())
    return {t: w / total for t, w in weights.items()}, diseases, all_diseases
'''


def resnik_similarity(df, G_hpo, ic, deprecated=deprecated):
    """
    Calcule la similarité entre chaque terme HPO selon l'article de Resnik. Calculer la similarité
    entre deux termes revient à prendre le maximum de l'information content parmi tous les parents
    communs.
    """
    colnames = [c for c in df.columns if c.startswith('HP:')]
    n = len(colnames)
    sim = np.zeros((n, n))
    print("Precomputing ancestors...")
    ancestors = {t: get_ancestors0(G_hpo, t) | {t} for t in colnames}
    print("Finished !")
    for i, term_i in tqdm(enumerate(colnames)):
        for j, term_j in enumerate(colnames):
            if j < i:
                sim[i, j] = sim[j, i]
                continue
            if i == j:
                sim[i, j] = ic.get(term_i, 0.0)
                continue
            common_ancestors = ancestors[term_i] & ancestors[term_j]
            if not common_ancestors:
                sim[i, j] = 0.0
            else:
                sim[i, j] = max(ic.get(c, 0.0) for c in common_ancestors)
    return sim


