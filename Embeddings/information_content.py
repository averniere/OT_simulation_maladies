import numpy as np
import pandas as pd
import networkx as nx
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
    '''
    Entrées :
        - df_omim : dataset de maladies annotées,
        - G_hpo : graphe de l'ontologie HPO,
        - deprecated : liste de termes apparaissant dans les maladies et mis à jour depuis.
    Sortie :
        - Dictionnaire qui associe à chaque terme HPO apparaissant dans au moins une maladie la 
        valeur de l'IC.
        - diseases : dictionnaire {terme HPO : liste des maladies qui le contiennent}
        - all_diseases : dictionnaire {terme HPO : maladies qui le contiennent avec propagation ancestrale}
    NB : On pondère sur l'ensemble des noeuds du graphe, pas juste sur les maladies. Du point de vue
    de l'interprétation c'est moins élégant, car on n'a pas de probabilité d'apparition dans une maladie,
    mais au niveau des résultats c'est un peu mieux. 
    '''
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

    row_idxs, col_idxs = np.where(hp_matrix > 0)
    for row_idx, col_idx in zip(row_idxs, col_idxs):
        disease_id = ids[row_idx]
        term = colnames[col_idx]  # terme HPO
        resolved = deprecated.get(term, term)
        
        weights[resolved] += hp_matrix[row_idx, col_idx]  #1
        diseases[resolved].add(disease_id)
        all_diseases[resolved].add(disease_id)
        
        for ancestor in get_ancestors(resolved):
            weights[ancestor] += hp_matrix[row_idx, col_idx]  #1
            all_diseases[ancestor].add(disease_id)

    total = sum(weights.values())
    return {t: w / total for t, w in weights.items()}, diseases, all_diseases
    # return {t: w /df_omim.shape[0]  for t, w in weights.items()}, diseases, all_diseases

    
def resnik_similarity(df, G_hpo, ic, deprecated=deprecated):
    """
    Calcule la similarité entre chaque terme HPO selon l'article de Resnik. Calculer la similarité
    entre deux termes revient à prendre le maximum de l'information content parmi tous les parents
    communs (Most Informative Common Ancestor).
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


def inspect_weights(weights, disease, all_disease, G_hpo, top_n=5):
    '''
    Entrées :
        - weights : l'information content.
        - disease, all_disease : deux dernières sorties de compute_information_content.
        - G_hpo : le graphe utilisé, construit dans le sens enfant -> parent.
    Sortie :
        - Dataframe qui renseigne pour chaque terme HPO, son IC et ses caractéristiques dans le 
        graphe (profondeur, nombre de parents, d'enfants). n_disease indique les maladies dans
        lesquelles les termes apparaissent directement et all_diseases les maladies dans lesquelles
        ils apparaissent après remontée ancestrale.
    '''
    roots = [n for n in G_hpo.nodes if G_hpo.out_degree(n) == 0]
    root = roots[0] if roots else None

    G_inv = G_hpo.reverse()
    if root:
        depths = nx.single_source_shortest_path_length(G_inv, root)
    else:
        depths = {}

    rows = []
    for term, weight in weights.items():
        n_ancestors = len(get_ancestors0(G_hpo, term))
        n_children = G_hpo.in_degree(term)
        rows.append({
            'term'       : term,
            'weight'     : weight,
            'depth'      : depths.get(term, -1),
            'n_ancestors': n_ancestors,
            'n_children' : n_children,
            'n_diseases' : len(disease.get(term, set())) if term in disease else 0,
            'n_diseases tot':len(all_disease.get(term, set())) if term in all_disease else 0
        })

    df = (pd.DataFrame(rows)
            .sort_values('weight', ascending=False)
            .reset_index(drop=True))

    df.index += 1
    df['weight'] = df['weight'].map('{:.6f}'.format)

    print(f"Racine détectée : {root}")
    print(f"Termes : {len(weights)}\n")
    print(df.tail(top_n).to_string())
    return df
