from collections import defaultdict


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
    weights = defaultdict(float)
    diseases = defaultdict(set)
    all_diseases = defaultdict(set)
    ancestors = {}

    def get_ancestors(term):
        if term not in ancestors:
            ancestors[term]=get_ancestors0(G_hpo, term)
        return ancestors[term]

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