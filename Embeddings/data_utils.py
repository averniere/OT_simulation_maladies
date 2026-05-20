import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def f_active_terms(row, hpo_cols, node2id, deprecated):
    active=[]
    for term in hpo_cols:
        if row[term]==1:
            resolved = deprecated.get(term, term)
            if resolved in node2id:
                active.append(resolved)
    return active
    

def len_active_terms(row, hpo_cols, node2id, deprecated):
    active=[]
    for term in hpo_cols:
        if row[term]==1:
            resolved = deprecated.get(term, term)
            if resolved in node2id:
                active.append(resolved)
    return len(active)


def build_disease_correspondence(df):
    """
    Construit une table de correspondance OMIM <-> ORPHA basée sur le nom des maladies.
    """
    omim = df[df['database_id'].str.startswith('OMIM:')][['disease_name', 'database_id']].drop_duplicates()
    orpha = df[df['database_id'].str.startswith('ORPHA:')][['disease_name', 'database_id']].drop_duplicates()

    correspondence = omim.merge(orpha, on='disease_name', suffixes=('_omim', '_orpha'))
    
    return correspondence.rename(columns={
        'database_id_omim' : 'omim_id',
        'database_id_orpha': 'orpha_id'
    })[['disease_name', 'omim_id', 'orpha_id']]


def build_fuzzy_correspondence(df, threshold=0.85):
    omim = df[df['database_id'].str.startswith('OMIM:')][['disease_name', 'database_id']].drop_duplicates()
    orpha = df[df['database_id'].str.startswith('ORPHA:')][['disease_name', 'database_id']].drop_duplicates()

    # Normaliser
    omim_names  = omim['disease_name'].str.lower().str.strip()
    orpha_names = orpha['disease_name'].str.lower().str.strip()

    # TF-IDF sur les caractères (ngram sur chars capture mieux les variations médicales)
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
    all_names  = pd.concat([omim_names, orpha_names])
    vectorizer.fit(all_names)

    omim_vecs  = vectorizer.transform(omim_names)   # (n_omim, vocab)
    orpha_vecs = vectorizer.transform(orpha_names)  # (n_orpha, vocab)

    # Matrice de similarité (n_omim, n_orpha)
    sim_matrix = cosine_similarity(omim_vecs, orpha_vecs)

    # Pour chaque OMIM, prendre le meilleur match ORPHA
    best_orpha_idx   = sim_matrix.argmax(axis=1)
    best_orpha_score = sim_matrix.max(axis=1)

    results = []
    for i, (omim_row, orpha_idx, score) in enumerate(
        zip(omim.itertuples(), best_orpha_idx, best_orpha_score)
    ):
        if score >= threshold:
            orpha_row = orpha.iloc[orpha_idx]
            results.append({
                'omim_id' : omim_row.database_id,
                'orpha_id': orpha_row['database_id'],
                'omim_name' : omim_row.disease_name,
                'orpha_name': orpha_row['disease_name'],
                'similarity' : round(score, 4),
            })

    df_corr = pd.DataFrame(results).sort_values('similarity', ascending=False)
    print(f"Correspondances exactes (=1.0)  : {(df_corr['similarity'] == 1.0).sum()}")
    print(f"Correspondances >= {threshold}      : {len(df_corr)}")
    return df_corr