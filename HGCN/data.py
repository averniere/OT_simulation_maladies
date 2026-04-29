import networkx as nx
import scipy.sparse


def load_data(args, hpo_graph, omim_df):
    # adjacency matrix
    nodes = list(hpo_graph.nodes())
    node2idx = {n: i for i, n in enumerate(nodes)}
    adj = nx.to_scipy_sparse_array(hpo_graph, nodelist=nodes, format='csr')

    # Features
    hpo_cols = [col for col in omim_df.columns if col in node2idx]
    omim_reindexed = omim_df[sorted(hpo_cols, key=lambda x: node2idx[x])]
    
    features = scipy.sparse.csr_matrix(omim_reindexed.values.T)  # (termes x maladies)
    
    data = {'adj_train': adj,'features': features}

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_edges(
                    adj, args.val_prop, args.test_prop, args.split_seed)


