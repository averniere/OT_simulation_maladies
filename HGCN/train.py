import numpy as np
import os
import torch
import json
from tqdm import tqdm 
from RiemAdam import RiemannianAdam
from RSGD import RiemanianSGD
from model import LPModel
from data import load_data2, load_data
#from geoopt.optim import RiemannianAdam


def train(args, G_hpo, features, save_dir, ancestors, depths):
    roots = [n for n in G_hpo.nodes() if G_hpo.out_degree(n) == 0]
    print(f"Nombre de racines : {len(roots)}")
    print(f"Racines : {roots}")
    print(f"HP:0000001 dans le graphe : {'HP:0000001' in G_hpo.nodes()}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print("Device : ", device)
    args.patience = args.epochs if not args.patience else int(args.patience)

    data = load_data(args, G_hpo, features, ancestors, depths, p=3)
    #zero_rows = np.where(data["features"].sum(axis=1) == 0)[0]
    args.n_nodes, args.feat_dim = data['features'].shape
    print(f"Dimension des features : {args.feat_dim}")
    args.nb_false_edges = len(data['train_edges_false'])
    args.nb_edges = len(data['train_edges'])

    # Diagnostic
    train_edges = data['train_edges']
    edges_false = data['train_edges_false'][np.random.randint(0, args.nb_false_edges, args.nb_edges)]

    train_nodes = set(train_edges.flatten().tolist())
    neg_nodes = set(edges_false.flatten().tolist())
    all_nodes = set(range(data['features'].shape[0]))

    missing = list(all_nodes - train_nodes)
    print("Features nulles parmi les noeuds orphelins:", 
      (data['features'][missing].sum(dim=1) == 0).sum().item())
    nodes_only_in_neg = neg_nodes - train_nodes

    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs

    model = LPModel(args, device)

    if args.optimizer == 'Adam':
        optimizer = RiemannianAdam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == 'RSGD':
        optimizer = RiemanianSGD(params=model.parameters(), lr=args.lr, manifold=None)
    #if not args.optimizer:
        #optimizer = RiemannianAdam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    #lr_scheduler = torch.optim.lr_scheduler.StepLR(
        #optimizer,
        #step_size=int(args.lr_reduce_freq),
        #gamma=float(args.gamma))

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=50,
        min_lr=1e-6
        )

    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    model = model.to(device)
    for x, val in data.items():
        if torch.is_tensor(data[x]):
            data[x] = data[x].to(device)

    counter = 0
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = None
    best_emb = None
    train_losses = []
    val_metrics_history = []

    for epoch in tqdm(range(args.epochs)):

        model.train()
        optimizer.zero_grad()
        embeddings = model.encode(data['features'], data['adj_train_norm'])
        embeddings.retain_grad()

        train_metrics = model.compute_metrics(embeddings, data, 'train')
        train_metrics['loss'].backward()
        # Peut potentiellement faire faire n'importe quoi 
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"NaN dans grad de {name}")
        # for param in model.parameters():
            # if param.grad is not None:
                # param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
        train_losses.append(train_metrics['loss'].item())

        if args.grad_clip is not None:
            max_norm = float(args.grad_clip)
            all_params = list(model.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, max_norm)
    
        optimizer.step()

        #lr_scheduler.step()
        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            embeddings = model.encode(data['features'], data['adj_train_norm'])
            if epoch == 0:
                print(f"Features input - mean: {data['features'].mean():.6f}, std: {data['features'].std():.6f}, zeros: {(data['features']==0).float().mean():.2%}")
                print(f"Embeddings - mean: {embeddings.mean():.6f}, std: {embeddings.std():.6f}")
                print(f"Embeddings nuls: {(embeddings.abs().sum(dim=1)==0).sum().item()} / {embeddings.shape[0]}")
            val_metrics = model.compute_metrics(embeddings, data, 'val')
            val_metrics_history.append({'epoch': epoch + 1, 
            **{k: v.item() if torch.is_tensor(v) else v for k, v in val_metrics.items()}})
            lr_scheduler.step(val_metrics['ap'])
            
            if model.has_improved(best_val_metrics, val_metrics):
                best_test_metrics = model.compute_metrics(embeddings, data, 'test')
                best_emb = embeddings.cpu()
                if args.save:
                    np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.detach().numpy())
                best_val_metrics = val_metrics
                counter = 0
            else:
                counter += 1
                if counter == args.patience and epoch > args.min_epochs:
                    print(f"Early stopping : {epoch}")
                    break
        if epoch % 50 == 0:
            manifold = model.manifold
    
            u_pos = embeddings[data['train_edges'][:, 0]]
            v_pos = embeddings[data['train_edges'][:, 1]]
            u_neg = embeddings[data['train_edges_false'][:, 0]]
            v_neg = embeddings[data['train_edges_false'][:, 1]]
    
            dist_pos = manifold.sqdist(u_pos, v_pos, c=model.c)
            dist_neg = manifold.sqdist(u_neg, v_neg, c=model.c)
    
            pos_scores = model.dc.forward(dist_pos)
            neg_scores = model.dc.forward(dist_neg)
    
            print(f"Distance pos : mean={dist_pos.mean():.4f}, std={dist_pos.std():.4f}")
            print(f"Distance neg : mean={dist_neg.mean():.4f}, std={dist_neg.std():.4f}")
            print(f"Scores positifs : mean={pos_scores.mean():.4f}, std={pos_scores.std():.4f}")
            print(f"Scores négatifs : mean={neg_scores.mean():.4f}, std={neg_scores.std():.4f}")
            print(f"Embeddings std: {embeddings.std(dim=0).mean():.6f}")

    if not best_test_metrics:
        model.eval()
        best_emb = model.encode(data['features'], data['adj_train_norm'])
        best_test_metrics = model.compute_metrics(best_emb, data, 'test')
    if args.save:
        np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.cpu().detach().numpy())
        with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
            json.dump({'train_losses': train_losses,
            'val_metrics': val_metrics_history,
            'best_test_metrics': {k: v.item() if torch.is_tensor(v) else v for k, v in best_test_metrics.items()}}, f)
