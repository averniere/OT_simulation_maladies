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


def train(args, G_hpo, features, save_dir):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print("Device : ", device)
    args.patience = args.epochs if not args.patience else int(args.patience)

    data = load_data2(args, G_hpo, features)
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

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma))
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
        for param in model.parameters():
            if param.grad is not None:
                param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
        train_losses.append(train_metrics['loss'].item())

        if args.grad_clip is not None:
            max_norm = float(args.grad_clip)
            all_params = list(model.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, max_norm)
            '''
            for param in all_params:
                torch.nn.utils.clip_grad_norm_(param, max_norm)
            '''
        optimizer.step()
        for group in optimizer.param_groups:
            for p in group['params']:
                if p in optimizer.state:
                    state = optimizer.state[p]
                    if 'exp_avg' in state:
                        state['exp_avg'] = torch.nan_to_num(state['exp_avg'], nan=0.0, posinf=0.0, neginf=0.0)
                    if 'exp_avg_sq' in state:
                        state['exp_avg_sq'] = torch.nan_to_num(state['exp_avg_sq'], nan=0.0, posinf=0.0, neginf=0.0)
        with torch.no_grad():
            for param in model.parameters():
                param.data = torch.nan_to_num(param.data, nan=0.0, posinf=0.0, neginf=0.0)
        lr_scheduler.step()
        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            embeddings = model.encode(data['features'], data['adj_train_norm'])
            val_metrics = model.compute_metrics(embeddings, data, 'val')
            val_metrics_history.append({'epoch': epoch + 1, 
            **{k: v.item() if torch.is_tensor(v) else v for k, v in val_metrics.items()}})

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
