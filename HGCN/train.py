import numpy as np
import os
import torch

from RiemAdam import RiemannianAdam
from RSGD import RiemanianSGD
from model import LPModel
from data import load_data


def train(args, G_hpo, features, save_dir):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device : ", device)
    args.patience = args.epochs if not args.patience else int(args.patience)

    data = load_data(args, G_hpo, features)
    args.n_nodes, args.feat_dim = data['features'].shape
    args.nb_false_edges = len(data['train_edges_false'])
    args.nb_edges = len(data['train_edges'])

    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs

    model = LPModel(args, device)

    if args.optimizer == 'Adam':
        optimizer = RiemannianAdam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == 'RSGD':
        optimizer = RiemanianSGD(params=model.parameters(), lr=args.lr, manifold=None)

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
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        embeddings = model.encode(data['features'], data['adj_train_norm'])
        train_metrics = model.compute_metrics(embeddings, data, 'train')
        train_metrics['loss'].backward()

        if args.grad_clip is not None:
            max_norm = float(args.grad_clip)
            all_params = list(model.parameters())
            for param in all_params:
                torch.nn.utils.clip_grad_norm_(param, max_norm)

        optimizer.step()
        lr_scheduler.step()
        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            embeddings = model.encode(data['features'], data['adj_train_norm'])
            val_metrics = model.compute_metrics(embeddings, data, 'val')

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
                    print("Early stopping")
                    break
    if not best_test_metrics:
        model.eval()
        best_emb = model.encode(data['features'], data['adj_train_norm'])
        best_test_metrics = model.compute_metrics(best_emb, data, 'test')
    if args.save:
        np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.cpu().detach().numpy())
