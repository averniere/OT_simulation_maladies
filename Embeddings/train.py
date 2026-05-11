import torch
import os
import time
from tqdm import tqdm


_lr_multiplier = 0.1

def train(
    model,  # Distance_PE
    data, # BatchedDataset
    optimizer,  # RiemanianSGD
    epochs,
    lr,
    device,
    burnin,
    eval_each=50,
    progress=False,
    save_dir = None,  # Nouveau paramètre
    save_every=10,
    verbose=True,
    objects=None,
    node2id=None,
    edges=None,
    hyperparams=None,
    patience=None,
    early_stop=1e-4,
    c_optimizer=None
):
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    model = model.to(device)
    norm_history = []
    losses = []
    w_init = model.weight.detach().clone()

    best_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    for epoch in tqdm(range(epochs)):
        # Burn_in
        data.burnin = epoch < burnin
        if data.burnin:
            hard_ratio=0
        else:
            hard_ratio=0.5
        current_lr  = lr * _lr_multiplier if data.burnin else lr/(1 + 0.001 * (epoch - burnin))
        epoch_loss = torch.zeros(len(data))
        loader = tqdm(data.__iter__(model=model, hard_ratio=hard_ratio), total=len(data), desc=f"Epoch {epoch+1}/{epochs}") if progress else data.__iter__(model=model, hard_ratio=hard_ratio)

        # tqdm(data, desc=f"Epoch {epoch+1}/{epochs}") if progress else data

        for i_batch, inputs in enumerate(loader):
            # inputs : LongTensor (B, 2+nnegs)
            # target : index de la paire positive = 0 pour chaque ligne
            targets = torch.zeros(inputs.size(0), dtype=torch.long)
            inputs  = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(inputs)

            u = preds[:, 0, :]   # (B, dim)
            others = preds[:, 1:, :]  # (B, v+nnegs, dim)
            u_exp  = u.unsqueeze(1).expand_as(others)
            # Distance de Poincaré de u à v, n_negs
            scores = model.manifold.distance(u_exp, others, model.c)

            loss = model.loss(scores, targets)
            loss.backward()

            optimizer.step(lr=current_lr)
            if model._log_c.requires_grad:
                c_optimizer.step()
                c_optimizer.zero_grad()
                optimizer.update_c(model.c.item())

            model.manifold.normalize(model.weight.data, model.c)

            epoch_loss[i_batch] = loss.detach().cpu().item()
            
        avg_loss = epoch_loss.mean().item()
        losses.append(avg_loss)

        # Diagnostic 1
        with torch.no_grad():
            n = model.weight.norm(dim=-1)
            norm_history.append({
                'mean': n.mean().item(),
                'max':  n.max().item(),
                'min':  n.min().item(),
            })

        if (epoch + 1) % eval_each == 0 or epoch == 0:
            tag = "[burn-in]" if data.burnin else "         "
            print(
                f"json_stats: {{"
                f'"epoch": {epoch+1}, '
                f'"loss": {avg_loss:.4f}, '
                f'"lr": {current_lr}, '
                f'"burnin": {data.burnin}'
                f"}}"
            )
        
        if patience is not None and not data.burnin:
            if avg_loss < best_loss-early_stop:
                best_loss = avg_loss
                best_epoch = epoch+1
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping déclenché à l'epoch {epoch+1} (meilleure loss: {best_loss:.4f} à l'epoch {best_epoch})")
                break

        # if save_dir is not None and save_every and (epoch + 1) % save_every == 0:
            # ckpt_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt")
            # torch.save({
                # 'epoch': epoch + 1,
                # 'model_state_dict': model.state_dict(),
                # 'losses': losses,
                # 'norm_history': norm_history,
            # }, ckpt_path)
            # if verbose:
                # print(f"Checkpoint sauvegardé : {ckpt_path}")

    # Diagnostic 2
    w_final = model.weight.detach()
    delta = (w_final - w_init).norm(dim=-1)
    print(f"\nDéplacement moyen des embeddings : {delta.mean():.4f}")
    print(f"Déplacement max                  : {delta.max():.4f}")
    print(f"Embeddings non bougés (delta<1e-4): {(delta < 1e-4).sum().item()}")

    if save_dir is not None:
        final_path = os.path.join(save_dir, "model_final.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'data': data,
            'objects': objects,
            'node2id': node2id,
            'edges': edges,
            'losses': losses,
            "curvature": model.c.item(),
            'norm_history': norm_history,
            'hyperparams': hyperparams,
        }, final_path)
        if verbose:
            print(f"Modèle final sauvegardé : {final_path}")

    return losses, norm_history

