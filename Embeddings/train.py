import torch
import os
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
    eval_each=10,
    progress=False,
    checkpoint_dir='checkpoints',  # Nouveau paramètre
    save_every=10,
    verbose=True
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = model.to(device)

    norm_history = []
    losses = []
    w_init = model.weight.detach().clone()

    for epoch in range(epochs):
        # Burn_in
        data.burnin = epoch < burnin
        current_lr  = lr * _lr_multiplier if data.burnin else lr/(1 + 0.01 * (epoch - burnin))
        epoch_loss = torch.zeros(len(data))
        loader = tqdm(data, desc=f"Epoch {epoch+1}/{epochs}") if progress else data

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
            scores = model.manifold.distance(u_exp, others)

            loss = model.loss(scores, targets)
            loss.backward()

            optimizer.step(lr=current_lr)

            model.manifold.normalize(model.weight.data)

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

    # Diagnostic 2
    w_final = model.weight.detach()
    delta = (w_final - w_init).norm(dim=-1)
    print(f"\nDéplacement moyen des embeddings : {delta.mean():.4f}")
    print(f"Déplacement max                  : {delta.max():.4f}")
    print(f"Embeddings non bougés (delta<1e-4): {(delta < 1e-4).sum().item()}")

    final_save_path = os.path.join(checkpoint_dir, "model_final.pt")        
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses_history': losses
    }, final_save_path)
    if verbose:
        print(f"Modèle final sauvegardé : {final_save_path}")

    return losses, norm_history

