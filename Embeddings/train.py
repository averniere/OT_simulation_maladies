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
    burnin=10,
    eval_each=10,
    progress=False,
    device='cpu',
    checkpoint_dir='checkpoints',  # Nouveau paramètre
    save_every=10,
    verbose=True
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = model.to(device)
    losses = []

    for epoch in range(epochs):
        # Burn_in
        data.burnin = epoch < burnin
        current_lr  = lr * _lr_multiplier if data.burnin else lr
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

            epoch_loss[i_batch] = loss.detach().cpu().item()

        avg_loss = epoch_loss.mean().item()
        losses.append(avg_loss)

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
    if model is not None:
        model.eval()
        with torch.no_grad():
            embeddings = model.weight.detach().cpu().numpy()

    final_save_path = os.path.join(checkpoint_dir, "model_final.pt")        
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses_history': losses
    }, final_save_path)
    if verbose:
        print(f"Modèle final sauvegardé : {final_save_path}")

    return losses, embeddings
