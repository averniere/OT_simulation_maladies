import numpy as np
import torch
import os 
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.utils.data import TensorDataset, DataLoader


def train(
    model, 
    data, 
    optimizer, 
    args, 
    device, 
    save_dir, 
    labels=None, 
    earlystop=0.0,
    verbose=True,
    ):

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        
    loader = DataLoader(data, batch_size=args["batchsize"], shuffle=True)
    model = model.to(device)
    n_iter = 0
    losses = []
    norm_history = []
    earlystop_count = 0
    for epoch in tqdm(range(args["epochs"])):
        grad_norm = []

        # determine learning rate
        lr = args["lr"]
        if epoch < args["burnin"]:
            lr = lr * args["lrm"]

        epoch_loss = 0
        for inputs, targets in loader:
    
            loss = model.lossfn(model(inputs), targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step(lr=lr)

            epoch_loss += loss.item()
            
            grad_norm.append(model.embeddings.weight.grad.data.norm().item())            

            n_iter += 1

        epoch_loss /= len(loader)
        losses.append(epoch_loss)

        with torch.no_grad():
            n = model.weight.norm(dim=-1)
            norm_history.append({
                'mean': n.mean().item(),
                'max':  n.max().item(),
                'min':  n.min().item(),
            })

        if epoch > 10:
            delta = abs(losses[epoch] - losses[epoch-1])            
            if (delta < earlystop):
                earlystop_count += 1
            if earlystop_count > 50:
                break

    delta = abs(losses[epoch] - losses[epoch-1])

    if save_dir is not None:
        final_path = os.path.join(save_dir, "model_final.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'data': data,
            'args':args,
            'losses': losses,
            'norm_history': norm_history,
        }, final_path)
        if verbose:
            print(f"Modèle final sauvegardé : {final_path}")


    return model.embeddings.weight.cpu().detach().numpy(), losses, epoch_loss, epoch