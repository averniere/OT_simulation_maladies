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
    D_high, 
    labels=None, 
    earlystop=0.0,
    verbose=True,
    c_optimizer=None,
    ):

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        
    # loader = DataLoader(data, batch_size=args["batchsize"], shuffle=True)
    model = model.to(device)
    n_iter = 0
    losses = []
    norm_history = []
    earlystop_count = 0
    pbar = tqdm(range(args["epochs"]), ncols=80)
    data_inputs = data.tensors[0]   # indices, déjà sur GPU
    data_targets = data.tensors[1]   # RFA, déjà sur GPU
    n0 = int(data_inputs.shape[0])
    for epoch in pbar:
        grad_norm = []
        # determine learning rate
        lr = args["lr"]
        if epoch < args["burnin"]:
            lr = lr * args["lrm"]
        else:
            lr = lr/(1 + 0.001 * (epoch - args['burnin']))
        epoch_loss = 0
        n_batches = 0
        perm = torch.randperm(n0, device=device)
        for start in range(0, n0, args["batchsize"]):
            idx = perm[start:start + args["batchsize"]]
            inputs = data_inputs[idx]
            targets = data_targets[idx]
        #for inputs, targets in loader:
            loss = model.lossfn(model(inputs), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(lr=lr)
            if model._log_c.requires_grad:
                c_optimizer.step()
                c_optimizer.zero_grad()
                optimizer.update_c(model.c.item())
            epoch_loss += loss.detach() 
            n_batches +=1
            grad_norm.append(model.embeddings.weight.grad.data.norm().item())            

            n_iter += 1

        epoch_loss = epoch_loss.item()/n_batches
        losses.append(epoch_loss)
        pbar.set_description("loss: {:.5f}".format(epoch_loss))

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
            'distances': D_high,
            'data': data,
            'args':args,
            'losses': losses,
            'norm_history': norm_history,
        }, final_path)
        if verbose:
            print(f"Modèle final sauvegardé : {final_path}")


    return model.embeddings.weight.cpu().detach().numpy(), losses, epoch_loss, epoch