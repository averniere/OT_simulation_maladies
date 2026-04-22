import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.utils.data import TensorDataset, DataLoader


def train(model, data, optimizer, args, fout=None, labels=None, earlystop=0.0):
    loader = DataLoader(data, batch_size=args.batchsize, shuffle=True)

    pbar = tqdm(range(args.epochs))

    n_iter = 0
    losses = []
    earlystop_count = 0
    for epoch in pbar:        
        grad_norm = []

        # determine learning rate
        lr = args.lr
        if epoch < args.burnin:
            lr = lr * args.lrm

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

        if epoch > 10:
            delta = abs(losses[epoch] - losses[epoch-1])            
            if (delta < earlystop):
                earlystop_count += 1
            if earlystop_count > 50:
                break

    delta = abs(losses[epoch] - losses[epoch-1])

    return model.lt.weight.cpu().detach().numpy(), epoch_loss, epoch