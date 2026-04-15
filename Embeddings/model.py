import torch
import torch.nn as nn



class Poincarre_embeddings(nn.Module):

    def __init__(self, n, dim, sparse):
        super(Poincarre_embeddings, self).__init__()
        self.embeddings = nn.Embedding(n, dim, sparse=sparse)
        nn.init.uniform_(self.embeddings.weight, -1e-3, 1e-3)

    def forward(self, inputs):
        return self.embeddings(inputs)
    
    @property
    def weight(self):
        return self.embeddings.weight
 

    



    