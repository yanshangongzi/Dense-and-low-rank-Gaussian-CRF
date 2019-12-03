import torch
import torch.nn as nn
import torch.nn.functional as F

class BinPotentialsLayer(nn.Module):
    def __init__(self, P, L, D):
        super(BinPotentialsLayer, self).__init__()
        self.L = L
        self.D = D
        self.P = P
        self.layer = nn.Linear(2048, self.P * self.D * self.L)
    def forward(self, x):
        output = self.layer(x)
        A = output.reshape(self.L, self.P, self.D)
        return F.softmax(A, dim=0)


