import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .binary_potentials import BinPotentialsLayer

model = models.resnet101(pretrained=False)

model.fc = BinPotentialsLayer(P, L, D) # FIX PARAMETERS
