# Copyright (c) 2022 kakaoenterprise
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.layer1 = torch.nn.Linear(hparams.base_dim, hparams.mlp_hidden) # hidden layer
        self.layer2 = torch.nn.Linear(hparams.mlp_hidden, 1) # output layer
        self.relu = torch.nn.ReLU() # activation function


    def forward(self, x):
        out = self.layer2(self.relu(self.layer1(x)))
        return out
