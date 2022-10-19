import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.custom_layers import *

class PolicyEmbedding(nn.Module):
    def __init__(self, state_dim, **kwargs):
        super(PolicyEmbedding, self).__init__()
        self.state_dim = state_dim

        self.emb = PositionalEncoding(32)

        self.conv_1 = nn.Conv2d(1, 32, (3, 3), (1, 1))
        self.conv_2 = nn.Conv2d(32, 64, (3, 3), (1, 1))

        self.emb_1 = nn.Linear(2304, 256)
        self.emb_2 = nn.Linear(256, 256)
        self.emb_3 = nn.Linear(256, 256)
        self.emb_4 = nn.Linear(256, 256)

        self.output_dim = 256

    def forward(self, state):
        BS, _ = state.shape
        h = w = int(math.sqrt(self.state_dim))
        # state = self.emb(state)
        state = state.view(BS, 1, h, w)

        emb = F.relu(self.conv_1(state))
        emb = F.relu(self.conv_2(emb))
        BS, h, w, c = emb.shape
        emb = emb.view(BS, h*w*c)

        emb = F.relu(self.emb_1(emb))
        emb = F.relu(self.emb_2(emb))
        emb = F.relu(self.emb_3(emb))
        emb = F.relu(self.emb_4(emb))
        return emb

class CriticEmbedding(nn.Module):
    def __init__(self, state_dim, **kwargs):
        super(CriticEmbedding, self).__init__()
        self.state_dim = state_dim
        self.emb = PositionalEncoding(32)

        self.conv_1 = nn.Conv2d(1, 32, (3, 3), (1, 1))
        self.conv_2 = nn.Conv2d(32, 64, (3, 3), (1, 1))
        self.emb_1 = nn.Linear(2304, 256)
        self.emb_2 = nn.Linear(256, 256)
        self.emb_3 = nn.Linear(256, 256)
        self.emb_4 = nn.Linear(256, 256)

        self.output_dim = 256

    def forward(self, state):
        try:
            BS, _ = state.shape
        except Exception as e:
            _ = state.shape
            BS = 1
        h = w = int(math.sqrt(self.state_dim))
        # state = self.emb(state)
        state = state.view(BS, 1, h, w)

        emb = F.relu(self.conv_1(state))
        emb = F.relu(self.conv_2(emb))
        BS, h, w, c = emb.shape
        emb = emb.view(BS, h*w*c)

        emb = F.relu(self.emb_1(emb))
        emb = F.relu(self.emb_2(emb))
        emb = F.relu(self.emb_3(emb))
        emb = F.relu(self.emb_4(emb))
        return emb
