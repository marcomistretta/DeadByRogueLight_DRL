import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.custom_layers import *

class PolicyEmbedding(nn.Module):
    def __init__(self, state_dim, **kwargs):
        super(PolicyEmbedding, self).__init__()
        self.state_dim = state_dim

        self.embGlobal = nn.Embedding(5, 8)
        self.embLocal0 = nn.Embedding(5, 8)
        self.embLocal1 = nn.Embedding(5, 8)
        self.embLocal2 = nn.Embedding(5, 8)

        # Global map
        self.g_conv_1 = nn.Conv2d(8, 32, (3, 3), (2, 2))
        self.g_conv_2 = nn.Conv2d(32, 64, (3, 3), (2, 2))

        # Local map 0
        self.l0_conv_1 = nn.Conv2d(8, 32, (3, 3), (1, 1))
        self.l0_conv_2 = nn.Conv2d(32, 64, (3, 3), (1, 1))

        # Local map 1
        self.l1_conv_1 = nn.Conv2d(8, 32, (3, 3), (1, 1))
        self.l1_conv_2 = nn.Conv2d(32, 64, (3, 3), (1, 1))

        # Local map 2
        self.l2_conv_1 = nn.Conv2d(8, 32, (3, 3), (1, 1))
        self.l2_conv_2 = nn.Conv2d(32, 64, (3, 3), (1, 1))

        # Properties emb
        self.prop_emb_1 = nn.Linear(6, 256)
        self.prop_emb_2 = nn.Linear(256, 256)

        self.emb_1 = nn.Linear(7616, 256)
        self.emb_2 = nn.Linear(256, 256)

        self.output_dim = 256

    def forward(self, state):
        BS, _ = state.shape

        gw = gh = 20
        l0w = l0h = 9
        l1w = l1h = 5
        l2w = l2h = 3
        prop = 6

        # global_emb, local1_emb, local2_emb, properties = torch.split(state, [gw*gh, l1h*l1w, l2h*l2w, prop], dim=1)

        local0_emb, local1_emb, local2_emb, properties = torch.split(state, [l0w*l0h, l1h*l1w, l2h*l2w, prop], dim=1)

        # Global map 10x10
        # pg = global_emb.view(20, 20)
        # print(torch.flip(torch.rot90(pg, 1, dims=[1, 0]), dims=[1,]))

        # global_emb = F.tanh(self.embGlobal(global_emb.int()))
        # global_emb = global_emb.view(BS, 8, gh, gw)
        # global_emb = F.one_hot(global_emb.long(), 8)
        # global_emb = global_emb.view(BS, 8, gh, gw).float()
        #
        # global_emb = F.pad(global_emb, (1, 1, 1, 1))
        # global_emb = F.relu(self.g_conv_1(global_emb))
        # global_emb = F.pad(global_emb, (1, 1, 1, 1))
        # global_emb = F.relu(self.g_conv_2(global_emb))
        # global_emb = global_emb.view(BS, -1)

        # Local map 9x9
        # l1g = local1_emb.view(5, 5)
        # print(torch.flip(torch.rot90(l1g, 1, dims=[1, 0]), dims=[1,]))

        # local1_emb = F.tanh(self.embLocal1(local1_emb.int()))
        # local1_emb = local1_emb.view(BS, 8, l1h, l1w)
        local0_emb = F.one_hot(local0_emb.long(), 8)
        local0_emb = local0_emb.view(BS, 8, l0h, l0w).float()
        local0_emb = F.pad(local0_emb, (1, 1, 1, 1))
        local0_emb = F.relu(self.l0_conv_1(local0_emb))
        local0_emb = F.pad(local0_emb, (1, 1, 1, 1))
        local0_emb = F.relu(self.l0_conv_2(local0_emb))
        local0_emb = local0_emb.view(BS, -1)

        # Local map 5x5
        # l1g = local1_emb.view(5, 5)
        # print(torch.flip(torch.rot90(l1g, 1, dims=[1, 0]), dims=[1,]))

        # local1_emb = F.tanh(self.embLocal1(local1_emb.int()))
        # local1_emb = local1_emb.view(BS, 8, l1h, l1w)
        local1_emb = F.one_hot(local1_emb.long(), 8)
        local1_emb = local1_emb.view(BS, 8, l1h, l1w).float()
        local1_emb = F.pad(local1_emb, (1, 1, 1, 1))
        local1_emb = F.relu(self.l1_conv_1(local1_emb))
        local1_emb = F.pad(local1_emb, (1, 1, 1, 1))
        local1_emb = F.relu(self.l1_conv_2(local1_emb))
        local1_emb = local1_emb.view(BS, -1)

        # Local map 3x3
        # l2g = local2_emb.view(3, 3)
        # print(torch.flip(torch.rot90(l2g, 1, dims=[1, 0]), dims=[1,]))

        # local2_emb = F.tanh(self.embLocal2(local2_emb.int()))
        # local2_emb = local2_emb.view(BS, 8, l2h, l2w)
        local2_emb = F.one_hot(local2_emb.long(), 8)
        local2_emb = local2_emb.view(BS, 8, l2h, l2w).float()
        local2_emb = F.pad(local2_emb, (1, 1, 1, 1))
        local2_emb = F.relu(self.l2_conv_1(local2_emb))
        local2_emb = F.pad(local2_emb, (1, 1, 1, 1))
        local2_emb = F.relu(self.l2_conv_2(local2_emb))
        local2_emb = local2_emb.view(BS, -1)

        properties_emb = F.relu(self.prop_emb_1(properties))
        properties_emb = F.relu(self.prop_emb_2(properties_emb))

        # Concat
        state = torch.concat([local0_emb, local1_emb, local2_emb, properties_emb], dim=1)

        state = F.relu(self.emb_1(state))
        state = F.relu(self.emb_2(state))
        return state

class CriticEmbedding(nn.Module):
    def __init__(self, state_dim, **kwargs):
        super(CriticEmbedding, self).__init__()
        self.state_dim = state_dim

        self.embGlobal = nn.Embedding(5, 8)
        self.embLocal0 = nn.Embedding(5, 8)
        self.embLocal1 = nn.Embedding(5, 8)
        self.embLocal2 = nn.Embedding(5, 8)

        # Global map
        self.g_conv_1 = nn.Conv2d(8, 32, (3, 3), (2, 2))
        self.g_conv_2 = nn.Conv2d(32, 64, (3, 3), (2, 2))

        # Local map 0
        self.l0_conv_1 = nn.Conv2d(8, 32, (3, 3), (1, 1))
        self.l0_conv_2 = nn.Conv2d(32, 64, (3, 3), (1, 1))

        # Local map 1
        self.l1_conv_1 = nn.Conv2d(8, 32, (3, 3), (1, 1))
        self.l1_conv_2 = nn.Conv2d(32, 64, (3, 3), (1, 1))

        # Local map 2
        self.l2_conv_1 = nn.Conv2d(8, 32, (3, 3), (1, 1))
        self.l2_conv_2 = nn.Conv2d(32, 64, (3, 3), (1, 1))

        # Properties emb
        self.prop_emb_1 = nn.Linear(6, 256)
        self.prop_emb_2 = nn.Linear(256, 256)

        self.emb_1 = nn.Linear(7616, 256)
        self.emb_2 = nn.Linear(256, 256)

        self.output_dim = 256

    def forward(self, state):
        try:
            BS, _ = state.shape
        except Exception as e:
            _ = state.shape
            BS = 1
        gw = gh = 20
        l0w = l0h = 9
        l1w = l1h = 5
        l2w = l2h = 3
        prop = 6

        # global_emb, local1_emb, local2_emb, properties = torch.split(state, [gw*gh, l1h*l1w, l2h*l2w, prop], dim=1)

        local0_emb, local1_emb, local2_emb, properties = torch.split(state, [l0w * l0h, l1h * l1w, l2h * l2w, prop],
                                                                     dim=1)

        # Global map 10x10
        # pg = global_emb.view(20, 20)
        # print(torch.flip(torch.rot90(pg, 1, dims=[1, 0]), dims=[1,]))

        # global_emb = F.tanh(self.embGlobal(global_emb.int()))
        # global_emb = global_emb.view(BS, 8, gh, gw)
        # global_emb = F.one_hot(global_emb.long(), 8)
        # global_emb = global_emb.view(BS, 8, gh, gw).float()
        #
        # global_emb = F.pad(global_emb, (1, 1, 1, 1))
        # global_emb = F.relu(self.g_conv_1(global_emb))
        # global_emb = F.pad(global_emb, (1, 1, 1, 1))
        # global_emb = F.relu(self.g_conv_2(global_emb))
        # global_emb = global_emb.view(BS, -1)

        # Local map 9x9
        # l1g = local1_emb.view(5, 5)
        # print(torch.flip(torch.rot90(l1g, 1, dims=[1, 0]), dims=[1,]))

        # local1_emb = F.tanh(self.embLocal1(local1_emb.int()))
        # local1_emb = local1_emb.view(BS, 8, l1h, l1w)
        local0_emb = F.one_hot(local0_emb.long(), 8)
        local0_emb = local0_emb.view(BS, 8, l0h, l0w).float()
        local0_emb = F.pad(local0_emb, (1, 1, 1, 1))
        local0_emb = F.relu(self.l0_conv_1(local0_emb))
        local0_emb = F.pad(local0_emb, (1, 1, 1, 1))
        local0_emb = F.relu(self.l0_conv_2(local0_emb))
        local0_emb = local0_emb.view(BS, -1)

        # Local map 5x5
        # l1g = local1_emb.view(5, 5)
        # print(torch.flip(torch.rot90(l1g, 1, dims=[1, 0]), dims=[1,]))

        # local1_emb = F.tanh(self.embLocal1(local1_emb.int()))
        # local1_emb = local1_emb.view(BS, 8, l1h, l1w)
        local1_emb = F.one_hot(local1_emb.long(), 8)
        local1_emb = local1_emb.view(BS, 8, l1h, l1w).float()
        local1_emb = F.pad(local1_emb, (1, 1, 1, 1))
        local1_emb = F.relu(self.l1_conv_1(local1_emb))
        local1_emb = F.pad(local1_emb, (1, 1, 1, 1))
        local1_emb = F.relu(self.l1_conv_2(local1_emb))
        local1_emb = local1_emb.view(BS, -1)

        # Local map 3x3
        # l2g = local2_emb.view(3, 3)
        # print(torch.flip(torch.rot90(l2g, 1, dims=[1, 0]), dims=[1,]))

        # local2_emb = F.tanh(self.embLocal2(local2_emb.int()))
        # local2_emb = local2_emb.view(BS, 8, l2h, l2w)
        local2_emb = F.one_hot(local2_emb.long(), 8)
        local2_emb = local2_emb.view(BS, 8, l2h, l2w).float()
        local2_emb = F.pad(local2_emb, (1, 1, 1, 1))
        local2_emb = F.relu(self.l2_conv_1(local2_emb))
        local2_emb = F.pad(local2_emb, (1, 1, 1, 1))
        local2_emb = F.relu(self.l2_conv_2(local2_emb))
        local2_emb = local2_emb.view(BS, -1)

        properties_emb = F.relu(self.prop_emb_1(properties))
        properties_emb = F.relu(self.prop_emb_2(properties_emb))

        # Concat
        state = torch.concat([local0_emb, local1_emb, local2_emb, properties_emb], dim=1)

        state = F.relu(self.emb_1(state))
        state = F.relu(self.emb_2(state))
        return state
