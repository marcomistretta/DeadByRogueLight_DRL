import torch
from torch import nn

# Simple positional embedding for timestep, very similar to CCPT
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, state):
        BS, state_dim = state.shape
        state = state.view(-1, 1)
        d = torch.range(0, self.dim - 1)[None, ...]
        def get_angles(pos, i, d_model):
            angle_rates = 1 / torch.pow(10000., (2. * (i // 2.)) / d_model).to(state.device)
            return pos * angle_rates

        angles = get_angles(state, d, self.dim)
        sins = torch.sin(angles[:,0::2])
        coss = torch.cos(angles[:,1::2])

        sins = torch.unsqueeze(sins, -1)
        coss = torch.unsqueeze(coss, -1)

        embs = torch.cat([sins, coss], dim=2)
        embs = torch.reshape(embs, [BS, state_dim,  self.dim]).float()
        return embs