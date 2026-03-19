import torch
import torch.nn.functional as F
from torch import nn

class C_encoder_block_next(nn.Module):
    """
    ResNeXt style residual block
    """
    def __init__(self, in_channel, hidden_size, out_channel, cardinality=32): # Default cardinality is set to 32
        super().__init__()
        self.act = nn.ELU()
        self.cardinality = cardinality
        
        # Dimension of each branch
        hidden_card_dim = hidden_size // cardinality

        
        # Create multiple branches
        self.enc_branches = nn.ModuleList()
        for _ in range(cardinality):
            branch = nn.Sequential(
                nn.Linear(in_channel, hidden_card_dim),
                nn.ELU(),
                nn.Linear(hidden_card_dim, out_channel),
            )
            self.enc_branches.append(branch)
    
    def forward(self, x):
        # Apply each branch separately
        branch_outputs = [branch(x) for branch in self.enc_branches]

        y = torch.stack(branch_outputs, dim=1)
        
        # Sum along the stacked dimension
        y = torch.sum(y, dim=1)
        
        # Add the residual connection
        y += x
        
        # Apply the activation function
        x = self.act(y)
        
        return x

class C_encoder_block(nn.Module):
    """
    residual block
    """
    def __init__(self, in_channel, hidden_size, out_channel):
        super().__init__()
        self.act = nn.ELU()
        # self.act = nn.GELU()
        self.enc = nn.Sequential(
            nn.Linear(in_channel, hidden_size),
            self.act,
            nn.Linear(hidden_size, out_channel),
        )
    
    def forward(self, x):
        y = self.enc(x)
        y += x
        x = self.act(y)
        return x
    
class UnetBaiscBlock(nn.Module):
    def __init__(self, in_channel, hidden_size, out_channel):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_channel, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, out_channel),
            nn.ELU()
        )

    def forward(self, x):
        return self.linear(x)
