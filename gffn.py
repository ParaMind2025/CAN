import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import DropPath
from clifford_thrust import LayerNorm2d, CliffordInteraction        


class gFFN(nn.Module):
    """
    Geometric Feed-Forward Network (gFFN)
    
    A standalone module that replaces standard MLP with Clifford Geometric Interactions.
    Can be used as a plug-and-play component in various backbones.
    
    Args:
        dim (int): Feature dimension.
        cli_mode (str): 'full', 'wedge', 'inner'.
        ctx_mode (str): 'diff' (Laplacian) or 'abs'.
        gffn_mode (str): 
            - 'l': Local only (Conv context).
            - 'g': Global only (GlobalAvg context).
            - 'h': Hybrid (Local + Beta * Global).
        shifts (list): Shifts for rolling interaction.
    """
    def __init__(self, dim, cli_mode='full', ctx_mode='diff', gffn_mode='h', 
                 shifts=[1, 2, 4], drop_path=0., init_values=1e-5, enable_cuda=False):
        super().__init__()
        
        self.dim = dim
        self.cli_mode = cli_mode  
        self.ctx_mode = ctx_mode 
        self.gffn_mode = gffn_mode.lower()
        
        self.norm = LayerNorm2d(dim)
        InteractionLayer = CliffordInteraction 
            
        self.get_state = nn.Conv2d(dim, dim, kernel_size=1)

        if self.gffn_mode in ['l', 'h']:
            self.get_context_local = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
                nn.BatchNorm2d(dim),
                nn.SiLU()            
            )
            self.inter_local = InteractionLayer(dim, cli_mode=cli_mode, ctx_mode=ctx_mode, shifts=shifts)
        else:
            self.get_context_local = None
            self.inter_local = None

        if self.gffn_mode in ['g', 'h']:
            self.inter_global = InteractionLayer(dim, cli_mode='full', ctx_mode='diff', shifts=[1, 2])
        else:
            self.inter_global = None

        if self.gffn_mode == 'h':
            self.beta = nn.Parameter(torch.tensor([0.5]), requires_grad=True)

        self.gate_fc = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.gamma = nn.Parameter(init_values * torch.ones((1, dim, 1, 1)), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()        
        
    def forward(self, x):
        shortcut = x
        x_ln = self.norm(x) 
        z_state = self.get_state(x_ln)
        g_feat_total = None
        
        if self.get_context_local is not None:
            z_ctx_local = self.get_context_local(x_ln)
            g_feat_local = self.inter_local(z_state, z_ctx_local)
        
        if self.inter_global is not None:
            z_ctx_global = x_ln.mean(dim=[-2, -1], keepdim=True).expand_as(x_ln)                
            g_feat_global = self.inter_global(z_state, z_ctx_global)

        if self.gffn_mode == 'l':
            g_feat_total = g_feat_local
        elif self.gffn_mode == 'g':
            g_feat_total = g_feat_global
        elif self.gffn_mode == 'h':
            g_feat_total = g_feat_local + self.beta * g_feat_global
            
        combined = torch.cat([x_ln, g_feat_total], dim=1)
        gate = torch.sigmoid(self.gate_fc(combined))
        
        x_mixed = F.silu(x_ln) + gate * g_feat_total
        x = shortcut + self.drop_path(self.gamma * x_mixed)
        
        return x    