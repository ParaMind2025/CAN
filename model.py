import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import DropPath

try:
    from clifford_thrust import LayerNorm2d, CliffordInteraction
    print("✅ Successfully loaded accelerated Clifford kernels (CUDA).")
    has_acceleration = True
except ImportError:
    print("⚠️ 'clifford_thrust' not found. Falling back to pure PyTorch implementation (Slower).")
    has_acceleration = False

class LayerNorm2d_PyTorch(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):

        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    
class CliffordInteraction_PyTorch(nn.Module):
    """
    Args:
        cli_mode: 'full', 'wedge', 'inner'
        ctx_mode: 
            - 'diff': C = C_local - H (discrete Laplacian)
            - 'abs' : C = C_local
            - 'others': to be added
    """    
    def __init__(self, dim, cli_mode='full', ctx_mode='diff', shifts=[1, 2]):
        super().__init__()
        self.dim = dim
        self.cli_mode = cli_mode  
        self.ctx_mode = ctx_mode 
        self.act = nn.SiLU()
        
        self.shifts = shifts
        self.shifts = [s for s in self.shifts if s < dim]
        
        self.branch_dim = dim * len(self.shifts)
        
        self.proj_ = None
        
        if self.cli_mode == 'full':
            cat_dim = self.branch_dim*2
        elif self.cli_mode in ['wedge', 'inner']:
            cat_dim = self.branch_dim
        else:
            raise ValueError(f"Invalid cli_mode: {cli_mode}")
        self.proj_ = nn.Conv2d(cat_dim, dim, kernel_size=1)    

    def forward(self, z1, z2):
        
        if self.ctx_mode == 'diff':
            C = z2 - z1  
        elif self.ctx_mode == 'abs':
            C = z2
            
        feats = []
        for s in self.shifts:
            C_shifted = torch.roll(C, shifts=s, dims=1)     
            if self.cli_mode in ['wedge', 'full']:
                z1_shifted = torch.roll(z1, shifts=s, dims=1)
                wedge = z1 * C_shifted - C * z1_shifted
                feats.append(wedge)
            if self.cli_mode in ['inner', 'full']:
                inner = self.act(z1 * C_shifted)
                feats.append(inner)
        x_ = torch.cat(feats, dim=1)
        out = self.proj_(x_)
        return out  

class CliffordAlgebraBlock(nn.Module):
    def __init__(self, dim, cli_mode='full', ctx_mode = 'diff', shifts=[1, 2], enable_gFFNG=False, num_heads=1, mlp_ratio=0., drop=0., drop_path=0.1, init_values=1e-5, enable_cuda=False):
        super().__init__()

        self.get_state = nn.Conv2d(dim, dim, kernel_size=1)
        self.get_context_local = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU()            
         )
        self.enable_gFFNG = enable_gFFNG
        if enable_cuda:
            self.norm = LayerNorm2d(dim)
            self.clifford_interaction_local = CliffordInteraction(dim, cli_mode, ctx_mode, shifts)
            if self.enable_gFFNG:
                self.clifford_interaction_global = CliffordInteraction(dim, cli_mode='full', ctx_mode='diff', shifts=[1,2])
        else:
            self.norm = LayerNorm2d_PyTorch(dim)
            self.clifford_interaction_local = CliffordInteraction_PyTorch(dim, cli_mode, ctx_mode, shifts) 
            if self.enable_gFFNG:
                self.clifford_interaction_global = CliffordInteraction_PyTorch(dim, cli_mode='full', ctx_mode='diff', shifts=[1,2])

        self.act = nn.SiLU()
        self.gate_fc = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), init_values))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        shortcut = x
        x_ln = self.norm(x) 
        
        z_state = self.get_state(x_ln)
        z_context_local = self.get_context_local(x_ln)
        g_feat_total = self.clifford_interaction_local(z_state, z_context_local)
        
        if self.enable_gFFNG:
            z_context_global = x_ln.mean(dim=[-2, -1], keepdim=True).expand_as(x_ln)
            g_feat_global = self.clifford_interaction_global(z_state, z_context_global)
            g_feat_total = g_feat_total + g_feat_global
        
        combined = torch.cat([x_ln, g_feat_total], dim=1)
        gate = torch.sigmoid(self.gate_fc(combined))
        x_mixed = F.silu(x_ln) + gate * g_feat_total
        x = shortcut + self.drop_path(self.gamma * x_mixed)
        
        return x    

class GeometricStem(nn.Module):
    def __init__(self, in_chans=3, embed_dim=128, patch_size=2):
        super().__init__()
        
        if patch_size == 1:
            self.proj = nn.Sequential(
                nn.Conv2d(in_chans, embed_dim // 2, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(embed_dim // 2),
                nn.SiLU(),
                nn.Conv2d(embed_dim // 2, embed_dim, 3, stride=1, padding=1, bias=False),
            )
        elif patch_size == 2:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=2, padding=1)
            
        elif patch_size == 4:
            self.proj = nn.Sequential(
                nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(embed_dim // 2),
                nn.SiLU(),
                nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            )
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.norm = nn.BatchNorm2d(embed_dim) 

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x
    
class CliffordNet(nn.Module):
    def __init__(self, num_classes=10, patch_size=4, embed_dim=128, cli_mode='full', ctx_mode='diff', shifts=[1, 2], depth=6, num_heads=1, mlp_ratio=0., drop_rate=0., drop_path_rate=0.1,enable_cuda=False):
        super().__init__()
        
        self.patch_embed = GeometricStem(in_chans=3, embed_dim=embed_dim, patch_size=patch_size)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.blocks = nn.ModuleList([
            CliffordAlgebraBlock(
                dim=embed_dim, 
                cli_mode=cli_mode,
                ctx_mode=ctx_mode,
                shifts=shifts,                
                num_heads=num_heads, 
                drop_path=dpr[i],
                enable_cuda=enable_cuda,
            )
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):

        x = self.patch_embed(x) 
        for block in self.blocks:
            x = block(x) 
        return x

    def forward(self, x):
        
        x = self.forward_features(x)
        x = x.mean(dim=[-2, -1]) 
        x = self.norm(x)
        x = self.head(x)
        return x   