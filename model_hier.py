import torch
import torch.nn as nn

from utils import DropPath
from clifford_thrust import LayerNorm2d, CliffordInteraction


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
    def __init__(self, dim, cli_mode="full", ctx_mode="diff", shifts=(1, 2)):
        super().__init__()
        self.dim = dim
        self.cli_mode = cli_mode
        self.ctx_mode = ctx_mode
        self.act = nn.SiLU()
        self.shifts = [s for s in shifts if s < dim]
        self.branch_dim = dim * len(self.shifts)

        if self.cli_mode == "full":
            cat_dim = self.branch_dim * 2
        elif self.cli_mode in ("wedge", "inner"):
            cat_dim = self.branch_dim
        else:
            raise ValueError(f"Invalid cli_mode: {cli_mode}")
        self.proj_ = nn.Conv2d(cat_dim, dim, kernel_size=1)

    def forward(self, z1, z2):
        if self.ctx_mode == "diff":
            c = z2 - z1
        elif self.ctx_mode == "abs":
            c = z2
        else:
            raise ValueError(f"Invalid ctx_mode: {self.ctx_mode}")

        feats = []
        for s in self.shifts:
            c_shifted = torch.roll(c, shifts=s, dims=1)
            if self.cli_mode in ("wedge", "full"):
                z1_shifted = torch.roll(z1, shifts=s, dims=1)
                wedge = z1 * c_shifted - c * z1_shifted
                feats.append(wedge)
            if self.cli_mode in ("inner", "full"):
                inner = self.act(z1 * c_shifted)
                feats.append(inner)
        x_ = torch.cat(feats, dim=1)
        out = self.proj_(x_)
        return out

    
class MultiScaleContext(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw3_d1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.dw3_d2 = nn.Conv2d(dim, dim, kernel_size=3, padding=2, dilation=2, groups=dim, bias=False)
        self.dw3_d3 = nn.Conv2d(dim, dim, kernel_size=3, padding=3, dilation=3, groups=dim, bias=False)
        self.fuse = nn.Conv2d(dim * 3, dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.SiLU()

    def forward(self, x):
        x1 = self.dw3_d1(x)
        x2 = self.dw3_d2(x)
        x3 = self.dw3_d3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.fuse(x)
        x = self.bn(x)
        x = self.act(x)
        return x
    

class CliffordAlgebraBlock(nn.Module):
    def __init__(
        self,
        dim,
        cli_mode="full",
        ctx_mode="diff",
        shifts=(1, 2),
        drop_path=0.1,
        init_values=1e-5,
        enable_cuda=False,
    ):
        super().__init__()
        self.get_state = nn.Conv2d(dim, dim, kernel_size=1)
        self.get_context = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
        )
        # self.get_context = MultiScaleContext(dim)
        if enable_cuda:
            self.norm = LayerNorm2d(dim)
            self.clifford_interaction = CliffordInteraction(dim, cli_mode, ctx_mode, shifts)
        else:
            self.norm = LayerNorm2d_PyTorch(dim)
            self.clifford_interaction = CliffordInteraction_PyTorch(dim, cli_mode, ctx_mode, shifts)

        self.act = nn.SiLU()
        self.gate_fc = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), init_values))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x_ln = self.norm(x)
        z_state = self.get_state(x_ln)
        z_context = self.get_context(x_ln)
        g_feat = self.clifford_interaction(z_state, z_context)
        gate = torch.sigmoid(self.gate_fc(torch.cat([x_ln, g_feat], dim=1)))
        x_mixed = self.act(x_ln) + gate * g_feat
        x_mixed = self.gamma * x_mixed
        x = shortcut + self.drop_path(x_mixed)
        return x


class GeometricStem(nn.Module):
    def __init__(self, in_chans=3, embed_dim=128, patch_size=4):
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


class StageDownsample(nn.Module):
    def __init__(self, in_dim, out_dim, mode="avgpool"):
        super().__init__()
        if mode == "avgpool":
            self.down = nn.AvgPool2d(kernel_size=2, stride=2)
            self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1) if in_dim != out_dim else nn.Identity()
        elif mode == "conv":
            self.down = nn.Sequential(
                nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=2, padding=1, groups=in_dim, bias=False),
                nn.BatchNorm2d(in_dim),
                nn.SiLU(),
            )
            self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1) if in_dim != out_dim else nn.Identity()
        elif mode == "patch":
            self.down = nn.Identity()
            self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)
        else:
            raise ValueError(f"Invalid downsample mode: {mode}")

    def forward(self, x):
        x = self.down(x)
        x = self.proj(x)
        return x


class HierarchicalCliffordNet(nn.Module):
    """
    stage_depths: number of blocks per stage, e.g. (2, 2, 4)
    stage_dims:
        - None + dim_policy='double': [embed_dim, 2*embed_dim, 4*embed_dim, ...]
        - None + dim_policy='constant': [embed_dim] * num_stages
        - explicit list/tuple of per-stage dims
    downsample_mode: 'avgpool', 'conv', or 'patch'
    """

    def __init__(
        self,
        num_classes=100,
        in_chans=3,
        patch_size=1,
        embed_dim=32,
        cli_mode="full",
        ctx_mode="diff",
        shifts=(1,2),
        stage_depths=(3, 4, 5),
        stage_dims=None,
        dim_policy="double",
        downsample_mode="conv",
        drop_path_rate=0.1,
        enable_cuda=False,
    ):
        super().__init__()
        if len(stage_depths) == 0:
            raise ValueError("stage_depths must not be empty")

        if stage_dims is None:
            if dim_policy == "double":
                stage_dims = [embed_dim * (2 ** i) for i in range(len(stage_depths))]
            elif dim_policy == "constant":
                stage_dims = [embed_dim for _ in stage_depths]
            else:
                raise ValueError(f"Invalid dim_policy: {dim_policy}")
        else:
            if len(stage_dims) != len(stage_depths):
                raise ValueError("stage_dims must have the same length as stage_depths")
            stage_dims = list(stage_dims)

        self.stage_depths = list(stage_depths)
        self.stage_dims = stage_dims
        self.patch_embed = GeometricStem(in_chans=in_chans, embed_dim=embed_dim, patch_size=patch_size)

        self.stem_proj = (
            nn.Conv2d(embed_dim, stage_dims[0], kernel_size=1) if embed_dim != stage_dims[0] else nn.Identity()
        )

        total_blocks = sum(stage_depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        dp_idx = 0
        for stage_idx, (depth, dim) in enumerate(zip(stage_depths, stage_dims)):
            blocks = []
            for _ in range(depth):
                blocks.append(
                    CliffordAlgebraBlock(
                        dim=dim,
                        cli_mode=cli_mode,
                        ctx_mode=ctx_mode,
                        shifts=shifts,
                        drop_path=dpr[dp_idx],
                        enable_cuda=enable_cuda,
                    )
                )
                dp_idx += 1
            self.stages.append(nn.Sequential(*blocks))

            if stage_idx < len(stage_depths) - 1:
                self.downsamples.append(
                    StageDownsample(
                        in_dim=stage_dims[stage_idx],
                        out_dim=stage_dims[stage_idx + 1],
                        mode=downsample_mode,
                    )
                )

        self.norm = nn.LayerNorm(stage_dims[-1])
        self.head = nn.Linear(stage_dims[-1], num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.stem_proj(x)
        for stage_idx, stage in enumerate(self.stages):
            x = stage(x)
            if stage_idx < len(self.downsamples):
                x = self.downsamples[stage_idx](x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.mean(dim=[-2, -1])
        x = self.norm(x)
        x = self.head(x)
        return x

