import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba3 import Mamba3Block, Mamba3Config, RMSNorm

# Optional: Try to import DropPath from timm, fallback to local implementation
try:
    from timm.models.layers import DropPath
except ImportError:
    # Use local implementation of DropPath if timm is not installed
    def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1) 
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    class DropPath(nn.Module):
        def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
            super(DropPath, self).__init__()
            self.drop_prob = drop_prob
            self.scale_by_keep = scale_by_keep

        def forward(self, x):
            return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

class PatchEmbedding(nn.Module):
    """
    Convert 2D Image to 1D Sequence of Patches.
    """
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Use Conv2d for Patchify: Stride = Patch Size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, D, H/p, W/p)
        x = self.proj(x)
        # Flatten: (B, D, H/p, W/p) -> (B, D, L)
        x = x.flatten(2)
        # Transpose: (B, L, D) for Mamba
        x = x.transpose(1, 2)
        return x

class VisionMamba(nn.Module):
    """
    Vision Mamba (Mamba-3 Edition) for CIFAR-10 / ImageNet.
    
    New Features:
    - Supports Mamba-3 Parameters (n_groups, expand, use_conv).
    - Robust Snake Scan implementation using register_buffer.
    """
    def __init__(self, 
                 img_size=32, 
                 patch_size=4, 
                 depth=4, 
                 embed_dim=128, 
                 d_state=64, 
                 d_head=32, 
                 n_groups=1,      # [New] Grouped SSM
                 expand=2,        # [New] Expansion factor (standard is 2)
                 mimo_rank=8,
                 use_conv=True,   # [New] Recommended True for Vision
                 num_classes=10,
                 drop_path_rate=0.0,
                 bidirectional=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.bidirectional = bidirectional
        self.img_size = img_size
        self.patch_size = patch_size
        self.drop_path_rate = drop_path_rate
        
        # 1. Patchify
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        
        # 2. Config Setup (Mapping to Mamba3Config)
        self.config = Mamba3Config(
            d_model=embed_dim, 
            d_state=d_state, 
            d_head=d_head, 
            n_groups=n_groups,
            expand=expand,
            mimo_rank=mimo_rank,
            use_conv=use_conv,  # Pass use_conv to config
            d_conv=4,
            use_parallel_scan=True # Force True for Training
        )
        
        # 3. Backbone Layers
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            if bidirectional:
                # Vim Style: Independent Forward and Backward Blocks
                # Note: This doubles parameters per layer effectively
                self.layers.append(nn.ModuleDict({
                    'fwd': Mamba3Block(self.config),
                    'bwd': Mamba3Block(self.config)
                }))
            else:
                self.layers.append(Mamba3Block(self.config))
        
        # [OPTIMIZATION] Stochastic Depth (DropPath)
        # Linear decay of drop path rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] 
        self.drop_paths = nn.ModuleList([DropPath(dpr[i]) for i in range(depth)])
        
        # [STABILITY FIX] Pre-Norm Residual Loop
        # Add norms for each layer
        self.norms = nn.ModuleList([RMSNorm(embed_dim) for _ in range(depth)])
        self.final_norm = RMSNorm(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes)
        
        # Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 4. Pre-calculate Snake Indices (Register Buffer ensures device compatibility)
        self._init_snake_indices()

    def _init_snake_indices(self):
        """
        Generate indices for Snake Scan (ZigZag) and register as buffer.
        Buffer is saved in state_dict and moves with .cuda() automatically.
        """
        H_grid = W_grid = self.img_size // self.patch_size
        grid = torch.arange(H_grid * W_grid).view(H_grid, W_grid)
        
        # Flip odd rows: 0->Right, 1->Left, 2->Right...
        grid[1::2] = grid[1::2].flip(1)
        
        snake_indices = grid.flatten()
        self.register_buffer('snake_indices', snake_indices)

    def forward(self, x):
        # 1. Patch Embedding & Positional Encoding
        x = self.patch_embed(x) # (B, L, D)
        x = x + self.pos_embed
        
        # 2. Apply Snake Permutation
        # Direct indexing using the registered buffer
        x = x[:, self.snake_indices, :]
        
        # 3. Mamba Blocks
        for i, layer in enumerate(self.layers):
            norm_x = self.norms[i](x)

            if self.bidirectional:
                # A. Forward Path (Scan Grid normally)
                out_fwd = layer['fwd'](norm_x)
                
                # B. Backward Path (Flip sequence dim 1)
                # Note: Mamba is causal, so flipping input makes it process from end to start
                norm_x_rev = norm_x.flip(dims=[1])
                out_bwd = layer['bwd'](norm_x_rev)
                out_bwd = out_bwd.flip(dims=[1]) # Flip back to align with fwd
                
                # C. Fusion
                # Since Mamba3Block is Residual (y = x + SSM(x)),
                # Averaging helps keep variance stable: x_new = x + 0.5(SSM_fwd + SSM_bwd)
                out_combined = (out_fwd + out_bwd) / 2
                x = x + self.drop_paths[i](out_combined)
            else:
                x = x + self.drop_paths[i](layer(norm_x))
                
        # 4. Global Pooling & Classify
        x = self.final_norm(x)
        x = x.mean(dim=1) # Global Average Pooling
        logits = self.head(x)
        
        return logits
