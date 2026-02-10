from __future__ import annotations

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, in_channels, H, W) --> (batch_size, n_patches, embed_dim)
        # flatten(2) will flatten from and after axis=2.
        return self.proj(x).flatten(2).transpose(1, 2)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim**-0.5

        assert embed_dim % n_heads == 0, "Embedding dim must be divisible by n_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.fc_out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, n_tokens, embed_dim) --> (batch_size, n_tokens, embed_dim)
        batch_size, n_tokens, _ = x.shape

        # Linear projection to Q, K, V
        head_dim = self.embed_dim // self.n_heads
        qkv = self.qkv(x).reshape(batch_size, n_tokens, 3, self.n_heads, head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (batch_size, n_heads, n_tokens, head_dim)

        # Scaled Dot-Product Attention
        # attn.shape == (batch_size, n_heads, n_tokens, n_tokens)
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(batch_size, n_tokens, self.embed_dim)
        return self.fc_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connections
        x = x + self.attn(self.layer_norm(x))
        x = x + self.ff(self.layer_norm(x))
        return x


class SimpleViT(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        n_classes: int = 10,
        embed_dim: int = 128,
        depth: int = 6,
        n_heads: int = 8,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        n_patches = (img_size // patch_size) ** 2
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)

        # Learnable CLS token and Position Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + n_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer Encoder Blocks
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, n_heads, hidden_dim, dropout) for _ in range(depth)]
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.patch_embed(x)  # --> (batch_size, n_patches, embed_dim)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # --> (batch_size, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, 1 + n_patches, embed_dim)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.layer_norm(x)
        cls_token_final = x[:, 0]
        return self.head(cls_token_final)


# --- Usage Example ---
if __name__ == "__main__":
    torch.manual_seed(42)
    # Create the model specialized for CIFAR-10
    model = SimpleViT(
        img_size=32,
        patch_size=4,  # Critical for small CIFAR images
        n_classes=10,
        embed_dim=128,  # Smaller dim for faster training on small data
        depth=6,  # Fewer layers than ViT-Base
        n_heads=4,
    )

    # Random CIFAR-10 batch: [Batch=2, Channels=3, Height=32, Width=32]
    img = torch.randn(2, 3, 32, 32)

    output = model(img)
    print(f"Input shape: {img.shape}")
    print(f"Output shape: {output.shape}")  # Should be [2, 10]
    print(output)

    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {params / 1e6:.2f}M")
