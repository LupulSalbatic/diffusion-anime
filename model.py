import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Self-Attention cu Flash Attention ───────────────────────────────────────
class SelfAttention(nn.Module):
    """
    Self-Attention optimizat cu scaled dot-product.
    Folosit doar la 64x64 pentru a evita OOM la rezolutii mari.
    """
    def __init__(self, ch):
        super().__init__()
        self.norm  = nn.GroupNorm(8, ch)
        self.query = nn.Conv2d(ch, ch // 8, 1, bias=False)
        self.key   = nn.Conv2d(ch, ch // 8, 1, bias=False)
        self.value = nn.Conv2d(ch, ch, 1, bias=False)
        self.proj  = nn.utils.spectral_norm(nn.Conv2d(ch, ch, 1, bias=False))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.query(h).view(B, C // 8, H * W).permute(0, 2, 1)
        k = self.key(h).view(B, C // 8, H * W)
        v = self.value(h).view(B, C, H * W).permute(0, 2, 1)
        # Scaled dot-product attention
        scale = (C // 8) ** -0.5
        attn = F.softmax(torch.bmm(q, k) * scale, dim=-1)
        out = torch.bmm(attn, v).permute(0, 2, 1).view(B, C, H, W)
        return x + self.gamma * self.proj(out)


# ─── Generator Block ──────────────────────────────────────────────────────────
class GenBlock(nn.Module):
    """
    Upsample 2x cu:
    - Spectral Norm pentru stabilitate
    - Pixel Shuffle pentru upscaling de calitate
    - Residual connection pentru gradient flow
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.utils.spectral_norm(
                nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False)
            ),
            nn.BatchNorm2d(out_ch, momentum=0.01),
            nn.GELU(),
            nn.utils.spectral_norm(
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
            ),
            nn.BatchNorm2d(out_ch, momentum=0.01),
            nn.GELU(),
        )
        self.skip = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, bias=False)),
        )

    def forward(self, x):
        return self.block(x) + self.skip(x)


# ─── Generator ───────────────────────────────────────────────────────────────
class Generator(nn.Module):
    """
    Z (128 dim) → RGB 512x512
    Arhitectura: 4 → 8 → 16 → 32 → 64[SA] → 128 → 256 → 512
    """
    def __init__(self, z_dim=128, base_ch=64):
        super().__init__()
        ch = base_ch
        self.base_ch = ch

        self.fc = nn.utils.spectral_norm(nn.Linear(z_dim, ch * 16 * 4 * 4))

        self.blocks = nn.ModuleList([
            GenBlock(ch * 16, ch * 16),  # 4→8
            GenBlock(ch * 16, ch * 8),   # 8→16
            GenBlock(ch * 8,  ch * 8),   # 16→32
            GenBlock(ch * 8,  ch * 4),   # 32→64
            GenBlock(ch * 4,  ch * 2),   # 64→128
            GenBlock(ch * 2,  ch),        # 128→256
            GenBlock(ch,      ch // 2),   # 256→512
        ])

        # Self-Attention doar la 64x64 (dupa blocul 3)
        self.attn64 = SelfAttention(ch * 4)

        self.out = nn.Sequential(
            nn.GroupNorm(8, ch // 2),
            nn.GELU(),
            nn.utils.spectral_norm(nn.Conv2d(ch // 2, 3, 3, padding=1)),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z).view(z.shape[0], self.base_ch * 16, 4, 4)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == 3:
                x = self.attn64(x)
        return self.out(x)


# ─── Discriminator Block ──────────────────────────────────────────────────────
class DiscBlock(nn.Module):
    """
    Downsample 2x cu:
    - Spectral Norm pe toate layerele
    - GroupNorm in loc de BatchNorm (mai stabil pentru discriminator)
    - Residual connection
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        groups = min(8, out_ch)
        self.block = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False)),
            nn.GroupNorm(groups, out_ch),
            nn.LeakyReLU(0.2, True),
            nn.utils.spectral_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)),
            nn.GroupNorm(groups, out_ch),
            nn.LeakyReLU(0.2, True),
        )
        self.skip = nn.Sequential(
            nn.AvgPool2d(2, ceil_mode=True),
            nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, bias=False)),
        )

    def forward(self, x):
        return self.block(x) + self.skip(x)


# ─── Discriminator ───────────────────────────────────────────────────────────
class Discriminator(nn.Module):
    """
    RGB 512x512 → scor real/fake
    Self-Attention la 64x64 si 32x32
    """
    def __init__(self, base_ch=64):
        super().__init__()
        ch = base_ch
        self.blocks = nn.ModuleList([
            DiscBlock(3,       ch),       # 512→256
            DiscBlock(ch,      ch * 2),   # 256→128
            DiscBlock(ch * 2,  ch * 4),   # 128→64
            DiscBlock(ch * 4,  ch * 8),   # 64→32
            DiscBlock(ch * 8,  ch * 16),  # 32→16
            DiscBlock(ch * 16, ch * 16),  # 16→8
        ])

        self.attn64 = SelfAttention(ch * 4)
        self.attn32 = SelfAttention(ch * 8)

        self.out = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(ch * 16, ch * 4, 3, padding=1)),
            nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(ch * 4, 1)),
        )

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == 2: x = self.attn64(x)
            if i == 3: x = self.attn32(x)
        return self.out(x).squeeze(1)
