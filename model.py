import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Self-Attention ───────────────────────────────────────────────────────────
class SelfAttention(nn.Module):
    """
    Permite modelului sa vada relatii intre parti indepartate ale imaginii.
    Ex: ochiul stang si cel drept sa fie aliniate corect.
    """
    def __init__(self, ch):
        super().__init__()
        self.query = nn.Conv2d(ch, ch // 8, 1)
        self.key   = nn.Conv2d(ch, ch // 8, 1)
        self.value = nn.Conv2d(ch, ch, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.norm  = nn.GroupNorm(8, ch)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.query(h).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(h).view(B, -1, H * W)
        attn = F.softmax(torch.bmm(q, k) / (C ** 0.5), dim=-1)
        v = self.value(h).view(B, -1, H * W)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return x + self.gamma * out


# ─── Generator Block ──────────────────────────────────────────────────────────
class GenBlock(nn.Module):
    """Upsample 2x cu Spectral Norm + residual connection."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.utils.spectral_norm(
                nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False)
            ),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.utils.spectral_norm(
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
            ),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        self.skip = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, bias=False)),
        )

    def forward(self, x):
        return self.block(x) + self.skip(x)


# ─── Generator ───────────────────────────────────────────────────────────────
class Generator(nn.Module):
    """
    Z (128 dim) → imagine RGB 512x512
    4x4 → 8 → 16 → 32 → 64 → 128 → 256 → 512
    Self-Attention la 64x64 si 128x128
    """
    def __init__(self, z_dim=128, base_ch=64):
        super().__init__()
        ch = base_ch
        self.fc = nn.utils.spectral_norm(nn.Linear(z_dim, ch * 16 * 4 * 4))
        self.base_ch = base_ch

        self.blocks = nn.ModuleList([
            GenBlock(ch * 16, ch * 16),
            GenBlock(ch * 16, ch * 8),
            GenBlock(ch * 8,  ch * 8),
            GenBlock(ch * 8,  ch * 4),
            GenBlock(ch * 4,  ch * 2),
            GenBlock(ch * 2,  ch),
            GenBlock(ch,      ch // 2),
        ])

        self.attn64  = SelfAttention(ch * 4)

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
            if i == 3: x = self.attn64(x)
        return self.out(x)


# ─── Discriminator Block ──────────────────────────────────────────────────────
class DiscBlock(nn.Module):
    """Downsample 2x cu Spectral Norm + residual."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False)),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.LeakyReLU(0.2, True),
            nn.utils.spectral_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.LeakyReLU(0.2, True),
        )
        self.skip = nn.Sequential(
            nn.AvgPool2d(2),
            nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, bias=False)),
        )

    def forward(self, x):
        return self.block(x) + self.skip(x)


# ─── Discriminator ───────────────────────────────────────────────────────────
class Discriminator(nn.Module):
    """
    Imagine RGB 512x512 → scor real/fake
    Self-Attention la 64x64 si 32x32
    """
    def __init__(self, base_ch=64):
        super().__init__()
        ch = base_ch
        self.blocks = nn.ModuleList([
            DiscBlock(3,       ch),
            DiscBlock(ch,      ch * 2),
            DiscBlock(ch * 2,  ch * 4),
            DiscBlock(ch * 4,  ch * 8),
            DiscBlock(ch * 8,  ch * 16),
            DiscBlock(ch * 16, ch * 16),
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
