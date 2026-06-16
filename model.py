import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─── Sinusoidal Time Embedding ───────────────────────────────────────────────
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# ─── Residual Block cu Time Embedding ────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch * 2)
        )
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.block1(x)
        scale, shift = self.time_mlp(t).chunk(2, dim=-1)
        h = h * (scale[..., None, None] + 1) + shift[..., None, None]
        h = self.block2(h)
        return h + self.shortcut(x)


# ─── Self-Attention Block ─────────────────────────────────────────────────────
class SelfAttention(nn.Module):
    def __init__(self, ch, num_heads=8):
        super().__init__()
        self.norm = nn.GroupNorm(32, ch)
        self.attn = nn.MultiheadAttention(ch, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).permute(0, 2, 1)
        h, _ = self.attn(h, h, h)
        return x + h.permute(0, 2, 1).view(B, C, H, W)


# ─── Downsample / Upsample ────────────────────────────────────────────────────
class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


# ─── U-Net complet ────────────────────────────────────────────────────────────
class UNet(nn.Module):
    """
    U-Net complex pentru DDPM 512x512 anime faces.
    Canale: [128, 256, 384, 512, 512]
    Attention la rezolutiile: 32, 16, 8
    """
    def __init__(
        self,
        in_ch=3,
        base_ch=128,
        ch_mult=(1, 2, 3, 4, 4),
        num_res_blocks=3,
        attn_resolutions=(32, 16, 8),
        dropout=0.1,
        image_size=512,
    ):
        super().__init__()
        time_dim = base_ch * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_ch),
            nn.Linear(base_ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        channels = [base_ch * m for m in ch_mult]
        self.init_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # ── Encoder ──
        self.downs = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        in_ch_cur = base_ch
        res = image_size
        self.skip_channels = []

        for i, out_ch in enumerate(channels):
            blocks = nn.ModuleList()
            attns = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(in_ch_cur, out_ch, time_dim, dropout))
                attns.append(SelfAttention(out_ch) if res in attn_resolutions else nn.Identity())
                in_ch_cur = out_ch
            self.skip_channels.append(out_ch)
            self.downs.append(blocks)
            self.down_attn.append(attns)
            if i != len(channels) - 1:
                self.downs.append(Downsample(out_ch))
                self.down_attn.append(None)
                res //= 2

        # ── Bottleneck ──
        mid_ch = channels[-1]
        self.mid_block1 = ResBlock(mid_ch, mid_ch, time_dim, dropout)
        self.mid_attn   = SelfAttention(mid_ch)
        self.mid_block2 = ResBlock(mid_ch, mid_ch, time_dim, dropout)

        # ── Decoder ──
        self.ups = nn.ModuleList()
        self.up_attn = nn.ModuleList()
        rev_channels = list(reversed(channels))
        rev_skip     = list(reversed(self.skip_channels))

        for i, out_ch in enumerate(rev_channels):
            blocks = nn.ModuleList()
            attns = nn.ModuleList()
            skip_ch = rev_skip[i]
            for j in range(num_res_blocks + 1):
                extra = skip_ch if j == 0 else 0
                blocks.append(ResBlock(in_ch_cur + extra, out_ch, time_dim, dropout))
                attns.append(SelfAttention(out_ch) if res in attn_resolutions else nn.Identity())
                in_ch_cur = out_ch
            self.ups.append(blocks)
            self.up_attn.append(attns)
            if i != len(rev_channels) - 1:
                self.ups.append(Upsample(out_ch))
                self.up_attn.append(None)
                res *= 2

        self.out = nn.Sequential(
            nn.GroupNorm(32, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, 3, 3, padding=1),
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x = self.init_conv(x)

        skips = []
        idx = 0
        for layer in self.downs:
            if isinstance(layer, Downsample):
                x = layer(x)
            else:
                attn_list = self.down_attn[idx]
                for blk, atn in zip(layer, attn_list):
                    x = blk(x, t_emb)
                    x = atn(x) if not isinstance(atn, nn.Identity) else x
                skips.append(x)
            idx += 1

        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        idx = 0
        for layer in self.ups:
            if isinstance(layer, Upsample):
                x = layer(x)
            else:
                attn_list = self.up_attn[idx]
                for j, (blk, atn) in enumerate(zip(layer, attn_list)):
                    if j == 0:
                        x = torch.cat([x, skips.pop()], dim=1)
                    x = blk(x, t_emb)
                    x = atn(x) if not isinstance(atn, nn.Identity) else x
            idx += 1

        return self.out(x)
