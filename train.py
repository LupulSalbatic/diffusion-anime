import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image
from torch.cuda.amp import autocast, GradScaler
import subprocess
import shutil
from model import UNet

# ─── CONFIG ──────────────────────────────────────────────────────────────────
IMAGE_SIZE    = 512
BATCH_SIZE    = 4          # A4000 16GB — mareste la 6-8 daca nu ai OOM
NUM_EPOCHS    = 500
LEARNING_RATE = 1e-4
T_STEPS       = 1000       # pasi de difuzie
BETA_START    = 1e-4
BETA_END      = 0.02
SAVE_EVERY    = 10         # salveaza checkpoint la fiecare X epoci
SAMPLE_EVERY  = 5          # genereaza sample-uri la fiecare X epoci
NUM_SAMPLES   = 16

DATASET_PATH  = "./dataset"
OUTPUT_PATH   = "./outputs"
CKPT_PATH     = "./checkpoints"
GITHUB_REPO   = "https://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@github.com/LupulSalbatic/diffusion-anime.git"  # <- schimba asta
GITHUB_FOLDER = "./github_repo"

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(CKPT_PATH, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {DEVICE}")
print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ─── DDPM NOISE SCHEDULE ─────────────────────────────────────────────────────
class DDPM:
    def __init__(self, T=T_STEPS, beta_start=BETA_START, beta_end=BETA_END, device=DEVICE):
        self.T = T
        self.device = device

        betas = torch.linspace(beta_start, beta_end, T, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.betas = betas
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)

    def q_sample(self, x0, t, noise=None):
        """Adauga zgomot la imagine (forward process)"""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha * x0 + sqrt_one_minus * noise, noise

    @torch.no_grad()
    def p_sample(self, model, x, t_idx):
        """Un pas de denoising (reverse process)"""
        t = torch.full((x.shape[0],), t_idx, device=self.device, dtype=torch.long)
        pred_noise = model(x, t)

        beta_t = self.betas[t_idx]
        sqrt_recip = self.sqrt_recip_alphas[t_idx]
        sqrt_recipm1 = self.sqrt_recipm1_alphas_cumprod[t_idx]

        mean = sqrt_recip * (x - beta_t * sqrt_recipm1 * pred_noise)

        if t_idx == 0:
            return mean
        noise = torch.randn_like(x)
        var = self.posterior_variance[t_idx]
        return mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, model, n=NUM_SAMPLES, img_size=IMAGE_SIZE):
        """Genereaza n imagini de la zero"""
        model.eval()
        x = torch.randn(n, 3, img_size, img_size, device=self.device)
        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t)
            if t % 100 == 0:
                print(f"  Sampling step {self.T - t}/{self.T}", end="\r")
        model.train()
        return (x.clamp(-1, 1) + 1) / 2  # [0, 1]


# ─── DATASET ─────────────────────────────────────────────────────────────────
def get_dataloader():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [-1, 1]
    ])
    dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
    print(f"[INFO] Dataset: {len(dataset)} imagini")
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )


# ─── GITHUB SAVE ─────────────────────────────────────────────────────────────
def save_to_github(epoch, ckpt_file):
    """Push checkpoint pe GitHub"""
    try:
        if not os.path.exists(GITHUB_FOLDER):
            subprocess.run(["git", "clone", GITHUB_REPO, GITHUB_FOLDER], check=True)

        dest = os.path.join(GITHUB_FOLDER, "checkpoints", f"epoch_{epoch}.pth")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy(ckpt_file, dest)

        subprocess.run(["git", "-C", GITHUB_FOLDER, "add", "."], check=True)
        subprocess.run(["git", "-C", GITHUB_FOLDER, "commit", "-m", f"checkpoint epoch {epoch}"], check=True)
        subprocess.run(["git", "-C", GITHUB_FOLDER, "push"], check=True)
        print(f"[GitHub] Checkpoint epoch {epoch} salvat!")
    except Exception as e:
        print(f"[GitHub] EROARE: {e}")


# ─── TRAINING ────────────────────────────────────────────────────────────────
def train():
    model = UNet(
        in_ch=3,
        base_ch=128,
        ch_mult=(1, 2, 3, 4, 4),
        num_res_blocks=3,
        attn_resolutions=(32, 16, 8),
        dropout=0.1,
        image_size=IMAGE_SIZE,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[INFO] Parametri model: {total_params:.1f}M")

    ddpm      = DDPM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler    = GradScaler()
    loader    = get_dataloader()

    # Resume daca exista checkpoint
    start_epoch = 0
    latest_ckpt = os.path.join(CKPT_PATH, "latest.pth")
    if os.path.exists(latest_ckpt):
        ckpt = torch.load(latest_ckpt, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[INFO] Resumed de la epoca {start_epoch}")

    print(f"[INFO] Incep antrenamentul de la epoca {start_epoch}...")

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        total_loss = 0

        for step, (imgs, _) in enumerate(loader):
            imgs = imgs.to(DEVICE)
            t    = torch.randint(0, ddpm.T, (imgs.shape[0],), device=DEVICE).long()

            with autocast():
                noisy_imgs, noise = ddpm.q_sample(imgs, t)
                pred_noise        = model(noisy_imgs, t)
                loss              = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if step % 50 == 0:
                print(f"Epoca [{epoch}/{NUM_EPOCHS}] Step [{step}/{len(loader)}] Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        scheduler.step()
        print(f"\n[Epoca {epoch}] Loss mediu: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}\n")

        # Sample-uri vizuale
        if epoch % SAMPLE_EVERY == 0:
            samples = ddpm.sample(model, n=NUM_SAMPLES)
            save_image(samples, os.path.join(OUTPUT_PATH, f"sample_epoch_{epoch}.png"), nrow=4)
            print(f"[INFO] Sample salvat: sample_epoch_{epoch}.png")

        # Checkpoint local
        if epoch % SAVE_EVERY == 0:
            ckpt_file = os.path.join(CKPT_PATH, f"epoch_{epoch}.pth")
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss":      avg_loss,
            }, ckpt_file)
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss":      avg_loss,
            }, latest_ckpt)
            print(f"[INFO] Checkpoint salvat: epoch_{epoch}.pth")
            save_to_github(epoch, ckpt_file)


if __name__ == "__main__":
    train()
