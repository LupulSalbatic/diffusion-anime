import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image
from torch.amp import autocast, GradScaler
import subprocess
import shutil
from model import Generator, Discriminator

# ─── CONFIG ──────────────────────────────────────────────────────────────────
IMAGE_SIZE    = 512
BATCH_SIZE    = 4
NUM_EPOCHS    = 300
LR_G          = 2e-5
LR_D          = 8e-5
Z_DIM         = 128
BASE_CH       = 64
N_CRITIC      = 1
LAMBDA_GP     = 5
SAVE_EVERY    = 1
SAMPLE_EVERY  = 1
NUM_SAMPLES   = 4          # redus la 4 pentru a evita OOM la sampling

DATASET_PATH  = "./dataset"
OUTPUT_PATH   = "./outputs"
CKPT_PATH     = "./checkpoints"
GITHUB_REPO   = "https://github.com/USERNAME/REPO.git"  # <- schimba asta
GITHUB_FOLDER = "./github_repo"

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(CKPT_PATH, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {DEVICE}")
print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ─── GRADIENT PENALTY ─────────────────────────────────────────────────────────
def gradient_penalty(disc, real, fake):
    B = real.shape[0]
    alpha = torch.rand(B, 1, 1, 1, device=DEVICE)
    interp = (alpha * real + (1 - alpha) * fake.detach()).requires_grad_(True)
    d_interp = disc(interp)
    grad = torch.autograd.grad(
        outputs=d_interp, inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True, retain_graph=True,
    )[0]
    return ((grad.view(B, -1).norm(2, dim=1) - 1) ** 2).mean()


# ─── DATASET ─────────────────────────────────────────────────────────────────
def get_dataloader():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32),
                          interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15,
                               saturation=0.15, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
    print(f"[INFO] Dataset: {len(dataset)} imagini")
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,  # workers raman activi intre epoci
    )


# ─── GITHUB SAVE ─────────────────────────────────────────────────────────────
def save_to_github(epoch, ckpt_file):
    try:
        if not os.path.exists(GITHUB_FOLDER):
            subprocess.run(["git", "clone", GITHUB_REPO, GITHUB_FOLDER], check=True)
        # Checkpoint
        dest = os.path.join(GITHUB_FOLDER, "checkpoints", f"epoch_{epoch:04d}.pth")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy(ckpt_file, dest)
        # Sample-uri
        dst_out = os.path.join(GITHUB_FOLDER, "outputs")
        os.makedirs(dst_out, exist_ok=True)
        for f in os.listdir(OUTPUT_PATH):
            shutil.copy(os.path.join(OUTPUT_PATH, f), os.path.join(dst_out, f))
        # Push
        subprocess.run(["git", "-C", GITHUB_FOLDER, "add", "."], check=True)
        subprocess.run(["git", "-C", GITHUB_FOLDER, "commit",
                        "-m", f"GAN epoch {epoch}"], check=True)
        subprocess.run(["git", "-C", GITHUB_FOLDER, "push"], check=True)
        print(f"[GitHub] Salvat epoch {epoch}!")
    except Exception as e:
        print(f"[GitHub] EROARE: {e}")


# ─── TRAINING ────────────────────────────────────────────────────────────────
def train():
    torch.backends.cudnn.benchmark = True

    G = Generator(z_dim=Z_DIM, base_ch=BASE_CH).to(DEVICE)
    D = Discriminator(base_ch=BASE_CH).to(DEVICE)

    g_params = sum(p.numel() for p in G.parameters()) / 1e6
    d_params = sum(p.numel() for p in D.parameters()) / 1e6
    print(f"[INFO] Generator: {g_params:.1f}M parametri")
    print(f"[INFO] Discriminator: {d_params:.1f}M parametri")

    opt_G = torch.optim.Adam(G.parameters(), lr=LR_G, betas=(0.0, 0.9))
    opt_D = torch.optim.Adam(D.parameters(), lr=LR_D, betas=(0.0, 0.9))

    # LR scade lin de la 100% la 10% pe parcursul antrenamentului
    sched_G = torch.optim.lr_scheduler.LinearLR(
        opt_G, start_factor=1.0, end_factor=0.1, total_iters=NUM_EPOCHS)
    sched_D = torch.optim.lr_scheduler.LinearLR(
        opt_D, start_factor=1.0, end_factor=0.1, total_iters=NUM_EPOCHS)

    scaler_G = GradScaler('cuda')
    scaler_D = GradScaler('cuda')

    loader = get_dataloader()

    # Resume automat
    start_epoch = 0
    latest_ckpt = os.path.join(CKPT_PATH, "latest.pth")
    if os.path.exists(latest_ckpt):
        ckpt = torch.load(latest_ckpt, map_location=DEVICE, weights_only=True)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D.load_state_dict(ckpt["opt_D"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[INFO] Resumed de la epoca {start_epoch}")

    # Noise fix pentru sample-uri consistente intre epoci
    fixed_z = torch.randn(NUM_SAMPLES, Z_DIM, device=DEVICE)

    print(f"[INFO] Incep antrenamentul de la epoca {start_epoch}...")

    for epoch in range(start_epoch, NUM_EPOCHS):
        G.train()
        D.train()
        total_d, total_g = 0.0, 0.0

        for step, (real, _) in enumerate(loader):
            real = real.to(DEVICE, non_blocking=True)
            B = real.shape[0]

            # ── Discriminator ──
            for _ in range(N_CRITIC):
                z = torch.randn(B, Z_DIM, device=DEVICE)
                with autocast('cuda'):
                    fake = G(z).detach()
                    d_loss = (-D(real).mean() + D(fake).mean()
                              + LAMBDA_GP * gradient_penalty(D, real, fake))
                opt_D.zero_grad(set_to_none=True)
                scaler_D.scale(d_loss).backward()
                scaler_D.step(opt_D)
                scaler_D.update()

            # ── Generator ──
            z = torch.randn(B, Z_DIM, device=DEVICE)
            with autocast('cuda'):
                g_loss = -D(G(z)).mean()
            opt_G.zero_grad(set_to_none=True)
            scaler_G.scale(g_loss).backward()
            scaler_G.step(opt_G)
            scaler_G.update()

            total_d += d_loss.item()
            total_g += g_loss.item()

            if step % 50 == 0:
                print(f"Epoca [{epoch}/{NUM_EPOCHS}] Step [{step}/{len(loader)}] "
                      f"D: {d_loss.item():.4f} G: {g_loss.item():.4f}")

        sched_G.step()
        sched_D.step()
        print(f"\n[Epoca {epoch}] D: {total_d/len(loader):.4f} | "
              f"G: {total_g/len(loader):.4f} | "
              f"LR_G: {sched_G.get_last_lr()[0]:.2e}\n")

        # Sample-uri vizuale (cu torch.no_grad + batch mic)
        if epoch % SAMPLE_EVERY == 0:
            G.eval()
            with torch.no_grad():
                samples = (G(fixed_z).clamp(-1, 1) + 1) / 2
            save_image(samples, os.path.join(OUTPUT_PATH, f"epoch_{epoch:04d}.png"), nrow=2)
            print(f"[INFO] Sample: epoch_{epoch:04d}.png")
            G.train()

        # Checkpoint + GitHub
        if epoch % SAVE_EVERY == 0:
            ckpt_data = {
                "epoch": epoch,
                "G": G.state_dict(),
                "D": D.state_dict(),
                "opt_G": opt_G.state_dict(),
                "opt_D": opt_D.state_dict(),
            }
            ckpt_file = os.path.join(CKPT_PATH, f"epoch_{epoch:04d}.pth")
            torch.save(ckpt_data, ckpt_file)
            torch.save(ckpt_data, latest_ckpt)
            print(f"[INFO] Checkpoint: epoch_{epoch:04d}.pth")
            save_to_github(epoch, ckpt_file)


if __name__ == "__main__":
    train()
