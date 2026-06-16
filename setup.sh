#!/bin/bash
# ─── SETUP GAN ANIME 512x512 pe Vast.ai ──────────────────────────────────────
# Ruleaza o singura data dupa ce pornesti instanta:
#   bash setup.sh
set -e

echo "=============================="
echo "  SETUP GAN ANIME 512x512"
echo "=============================="

# 1. Dependinte sistem
echo "[1/5] Instalez dependinte sistem..."
apt-get update -q && apt-get install -y git unzip -q

# 2. PyTorch + librarii
echo "[2/5] Instalez PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
pip install kaggle pillow -q

# 3. Verifica GPU
echo "[3/5] Verific GPU..."
python3 -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# 4. Dataset de pe Kaggle
echo "[4/5] Descarc dataset anime 512x512..."
mkdir -p /root/.kaggle
if [ ! -f "/root/.kaggle/kaggle.json" ]; then
    echo ""
    echo "EROARE: Lipseste /root/.kaggle/kaggle.json"
    echo "Creeaza-l manual cu:"
    echo "  mkdir -p /root/.kaggle"
    echo "  echo '{\"username\":\"bertealeonardandrei\",\"key\":\"KAGGLE_KEY\"}' > /root/.kaggle/kaggle.json"
    echo "  chmod 600 /root/.kaggle/kaggle.json"
    echo "Apoi ruleaza din nou: bash setup.sh"
    exit 1
fi
chmod 600 /root/.kaggle/kaggle.json

# Descarca in /tmp ca sa nu ocupe spatiu dublu pe disk
kaggle datasets download subinium/highresolution-anime-face-dataset-512x512 -p /tmp/kaggle
echo "Extrag dataset..."
unzip -q /tmp/kaggle/*.zip -d /tmp/extracted
rm /tmp/kaggle/*.zip

# Organizeaza pentru PyTorch ImageFolder
mkdir -p ./dataset/anime
find /tmp/extracted/portraits/ -name "*.jpg" -exec mv {} ./dataset/anime/ \;
rm -rf /tmp/extracted /tmp/kaggle

TOTAL=$(ls ./dataset/anime | wc -l)
echo "Total imagini extrase: $TOTAL"

# Pastreaza doar 50k imagini random
echo "Pastrez 50000 imagini..."
python3 -c "
import os, random
folder = './dataset/anime'
files = os.listdir(folder)
keep = set(random.sample(files, 50000))
[os.remove(os.path.join(folder, f)) for f in files if f not in keep]
print(f'Ramase: {len(os.listdir(folder))}')
"

# 5. Git config
echo "[5/5] Configurez Git..."
git config --global user.email "vast.training@gmail.com"
git config --global user.name "VastAI Training"

echo ""
echo "=============================="
echo "  SETUP COMPLET!"
echo ""
echo "  Modifica GITHUB_REPO din train.py:"
echo "  https://ghp_TOKEN@github.com/USERNAME/REPO.git"
echo ""
echo "  Apoi porneste antrenamentul:"
echo "  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 train.py"
echo "=============================="

