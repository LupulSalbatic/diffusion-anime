#!/bin/bash
# ─── SETUP VAST.AI ───────────────────────────────────────────────────────────
# Ruleaza asta prima data dupa ce te conectezi pe instanta
# Comanda: bash setup.sh

set -e
echo "=============================="
echo "  SETUP DIFFUSION MODEL ANIME"
echo "=============================="

# 1. Update si dependinte sistem
echo "[1/6] Instalez dependinte sistem..."
apt-get update -q
apt-get install -y git unzip wget

# 2. Python packages
echo "[2/6] Instalez PyTorch si librarii..."
pip install --upgrade pip -q
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
pip install kaggle pillow tqdm -q

# 3. Verifica GPU
echo "[3/6] Verific GPU..."
python3 -c "
import torch
print(f'CUDA disponibil: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# 4. Descarca dataset de pe Kaggle
echo "[4/6] Descarc dataset anime 512x512..."
echo "--- Ai nevoie de kaggle.json in /root/.kaggle/ ---"
echo "--- Du-te la kaggle.com -> Account -> API -> Create Token ---"
mkdir -p /root/.kaggle

if [ ! -f "/root/.kaggle/kaggle.json" ]; then
    echo "EROARE: Lipseste /root/.kaggle/kaggle.json"
    echo "Upload-eaza fisierul kaggle.json manual in Jupyter, apoi ruleaza din nou."
    exit 1
fi

chmod 600 /root/.kaggle/kaggle.json
kaggle datasets download subinium/highresolution-anime-face-dataset-512x512 -p ./raw_dataset
echo "Extrag dataset..."
unzip -q ./raw_dataset/*.zip -d ./raw_dataset/extracted

# 5. Organizeaza dataset pentru ImageFolder
echo "[5/6] Organizez dataset..."
mkdir -p ./dataset/anime
find ./raw_dataset/extracted -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" | \
    xargs -I {} cp {} ./dataset/anime/
echo "Total imagini: $(ls ./dataset/anime | wc -l)"

# 6. Setup GitHub
echo "[6/6] Configurez Git..."
git config --global user.email "vast.ai.training@gmail.com"
git config --global user.name "VastAI Training"
echo "--- Schimba GITHUB_REPO din train.py cu link-ul tau! ---"
echo "--- Format: https://TOKEN@github.com/USERNAME/REPO.git ---"

echo ""
echo "=============================="
echo "  SETUP COMPLET!"
echo "  Ruleaza: python3 train.py"
echo "=============================="
