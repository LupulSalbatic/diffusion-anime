# Diffusion Model Anime Faces

DDPM (Denoising Diffusion Probabilistic Model) pentru generare fete anime 512x512.

## Fisiere
- `model.py` — arhitectura U-Net complex cu Self-Attention
- `train.py` — antrenament complet cu save pe GitHub
- `setup.sh` — setup automat pe Vast.ai

## Pasi pe Vast.ai

### 1. Inchiriaza instanta
- RTX A4000 16GB la ~$0.097/hr
- Template: PyTorch
- Container Size: minim 50GB

### 2. In Jupyter Lab, deschide Terminal si ruleaza:
```bash
git clone https://github.com/USERNAME/REPO.git
cd REPO
bash setup.sh
```

### 3. Inainte de setup, upload kaggle.json:
- Du-te la kaggle.com → Account → API → Create New Token
- Upload fisierul in /root/.kaggle/ din Jupyter file browser

### 4. Modifica train.py:
```python
GITHUB_REPO = "https://TOKEN@github.com/USERNAME/REPO.git"
```
- Genereaza TOKEN la: github.com → Settings → Developer Settings → Personal Access Tokens

### 5. Porneste antrenamentul:
```bash
python3 train.py
```

## Output-uri
- `./outputs/sample_epoch_X.png` — imagini generate (grila 4x4)
- `./checkpoints/epoch_X.pth` — checkpoint local
- `./checkpoints/latest.pth` — ultimul checkpoint (pentru resume)
- GitHub → checkpoints/ — backup automat

## Resume antrenament
Daca instanta se opreste, pe noua instanta:
```bash
git clone https://github.com/USERNAME/REPO.git
cd REPO
bash setup.sh
# Descarca ultimul checkpoint de pe GitHub in ./checkpoints/latest.pth
python3 train.py  # porneste automat de unde a ramas
```

## Ajustari VRAM
Daca ai OOM (out of memory):
- Micsoreaza BATCH_SIZE din 4 la 2
- Daca ai mai mult VRAM, mareste la 6-8
