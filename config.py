"""
Configurações do projeto de detecção de violência em vídeos
Dataset: RWF-2000
"""

import os
from pathlib import Path

# Caminhos
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "RWF-2000"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Classes do dataset
CLASSES = ["NonFight", "Fight"]
NUM_CLASSES = len(CLASSES)

# Parâmetros dos vídeos (menos frames = mais rápido)
IMG_SIZE = 224          # Tamanho da imagem para a rede (ResNet padrão)
NUM_FRAMES = 8          # Número de frames extraídos por vídeo
FRAME_SAMPLE_RATE = 2   # Pegar 1 frame a cada N frames (30fps -> ~15 frames no 5s)

# Treinamento (batch maior = menos iterações por época)
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Device - GPU para treinamento
DEVICE = "cuda"

# Criar diretórios se não existirem
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
