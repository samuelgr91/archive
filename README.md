# Detecção de Violência em Vídeos - RWF-2000

Trabalho de processamento de imagens para identificar violência em vídeos usando o dataset **RWF-2000**.

## Dataset

- **RWF-2000**: Dataset com ~2000 vídeos de câmeras de vigilância
- **Classes**: Fight (violência) e NonFight (sem violência)
- **Estrutura**: Vídeos em `.avi`, ~5 segundos, 30 fps
- **Divisão**: `train/` (Fight, NonFight) e `val/` (Fight, NonFight)

> **Nota**: O dataset não está incluído no repositório (~2GB). Baixe em [Papers With Code](https://paperswithcode.com/dataset/rwf-2000) ou na fonte original, e coloque na pasta `RWF-2000/` com a estrutura `train/Fight/`, `train/NonFight/`, `val/Fight/`, `val/NonFight/`.

## Estrutura do Projeto

```
archive/
├── RWF-2000/           # Dataset
│   ├── train/
│   │   ├── Fight/
│   │   └── NonFight/
│   └── val/
│       ├── Fight/
│       └── NonFight/
├── config.py           # Configurações
├── dataset.py          # Carregamento de vídeos
├── model.py            # Modelo ResNet18 + late fusion
├── train.py            # Script de treinamento
├── predict.py          # Script de predição
├── visualize_training.py
├── requirements.txt
└── models/             # Modelos salvos (criado após treino)
```

## Instalação

```bash
# Criar ambiente virtual (recomendado)
python -m venv venv
venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt
```

## Uso

### 0. (Opcional) Pré-processar vídeos para acelerar o treino

O gargalo principal é decodificar .avi durante o treino. Rodar isso UMA VEZ acelera muito:

```bash
python preprocess_cache.py
```

Depois treine com `--use-cache`.

### 1. Treinar o modelo

```bash
python train.py
```

Com cache (se rodou preprocess_cache.py):
```bash
python train.py --use-cache
```

Opções:
- `--epochs 20` - Número de épocas
- `--batch-size 8` - Tamanho do batch
- `--lr 0.0001` - Learning rate
- `--num-frames 16` - Frames extraídos por vídeo
- `--workers 4` - Workers para carregar dados em paralelo (0 se der erro no Windows)
- `--use-cache` - Usar frames pré-extraídos (muito mais rápido)

### 2. Fazer predição

```bash
# Testar em amostras da validação
python predict.py

# Predição em um vídeo específico
python predict.py --video "RWF-2000/val/Fight/exemplo.avi"
```

### 3. Visualizar treinamento

```bash
python visualize_training.py
```

### 4. Ver o modelo funcionando em vídeos

```bash
# Vídeo aleatório da validação
python demo_video.py --sample

# Vídeo específico
python demo_video.py --video "RWF-2000/val/Fight/exemplo.avi"

# Vários vídeos em sequência
python demo_video.py --sample --num 5

# Salvar vídeo com overlay da predição
python demo_video.py --sample --save resultado.mp4
```

Pressione **q** para sair, **ESPAÇO** para pausar.

## Abordagem Técnica

1. **Extração de frames**: 16 frames uniformemente distribuídos por vídeo
2. **Rede base**: ResNet18 pré-treinada no ImageNet
3. **Fusão**: Média das predições de todos os frames (late fusion)
4. **Data augmentation**: Flip horizontal, variação de cor/contraste

## Referência

- Cheng, M., Cai, K., & Li, M. (2021). RWF-2000: An Open Large Scale Video Database for Violence Detection. ICPR 2021.
