# Documentação Completa do Projeto – Detecção de Violência em Vídeos

## 1. Visão Geral

Projeto de **classificação binária** para detectar violência em vídeos de câmeras de vigilância usando o dataset **RWF-2000**. O modelo usa uma **CNN** (ResNet18) para extrair features de frames e faz **late fusion** (média das predições) para classificar o vídeo como **Fight** ou **NonFight**.

---

## 2. Dataset RWF-2000

### 2.1 Estrutura

```
RWF-2000/
├── train/
│   ├── Fight/      → 789 vídeos .avi (violência)
│   └── NonFight/   → 800 vídeos .avi (sem violência)
└── val/
    ├── Fight/      → 200 vídeos .avi
    └── NonFight/   → 240 vídeos .avi
```

### 2.2 Labels

| Índice | Classe    | Significado     |
|--------|-----------|-----------------|
| 0      | NonFight  | Sem violência   |
| 1      | Fight     | Com violência   |

Os labels vêm da **estrutura de pastas**:
- Vídeos em `train/Fight/` → label 1
- Vídeos em `train/NonFight/` → label 0

Não há arquivo de labels separado; o rótulo é definido pelo nome da pasta.

### 2.3 Características dos Vídeos

- Formato: `.avi`
- Duração: ~5 segundos
- Taxa: ~30 fps (~150 frames por vídeo)
- Origem: câmeras de vigilância do YouTube

---

## 3. Como o Dataset é Lido

### 3.1 Fluxo de carregamento (`dataset.py`)

1. **Caminho**: `ViolenceVideoDataset` recebe `train/` ou `val/` como `root_dir`.
2. **Listagem**: percorre `Fight/` e `NonFight/` e coleta todos os `.avi`.
3. **Labels**: associa cada vídeo a um índice:
   - `NonFight` → 0  
   - `Fight` → 1  

### 3.2 Extração de frames

Para cada vídeo:

1. Abre o arquivo com OpenCV (`cv2.VideoCapture`).
2. Calcula `total_frames`.
3. Gera índices uniformes com `np.linspace` para pegar N frames ao longo do vídeo.
4. Decodifica cada frame com `cap.read()` e converte para RGB.
5. Redimensiona para 224×224 (exigido pela ResNet).

Exemplo para 8 frames em 150 frames totais:
- Índices: `[0, 21, 42, 64, 85, 107, 128, 149]`

### 3.3 Cache opcional

Com `--use-cache`:

- `preprocess_cache.py` extrai frames e salva em `frame_cache/` como `.npy`.
- O dataset carrega esses `.npy` em vez de decodificar `.avi` durante o treino.

---

## 4. Pré-processamento e Transformações

### 4.1 Treino (`is_train=True`)

- `Resize` → 224×224  
- `RandomHorizontalFlip` (p=0.5)  
- `ColorJitter` (brightness, contrast, saturation = 0.2)  
- `ToTensor`  
- `Normalize` (média e desvio padrão do ImageNet):

  - mean: [0.485, 0.456, 0.406]  
  - std: [0.229, 0.224, 0.225]

### 4.2 Validação/Teste (`is_train=False`)

- `Resize` → 224×224  
- `ToTensor`  
- `Normalize` (mesmo padrão ImageNet)

---

## 5. Modelo

### 5.1 Arquitetura

| Componente      | Descrição                                                         |
|-----------------|-------------------------------------------------------------------|
| **Backbone**    | ResNet18 (CNN) pré-treinada no ImageNet                          |
| **Tipo**        | CNN (Redes Neurais Convolucionais)                               |
| **Camadas**     | 18 camadas convolucionais + BatchNorm + ReLU                      |
| **Features**    | 512 dimensões após a última camada convolucional                  |
| **Classificador** | Dropout(0.5) + Linear(512 → 2)                                 |
| **Fusão**       | Late fusion – média das predições de todos os frames de um vídeo  |

### 5.2 Fluxo no forward

```
Entrada: [batch, 8, 3, 224, 224]
         ↓
Reshape: [batch×8, 3, 224, 224]
         ↓
ResNet18: [batch×8, 512]
         ↓
Classifier: [batch×8, 2]
         ↓
Reshape: [batch, 8, 2]
         ↓
Média em dim=1: [batch, 2]
         ↓
Saída: logits para NonFight e Fight
```

### 5.3 Por que ResNet?

- Treinada no ImageNet, boa para reconhecimento visual.
- Curva de aprendizado mais favorável.
- Custo computacional razoável para treino e inferência.

### 5.4 Late fusion

O modelo avalia cada frame e faz a média das predições para o vídeo inteiro. Isso é mais simples que 3D-CNN ou LSTM e tende a funcionar bem em vídeos curtos.

---

## 6. Parâmetros de Treinamento

### 6.1 Configuração (`config.py`)

| Parâmetro    | Valor | Descrição                         |
|-------------|-------|-----------------------------------|
| `IMG_SIZE`  | 224   | Tamanho da entrada para ResNet    |
| `NUM_FRAMES`| 8     | Frames por vídeo                  |
| `BATCH_SIZE`| 32    | Vídeos por batch                  |
| `NUM_EPOCHS`| 5     | Épocas de treinamento             |
| `LEARNING_RATE` | 1e-4 | Taxa de aprendizado            |
| `WEIGHT_DECAY` | 1e-5 | Regularização L2                |
| `DEVICE`    | cuda  | Uso de GPU                         |

### 6.2 Otimização

- **Otimizador**: AdamW  
- **Loss**: CrossEntropyLoss  
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=2)  
- **Precisão**: Mixed precision (AMP) na GPU  

### 6.3 Argumentos da linha de comando

```
--epochs 5
--batch-size 32
--lr 0.0001
--num-frames 8
--workers 4
--use-cache
```

---

## 7. Treinamento

### 7.1 Loop por época

1. Treino: `train_epoch()` para todos os batches.
2. Validação: `validate()` no conjunto de validação.
3. Scheduler: `ReduceLROnPlateau.step(val_acc)`.
4. Salvar: melhor modelo em `models/best_violence_detector.pt`.
5. Checkpoint: a cada 5 épocas, salva `checkpoint_epoch_N.pt`.

### 7.2 Métricas

- Loss (CrossEntropy)  
- Acurácia (%).  

### 7.3 Saídas

- `models/best_violence_detector.pt` – melhor modelo (por val_acc).  
- `models/checkpoint_epoch_5.pt` – checkpoint da última época.  
- `results/training_history.json` – histórico de treino e validação.

---

## 8. Estrutura de Arquivos

| Arquivo                | Função                               |
|------------------------|--------------------------------------|
| `config.py`            | Parâmetros globais                   |
| `dataset.py`           | Leitura do dataset e extração de frames |
| `model.py`             | Definição do modelo (ResNet18 + late fusion) |
| `train.py`             | Treinamento                          |
| `predict.py`           | Predição em vídeos                   |
| `demo_video.py`        | Reprodução de vídeos com overlay     |
| `preprocess_cache.py`  | Pré-processamento e cache de frames  |
| `visualize_training.py`| Gráficos de treino                   |

---

## 9. Resumo

| Aspecto        | Detalhe                                                              |
|----------------|----------------------------------------------------------------------|
| **Dataset**    | RWF-2000 – ~2000 vídeos .avi, 2 classes (Fight / NonFight)           |
| **Labels**     | Inferidos pela pasta (Fight=1, NonFight=0)                            |
| **Modelo**     | ResNet18 (CNN) + late fusion                                         |
| **Entrada**    | 8 frames 224×224 por vídeo                                           |
| **Épocas**     | 5                                                                    |
| **Batch**      | 32 vídeos                                                            |
| **Learning rate** | 1e-4                                                              |
| **Tarefa**     | Classificação binária (violência vs. não violência)                 |
