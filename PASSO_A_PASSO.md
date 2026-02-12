# Explicação Passo a Passo do Código

Este documento explica o fluxo completo do projeto, linha por linha.

---

## Visão Geral do Fluxo

```
1. config.py     → Define parâmetros (épocas, batch, caminhos...)
2. dataset.py    → Carrega vídeos e extrai frames
3. model.py      → Define a rede neural (ResNet18 + classificador)
4. train.py      → Treina o modelo
5. predict.py    → Faz predições em novos vídeos
6. demo_video.py → Mostra vídeos com a predição na tela
```

---

## PASSO 1: config.py – As Configurações

O que acontece quando você importa `config`:

```
1. BASE_DIR = pasta do projeto (onde está o config.py)
2. DATA_DIR = RWF-2000/ (pasta do dataset)
3. TRAIN_DIR = RWF-2000/train/
4. VAL_DIR = RWF-2000/val/
5. MODELS_DIR = models/ (onde salva os .pt)
6. RESULTS_DIR = results/ (histórico, gráficos)

7. CLASSES = ["NonFight", "Fight"]  ← ordem importa!
   - Índice 0 = NonFight
   - Índice 1 = Fight

8. IMG_SIZE = 224      ← tamanho que a ResNet espera
9. NUM_FRAMES = 16      ← quantos frames pegamos de cada vídeo
10. BATCH_SIZE = 16     ← quantos vídeos por vez na GPU
11. NUM_EPOCHS = 20    ← quantas vezes passa por todo o dataset
12. LEARNING_RATE = 1e-4
13. WEIGHT_DECAY = 1e-5

14. Cria as pastas models/ e results/ se não existirem
```

---

## PASSO 2: dataset.py – Como os Vídeos Viram Dados

### 2.1 ViolenceVideoDataset – O Construtor (`__init__`)

Quando você faz `ViolenceVideoDataset(TRAIN_DIR, ...)`:

```
1. Guarda: root_dir, num_frames, img_size, is_train, use_cache

2. PERCORRE as pastas para achar os vídeos:
   - Para class_idx=0, class_name="NonFight":
     → Abre RWF-2000/train/NonFight/
     → Para cada arquivo .avi: adiciona o caminho em video_paths, adiciona 0 em labels
   
   - Para class_idx=1, class_name="Fight":
     → Abre RWF-2000/train/Fight/
     → Para cada arquivo .avi: adiciona o caminho em video_paths, adiciona 1 em labels

3. Monta as transformações (frame_transform):
   - TREINO: Resize(224) → RandomHorizontalFlip → ColorJitter → ToTensor → Normalize
   - VALIDAÇÃO: Resize(224) → ToTensor → Normalize
   (Normalize usa valores do ImageNet para compatibilidade com ResNet pré-treinada)
```

### 2.2 _extract_frames – Extraindo Frames do Vídeo

Quando o DataLoader pede um vídeo (via `__getitem__`):

```
1. Abre o vídeo com cv2.VideoCapture("caminho/arquivo.avi")

2. Pega o total de frames: total_frames = 150 (exemplo)

3. Calcula quais frames pegar com np.linspace:
   - np.linspace(0, 149, 16) → [0, 10, 20, 30, ..., 140, 149]
   - São 16 posições uniformemente distribuídas

4. Para cada posição:
   - cap.set(POS_FRAMES, idx)  → pula para o frame idx
   - cap.read()                → lê o frame
   - cv2.cvtColor(BGR2RGB)     → OpenCV retorna BGR, convertemos para RGB
   - Adiciona na lista frames

5. Fecha o vídeo, retorna array de shape (16, altura, largura, 3)
```

### 2.3 __getitem__ – O Que o DataLoader Recebe

```
1. Recebe idx (índice do vídeo: 0, 1, 2, ...)

2. Pega video_path = video_paths[idx] e label = labels[idx]

3. CARREGA os frames:
   - Se use_cache: tenta carregar do arquivo .npy em frame_cache/
   - Senão: chama _extract_frames(video_path)

4. Para cada frame:
   - Aplica frame_transform (resize, flip, normalize...)
   - Vira tensor [3, 224, 224]

5. Junta todos: torch.stack(processed_frames)
   - Resultado: tensor [16, 3, 224, 224]

6. Retorna (video_tensor, label)
   - video_tensor: [16, 3, 224, 224]
   - label: 0 ou 1 (tensor)
```

### 2.4 SingleVideoDataset

Igual ao ViolenceVideoDataset, mas para **um único vídeo** (usado em `predict.py` e `demo_video.py`). Retorna só o tensor, sem label.

---

## PASSO 3: model.py – A Rede Neural

### 3.1 __init__ – Montando o Modelo

```
1. BACKBONE (ResNet18):
   - Carrega ResNet18 pré-treinada no ImageNet
   - Remove a última camada (fc) e substitui por Identity()
   - Ou seja: ResNet passa a retornar 512 números (features), não 1000 classes

2. GUARDA: feature_dim = 512

3. CLASSIFICADOR (substitui o fc original):
   - Dropout(0.5)  → desliga 50% dos neurônios aleatoriamente no treino
   - Linear(512, 2) → 512 entradas → 2 saídas (NonFight, Fight)
```

### 3.2 forward – O Fluxo dos Dados

```
Entrada: x com shape [batch=16, num_frames=16, C=3, H=224, W=224]
         (16 vídeos, cada um com 16 frames de 224x224 RGB)

PASSO 1 - Reshape:
   x.view(16 * 16, 3, 224, 224) → [256, 3, 224, 224]
   (trata como 256 imagens independentes)

PASSO 2 - Backbone (ResNet18):
   [256, 3, 224, 224] → ResNet18 → [256, 512]
   (cada "imagem" vira um vetor de 512 features)

PASSO 3 - Classificador:
   [256, 512] → Dropout → Linear → [256, 2]
   (cada frame recebe 2 scores: NonFight e Fight)

PASSO 4 - Reshape de volta:
   [256, 2] → view(16, 16, 2) → [16, 16, 2]
   (16 vídeos, 16 frames cada, 2 classes)

PASSO 5 - Late fusion (média):
   logits.mean(dim=1) → [16, 2]
   (média das 16 predições de cada vídeo = 1 predição por vídeo)

Saída: [16, 2] → 16 vídeos, cada um com 2 logits (um para NonFight, um para Fight)
```

---

## PASSO 4: train.py – O Treinamento

### 4.1 main() – Visão Geral

```
1. Lê argumentos (--epochs, --batch-size, etc.)

2. Define device (cuda ou cpu)

3. Cria train_dataset e val_dataset (ViolenceVideoDataset)

4. Cria DataLoaders:
   - train_loader: batches de 16 vídeos, shuffle=True, 4 workers
   - val_loader: batches de 16, shuffle=False

5. Cria modelo, criterion (CrossEntropyLoss), optimizer (AdamW), scheduler

6. Loop de 20 épocas:
   - train_epoch()
   - validate()
   - scheduler.step()
   - Salva melhor modelo se val_acc melhorou
   - Salva checkpoint a cada 5 épocas

7. Salva histórico em JSON
```

### 4.2 train_epoch() – Uma Época de Treino

```
Para cada batch (videos, labels) do train_loader:

1. MOVE para GPU:
   videos = [16, 16, 3, 224, 224]
   labels = [16]

2. Zera os gradientes: optimizer.zero_grad()

3. Forward (com AMP se tiver GPU):
   with autocast():
      outputs = model(videos)  → [16, 2]
      loss = CrossEntropyLoss(outputs, labels)

4. Backward:
   scaler.scale(loss).backward()  (ou loss.backward() se CPU)
   - Calcula gradientes de todos os parâmetros

5. Atualiza pesos:
   scaler.step(optimizer)  (ou optimizer.step())
   scaler.update()
   - Aplica os gradientes para ajustar os pesos

6. Acumula métricas:
   - loss total
   - quantas predições acertaram (comparando argmax(outputs) com labels)

7. Retorna: loss médio, acurácia %
```

### 4.3 validate()

Igual ao `train_epoch`, mas:
- `model.eval()` (desativa Dropout)
- `with torch.no_grad()` (não calcula gradientes)
- Sem `optimizer.step()`

---

## PASSO 5: predict.py – Fazendo Predições

```
1. Carrega o checkpoint (best_violence_detector.pt)

2. Carrega o modelo e faz load_state_dict()

3. Se --video foi passado:
   - SingleVideoDataset(video_path) → extrai 16 frames
   - model(video_tensor) → logits [1, 2]
   - softmax(logits) → probabilidades
   - argmax → classe predita (0 ou 1)

4. Se não passou --video:
   - Pega amostras aleatórias da validação
   - Roda o modelo em cada uma
   - Mostra predição vs label real
```

---

## PASSO 6: demo_video.py – Mostrando na Tela

```
1. Carrega o modelo (igual ao predict)

2. get_prediction():
   - Extrai frames do vídeo
   - Roda o modelo
   - Retorna: pred_class, prob_nonfight, prob_fight, class_name

3. play_video_with_prediction():
   - Abre o vídeo com OpenCV
   - Em loop: lê frame → desenha texto com a predição → mostra na janela
   - Pressione 'q' para sair, ESPAÇO para pausar
```

---

## Resumo do Fluxo de Dados

```
VÍDEO .avi
    ↓
_extract_frames() → 16 imagens [H, W, 3]
    ↓
frame_transform() → 16 tensores [3, 224, 224]
    ↓
torch.stack() → [16, 3, 224, 224] (1 vídeo)
    ↓
DataLoader agrupa 16 vídeos → [16, 16, 3, 224, 224]
    ↓
model.forward() → reshape → ResNet18 → classifier → mean → [16, 2]
    ↓
CrossEntropyLoss + backward + optimizer.step()
```

---

## Glossário Rápido

| Termo | Significado |
|-------|-------------|
| **Batch** | Conjunto de exemplos processados junto (ex: 16 vídeos) |
| **Epoch** | Uma passada completa pelo dataset de treino |
| **Loss** | Erro do modelo (quanto menor, melhor) |
| **Backward** | Cálculo dos gradientes (como ajustar cada peso) |
| **Optimizer** | Algoritmo que atualiza os pesos (AdamW) |
| **Late fusion** | Média das predições de cada frame para dar 1 predição por vídeo |
| **Logits** | Valores brutos antes do softmax (não são probabilidades) |
| **Softmax** | Converte logits em probabilidades (soma = 1) |
