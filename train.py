"""
Script de treinamento do modelo de detecção de violência em vídeos
Dataset: RWF-2000
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from config import (
    TRAIN_DIR, VAL_DIR, MODELS_DIR, RESULTS_DIR,
    IMG_SIZE, NUM_FRAMES, BATCH_SIZE, NUM_EPOCHS,
    LEARNING_RATE, WEIGHT_DECAY, DEVICE, CLASSES
)
from dataset import ViolenceVideoDataset
from model import get_model


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """
    Treina por uma época.
    Para cada batch: move para GPU → forward → loss → backward → atualiza pesos.
    """
    model.train()  # Ativa modo treino (Dropout ativo)
    total_loss = 0.0
    correct = 0
    total = 0
    use_amp = scaler is not None
    pin_memory = device.type == "cuda"
    
    pbar = tqdm(dataloader, desc="Treinando")
    
    for videos, labels in pbar:
        # PASSO 1: Mover batch para GPU (videos [16,16,3,224,224], labels [16])
        videos = videos.to(device, non_blocking=pin_memory)
        labels = labels.to(device, non_blocking=pin_memory)
        
        # PASSO 2: Zerar gradientes da iteração anterior
        optimizer.zero_grad()
        
        # PASSO 3: Forward pass (modelo gera predições)
        if use_amp:
            with autocast("cuda"):  # Mixed precision para economizar VRAM
                outputs = model(videos)   # [16, 2] - logits para NonFight e Fight
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()  # Calcula gradientes
            scaler.step(optimizer)         # Atualiza pesos
            scaler.update()
        else:
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # PASSO 4: Acumular métricas (loss e acurácia)
        total_loss += loss.item()
        _, predicted = outputs.max(1)       # Classe com maior score
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100.0 * correct / total:.2f}%"
        })
    
    return total_loss / len(dataloader), 100.0 * correct / total


def validate(model, dataloader, criterion, device):
    """
    Valida o modelo. Igual ao train_epoch mas sem backward/step.
    model.eval() desativa Dropout, torch.no_grad() economiza memória.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    pin_memory = device.type == "cuda"
    
    with torch.no_grad():  # Não calcula gradientes (só inferência)
        for videos, labels in tqdm(dataloader, desc="Validando"):
            videos = videos.to(device, non_blocking=pin_memory)
            labels = labels.to(device, non_blocking=pin_memory)
            
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(dataloader), 100.0 * correct / total


def main():
    # ========== LEITURA DOS ARGUMENTOS ==========
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers (0 se der erro no Windows)")
    parser.add_argument("--use-cache", action="store_true", 
                        help="Usar cache de frames (rode preprocess_cache.py antes)")
    args = parser.parse_args()
    
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # ========== DATASETS ==========
    # ViolenceVideoDataset: percorre train/Fight e train/NonFight, coleta .avi e labels (0 ou 1)
    print("Carregando datasets...")
    train_dataset = ViolenceVideoDataset(
        str(TRAIN_DIR),
        num_frames=args.num_frames,
        img_size=IMG_SIZE,
        is_train=True,
        use_cache=args.use_cache
    )
    val_dataset = ViolenceVideoDataset(
        str(VAL_DIR),
        num_frames=args.num_frames,
        img_size=IMG_SIZE,
        is_train=False,
        use_cache=args.use_cache
    )
    
    # ========== DATALOADERS ==========
    # Agrupa vídeos em batches. num_workers carrega em paralelo, prefetch mantém GPU ocupada
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True if device.type == "cuda" else False,
        prefetch_factor=4 if args.workers > 0 else None,
        persistent_workers=True if args.workers > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True if device.type == "cuda" else False,
        prefetch_factor=4 if args.workers > 0 else None,
        persistent_workers=True if args.workers > 0 else False,
    )
    
    print(f"Train: {len(train_dataset)} vídeos | Val: {len(val_dataset)} vídeos")
    
    # ========== MODELO, LOSS E OTIMIZADOR ==========
    model = get_model(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=WEIGHT_DECAY
    )
    # ReduceLROnPlateau: reduz LR em 50% se val_acc não melhorar em 5 épocas
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    
    # Mixed precision (AMP): usa float16 na GPU para ser mais rápido
    scaler = GradScaler() if device.type == "cuda" else None
    if scaler:
        print("Mixed precision (AMP) ativado para aceleração na GPU")
    if args.use_cache:
        print("Usando cache de frames (carregamento acelerado)")
    
    # ========== LOOP DE TREINAMENTO ==========
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    print("\n" + "="*60)
    print("Iniciando treinamento")
    print("="*60)
    
    for epoch in range(args.epochs):
        print(f"\nÉpoca {epoch + 1}/{args.epochs}")
        print("-" * 40)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_acc)  # Atualiza LR se val_acc não melhorou (patience)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"Train - Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"Val   - Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
        
        # Salvar melhor modelo (quando val_acc bate recorde)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = MODELS_DIR / "best_violence_detector.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, save_path)
            print(f"  -> Melhor modelo salvo! (acc: {val_acc:.2f}%)")
        
        # Checkpoint a cada 5 épocas (para retomar treino se precisar)
        if (epoch + 1) % 5 == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
            }, MODELS_DIR / f"checkpoint_epoch_{epoch + 1}.pt")
    
    print("\n" + "="*60)
    print(f"Treinamento finalizado! Melhor validação: {best_val_acc:.2f}%")
    print("="*60)
    
    # Salvar histórico (loss, acc por época) para visualize_training.py
    import json
    with open(RESULTS_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
