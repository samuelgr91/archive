"""
Gera matriz de confusão, F1, precisão, recall e gráficos de métricas.
Usa o modelo JÁ TREINADO - não precisa treinar de novo.
Execute: python evaluate_metrics.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import VAL_DIR, MODELS_DIR, RESULTS_DIR, IMG_SIZE, NUM_FRAMES, CLASSES
from dataset import ViolenceVideoDataset
from model import get_model


def collect_predictions(model, dataloader, device):
    """Roda o modelo em todo o dataset e retorna predições e labels reais."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []  # Probabilidades da classe positiva (Fight) para ROC

    with torch.no_grad():
        for videos, labels in tqdm(dataloader, desc="Gerando predições"):
            videos = videos.to(device)
            outputs = model(videos)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Prob de Fight (classe 1)

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plota e salva matriz de confusão."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASSES)
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão")

    # Valores nas células
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=16)

    plt.colorbar(im, ax=ax, label="Quantidade")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Matriz de confusão salva: {save_path}")


def plot_roc_curve(y_true, y_probs, save_path):
    """Plota e salva curva ROC."""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Aleatório")
    ax.set_xlabel("Taxa de Falsos Positivos")
    ax.set_ylabel("Taxa de Verdadeiros Positivos")
    ax.set_title("Curva ROC")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Curva ROC salva: {save_path}")


def plot_metrics_bar(report_dict, save_path):
    """Plota barras de precisão, recall e F1 por classe."""
    classes = [c for c in CLASSES if c in report_dict]
    precision = [float(report_dict[c]["precision"]) for c in classes]
    recall = [float(report_dict[c]["recall"]) for c in classes]
    f1 = [float(report_dict[c]["f1-score"]) for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar(x - width, precision, width, label="Precisão", color="#2ecc71")
    bars2 = ax.bar(x, recall, width, label="Recall", color="#3498db")
    bars3 = ax.bar(x + width, f1, width, label="F1-Score", color="#e74c3c")

    ax.set_ylabel("Score")
    ax.set_title("Métricas por Classe")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Métricas por classe salvas: {save_path}")


def main():
    model_path = MODELS_DIR / "best_violence_detector.pt"
    if not model_path.exists():
        print(f"Erro: Modelo não encontrado em {model_path}")
        print("Execute train.py primeiro para treinar o modelo.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Carregar modelo
    print("Carregando modelo...")
    model = get_model(num_classes=2)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # Dataset de validação
    print("Carregando dataset de validação...")
    val_dataset = ViolenceVideoDataset(
        str(VAL_DIR),
        num_frames=NUM_FRAMES,
        img_size=IMG_SIZE,
        is_train=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    # Coletar predições
    print("Gerando predições em todo o conjunto de validação...")
    y_pred, y_true, y_probs = collect_predictions(model, val_loader, device)

    # Resumo
    print("\n" + "=" * 50)
    print("RESULTADOS")
    print("=" * 50)
    accuracy = (y_pred == y_true).mean() * 100
    print(f"Acurácia: {accuracy:.2f}%")

    # Classification report (precisão, recall, F1)
    report = classification_report(
        y_true, y_pred,
        target_names=CLASSES,
        output_dict=True,
        digits=4,
    )
    print("\nRelatório de classificação:")
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4))

    # Matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    print("\nMatriz de confusão:")
    print(f"              Pred: NonFight  Pred: Fight")
    print(f"Real: NonFight     {cm[0,0]:>6}       {cm[0,1]:>6}")
    print(f"Real: Fight        {cm[1,0]:>6}       {cm[1,1]:>6}")

    # Salvar métricas em JSON
    import json

    def to_serializable(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        return obj

    metrics = {
        "accuracy": float(accuracy),
        "confusion_matrix": cm.tolist(),
        "classification_report": to_serializable(
            {k: v for k, v in report.items() if isinstance(v, dict)}
        ),
    }
    metrics_path = RESULTS_DIR / "evaluation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nMétricas salvas em: {metrics_path}")

    # Gerar gráficos
    print("\nGerando gráficos...")
    RESULTS_DIR.mkdir(exist_ok=True)
    plot_confusion_matrix(y_true, y_pred, RESULTS_DIR / "confusion_matrix.png")
    plot_roc_curve(y_true, y_probs, RESULTS_DIR / "roc_curve.png")
    plot_metrics_bar(report, RESULTS_DIR / "metrics_per_class.png")

    print("\nConcluído! Gráficos em:", RESULTS_DIR)


if __name__ == "__main__":
    main()
