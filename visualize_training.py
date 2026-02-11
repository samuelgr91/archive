"""
Visualiza o histórico de treinamento (gráficos de loss e acurácia)
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt

from config import RESULTS_DIR


def main():
    history_path = RESULTS_DIR / "training_history.json"
    if not history_path.exists():
        print("Execute train.py primeiro para gerar o histórico de treinamento.")
        return
    
    with open(history_path) as f:
        history = json.load(f)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    # Loss
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss durante o treinamento")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Acurácia
    ax2.plot(epochs, history["train_acc"], "b-", label="Train Acc")
    ax2.plot(epochs, history["val_acc"], "r-", label="Val Acc")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Acurácia (%)")
    ax2.set_title("Acurácia durante o treinamento")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = RESULTS_DIR / "training_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Gráfico salvo em: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
