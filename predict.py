"""
Script de predição/inferência para detecção de violência em vídeos
"""

import argparse
from pathlib import Path

import torch

from config import VAL_DIR, MODELS_DIR, IMG_SIZE, NUM_FRAMES, CLASSES
from dataset import ViolenceVideoDataset, SingleVideoDataset
from model import get_model


def predict_video(model, video_path: str, device, num_frames: int = NUM_FRAMES):
    """
    Faz predição para um único vídeo.
    
    Returns:
        (classe_predita, probabilidade, nome_classe)
    """
    dataset = SingleVideoDataset(
        video_path,
        num_frames=num_frames,
        img_size=IMG_SIZE
    )
    video_tensor = dataset[0].unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(video_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = outputs.argmax(dim=1).item()
        confidence = probs[0][pred_class].item()
    
    return pred_class, confidence, CLASSES[pred_class]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, help="Caminho para um vídeo .avi")
    parser.add_argument("--model", type=str, default=str(MODELS_DIR / "best_violence_detector.pt"))
    parser.add_argument("--val-samples", type=int, default=5, 
                        help="Número de amostras aleatórias da validação para testar")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Carregar modelo
    if not Path(args.model).exists():
        print(f"Erro: Modelo não encontrado em {args.model}")
        print("Execute train.py primeiro para treinar o modelo.")
        return
    
    model = get_model(num_classes=2)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    if args.video:
        # Predição para vídeo específico
        print(f"\nAnalisando vídeo: {args.video}")
        pred_class, confidence, class_name = predict_video(model, args.video, device)
        print(f"\nResultado: {class_name}")
        print(f"Confiança: {confidence*100:.2f}%")
        print(f"Probabilidades: Fight={confidence*100:.1f}% | NonFight={(1-confidence)*100:.1f}%" 
              if pred_class == 1 else f"Probabilidades: NonFight={confidence*100:.1f}% | Fight={(1-confidence)*100:.1f}%")
    else:
        # Testar em amostras da validação
        print("\nTestando em amostras da validação...")
        val_dataset = ViolenceVideoDataset(
            str(VAL_DIR),
            num_frames=NUM_FRAMES,
            img_size=IMG_SIZE,
            is_train=False
        )
        
        import random
        indices = random.sample(range(len(val_dataset)), min(args.val_samples, len(val_dataset)))
        
        correct = 0
        for idx in indices:
            video_tensor, label = val_dataset[idx]
            video_tensor = video_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(video_tensor)
                pred = outputs.argmax(dim=1).item()
                probs = torch.softmax(outputs, dim=1)[0]
            
            video_name = Path(val_dataset.video_paths[idx]).name
            correct += (pred == label)
            status = "✓" if pred == label else "✗"
            print(f"  {status} {video_name}: Pred={CLASSES[pred]} | Real={CLASSES[label]} | "
                  f"Fight={probs[1].item()*100:.1f}%")
        
        print(f"\nAcurácia nas amostras: {correct}/{len(indices)} ({100*correct/len(indices):.1f}%)")


if __name__ == "__main__":
    main()
