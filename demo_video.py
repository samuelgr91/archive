"""
Demonstração: reproduz vídeos com a predição do modelo sobreposta.
Mostra o modelo funcionando em tempo real na tela.
"""

import argparse
from pathlib import Path

import cv2
import torch

from config import VAL_DIR, MODELS_DIR, IMG_SIZE, NUM_FRAMES, CLASSES
from dataset import SingleVideoDataset
from model import get_model


def get_prediction(model, video_path: str, device):
    """Obtém predição e probabilidades do modelo."""
    dataset = SingleVideoDataset(video_path, num_frames=NUM_FRAMES, img_size=IMG_SIZE)
    video_tensor = dataset[0].unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(video_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    
    pred_class = probs.argmax().item()
    return pred_class, probs[0].item(), probs[1].item(), CLASSES[pred_class]


def play_video_with_prediction(video_path: str, pred_class: int, prob_fight: float, 
                               prob_nonfight: float, class_name: str, save_output: str = None):
    """
    Reproduz o vídeo com overlay mostrando a predição.
    Pressione 'q' para sair, 'espaço' para pausar.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Erro ao abrir vídeo: {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    writer = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(save_output, fourcc, fps, (width, height))
    
    # Cores: verde para NonFight, vermelho para Fight
    color = (0, 255, 0) if pred_class == 0 else (0, 0, 255)
    
    print("\n" + "="*50)
    print(f"Vídeo: {Path(video_path).name}")
    print(f"Predição: {class_name} ({max(prob_fight, prob_nonfight)*100:.1f}%)")
    print(f"  NonFight: {prob_nonfight*100:.1f}% | Fight: {prob_fight*100:.1f}%")
    print("="*50)
    print("Reproduzindo... Pressione 'q' para sair, ESPAÇO para pausar")
    print()
    
    paused = False
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reinicia
                continue
            
            # Overlay com retângulo e texto
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            
            # Texto da predição
            label = f"{class_name}: {max(prob_fight, prob_nonfight)*100:.1f}%"
            cv2.putText(frame, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(frame, f"NonFight: {prob_nonfight*100:.0f}% | Fight: {prob_fight*100:.0f}%", 
                       (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if writer:
                writer.write(frame)
            
            cv2.imshow("Detecção de Violência", frame)
        
        key = cv2.waitKey(30 if not paused else 0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
    
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Demonstra o modelo em vídeos")
    parser.add_argument("--video", type=str, help="Caminho para um vídeo .avi")
    parser.add_argument("--sample", action="store_true", 
                        help="Pegar vídeo aleatório da validação")
    parser.add_argument("--save", type=str, help="Salvar vídeo com overlay (ex: output.mp4)")
    parser.add_argument("--model", type=str, default=str(MODELS_DIR / "best_violence_detector.pt"))
    parser.add_argument("--num", type=int, default=1, 
                        help="Com --sample: quantos vídeos mostrar em sequência")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not Path(args.model).exists():
        print(f"Erro: Modelo não encontrado em {args.model}")
        return
    
    model = get_model(num_classes=2)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    
    # Obter vídeos para demonstrar
    if args.video:
        videos = [Path(args.video)]
    elif args.sample:
        import random
        all_videos = list(VAL_DIR.glob("Fight/*.avi")) + list(VAL_DIR.glob("NonFight/*.avi"))
        videos = random.sample(all_videos, min(args.num, len(all_videos)))
    else:
        print("Use --video <caminho> ou --sample para vídeo aleatório da validação")
        return
    
    for video_path in videos:
        pred_class, prob_nonfight, prob_fight, class_name = get_prediction(
            model, str(video_path), device
        )
        
        save_path = args.save
        if args.save and len(videos) > 1:
            save_path = str(Path(args.save).with_stem(
                Path(args.save).stem + "_" + video_path.stem
            ))
        
        play_video_with_prediction(
            str(video_path),
            pred_class, prob_fight, prob_nonfight, class_name,
            save_output=save_path
        )


if __name__ == "__main__":
    main()
