"""
Pré-processa os vídeos extraindo frames e salvando em cache.
Execute UMA VEZ antes do treino para acelerar MUITO o carregamento.
O carregamento de .avi durante o treino é o maior gargalo - este cache elimina isso.
"""

from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

from config import TRAIN_DIR, VAL_DIR, IMG_SIZE, NUM_FRAMES

CACHE_DIR = Path(__file__).parent / "frame_cache"


def extract_and_save(video_path: Path, output_path: Path, num_frames: int, img_size: int):
    """Extrai frames e salva como .npy"""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return False
    
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (img_size, img_size))
            frames.append(frame)
        elif frames:
            frames.append(frames[-1].copy())
    cap.release()
    
    if len(frames) < num_frames:
        return False
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.array(frames, dtype=np.uint8))
    return True


def process_split(root_dir: Path, split_name: str):
    """Processa train ou val"""
    cache_split = CACHE_DIR / split_name
    count = 0
    for class_name in ["NonFight", "Fight"]:
        class_dir = root_dir / class_name
        if not class_dir.exists():
            continue
        out_dir = cache_split / class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        videos = list(class_dir.glob("*.avi"))
        for video_path in tqdm(videos, desc=f"{split_name}/{class_name}"):
            out_path = out_dir / (video_path.stem + ".npy")
            if out_path.exists():
                continue
            if extract_and_save(video_path, out_path, NUM_FRAMES, IMG_SIZE):
                count += 1
    return count


def main():
    print("Pré-processando vídeos para cache de frames...")
    print("Isso acelera MUITO o treinamento (evita decodificar .avi a cada época)\n")
    
    CACHE_DIR.mkdir(exist_ok=True)
    process_split(TRAIN_DIR, "train")
    process_split(VAL_DIR, "val")
    
    print("\nCache criado! Agora rode: python train.py --use-cache")


if __name__ == "__main__":
    main()
