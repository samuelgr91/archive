"""
Dataset para carregamento de vídeos RWF-2000 para detecção de violência
Extrai frames uniformemente distribuídos de cada vídeo
"""

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


CACHE_DIR = Path(__file__).parent / "frame_cache"


class ViolenceVideoDataset(Dataset):
    """
    Dataset que carrega vídeos .avi e extrai frames para classificação.
    Cada vídeo tem ~150 frames (5s @ 30fps). Extraímos NUM_FRAMES uniformemente.
    Com use_cache=True, carrega frames pré-extraídos (muito mais rápido).
    """
    
    def __init__(self, root_dir: str, num_frames: int = 16, img_size: int = 224, 
                 transform=None, is_train: bool = True, use_cache: bool = False):
        """
        Args:
            root_dir: Caminho para train/ ou val/
            num_frames: Número de frames a extrair por vídeo
            img_size: Tamanho para redimensionar as imagens
            transform: Transformações adicionais (data augmentation)
            is_train: Se True, aplica data augmentation
        """
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.img_size = img_size
        self.is_train = is_train
        self.use_cache = use_cache
        self.cache_dir = CACHE_DIR / ("train" if "train" in str(root_dir) else "val")
        
        # Coletar todos os vídeos e seus labels
        self.video_paths = []
        self.labels = []
        
        for class_idx, class_name in enumerate(["NonFight", "Fight"]):
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                for video_file in class_dir.glob("*.avi"):
                    self.video_paths.append(str(video_file))
                    self.labels.append(class_idx)
        
        # Transformação base
        if is_train:
            self.frame_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.frame_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
        self.transform = transform
    
    def __len__(self):
        return len(self.video_paths)
    
    def _extract_frames(self, video_path: str) -> np.ndarray:
        """Extrai frames uniformemente distribuídos do vídeo."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            # Retorna frames zerados se vídeo inválido
            return np.zeros((self.num_frames, self.img_size, self.img_size, 3), dtype=np.uint8)
        
        # Índices dos frames a extrair (uniformemente distribuídos)
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                # Se falhar, usa o último frame disponível
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8))
        
        cap.release()
        return np.array(frames)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Carregar do cache (muito mais rápido) ou extrair do vídeo
        if self.use_cache:
            cache_path = self.cache_dir / Path(video_path).parent.name / (Path(video_path).stem + ".npy")
            if cache_path.exists():
                frames = np.load(cache_path)
            else:
                frames = self._extract_frames(video_path)
        else:
            frames = self._extract_frames(video_path)
        
        # Aplicar transformações em cada frame
        processed_frames = []
        for frame in frames:
            tensor = self.frame_transform(frame)
            processed_frames.append(tensor)
        
        # Stack: [num_frames, C, H, W]
        video_tensor = torch.stack(processed_frames)
        
        if self.transform:
            video_tensor = self.transform(video_tensor)
        
        return video_tensor, torch.tensor(label, dtype=torch.long)


class SingleVideoDataset(Dataset):
    """Dataset para carregar um único vídeo (útil para predição)."""
    
    def __init__(self, video_path: str, num_frames: int = 16, img_size: int = 224):
        self.video_path = str(video_path)
        self.num_frames = num_frames
        self.img_size = img_size
        self.frame_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __len__(self):
        return 1
    
    def _extract_frames(self, video_path: str) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return np.zeros((self.num_frames, self.img_size, self.img_size, 3), dtype=np.uint8)
        
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                frames.append(frames[-1].copy() if frames else np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8))
        cap.release()
        return np.array(frames)
    
    def __getitem__(self, idx):
        frames = self._extract_frames(self.video_path)
        processed = torch.stack([self.frame_transform(f) for f in frames])
        return processed
