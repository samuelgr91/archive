"""
Modelo para detecção de violência em vídeos
Usa ResNet18 pré-treinado para extrair features de cada frame
e agrega as predições por média (late fusion)
"""

import torch
import torch.nn as nn
from torchvision import models


class ViolenceDetector(nn.Module):
    """
    Modelo que processa múltiplos frames de um vídeo e classifica como Fight/NonFight.
    Estratégia: Extrai features de cada frame com ResNet e faz média das predições.
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        
        # Backbone: ResNet18 pré-treinado no ImageNet
        if pretrained:
            try:
                self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            except AttributeError:
                self.backbone = models.resnet18(pretrained=True)
        else:
            self.backbone = models.resnet18(weights=None)
        
        # Remover a camada fully connected original
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Classificador
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes)
        )
        
        self.num_classes = num_classes
    
    def forward(self, x):
        """
        Args:
            x: Tensor [batch, num_frames, C, H, W]
        Returns:
            logits: [batch, num_classes]
        """
        batch_size, num_frames, C, H, W = x.shape
        
        # Reshape para processar todos os frames: [batch*num_frames, C, H, W]
        x = x.view(batch_size * num_frames, C, H, W)
        
        # Extrair features
        features = self.backbone(x)
        
        # Classificar cada frame: [batch*num_frames, num_classes]
        logits = self.classifier(features)
        
        # Reshape: [batch, num_frames, num_classes]
        logits = logits.view(batch_size, num_frames, self.num_classes)
        
        # Média das predições de todos os frames (late fusion)
        logits = logits.mean(dim=1)
        
        return logits


def get_model(num_classes: int = 2, dropout: float = 0.5) -> ViolenceDetector:
    """Factory function para criar o modelo."""
    return ViolenceDetector(num_classes=num_classes, pretrained=True, dropout=dropout)
