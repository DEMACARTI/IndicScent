"""Image-only sentiment model: ImageNet-pretrained ResNet-50 + classification head."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RESNET50_WEIGHTS = "IMAGENET1K_V2"


def build_image_transforms(image_size: int = 224, train: bool = False):
    """Resize/normalise; light augmentation for training.

    Memes contain rendered text, so we avoid horizontal flips (mirrored text is
    out-of-distribution) and aggressive crops (they cut captions); we use mild
    resized-crops, small rotations and colour jitter instead.
    """
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.8, 1.25)),
            transforms.RandomRotation(5),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ImageEncoder(nn.Module):
    """ResNet-50 trunk returning the pooled 2048-d vector and the 7x7 spatial feature grid.

    Only the stages listed in ``trainable_blocks`` are fine-tuned. BatchNorm layers in
    frozen stages are kept in eval mode so their running statistics stay at ImageNet
    values (otherwise ``model.train()`` would silently modify "frozen" weights).
    """

    out_dim = 2048

    def __init__(self, trainable_blocks=("layer4",), pretrained: bool = True):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights[RESNET50_WEIGHTS] if pretrained else None
        net = torchvision.models.resnet50(weights=weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = net.layer1, net.layer2, net.layer3, net.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.trainable_blocks = tuple(trainable_blocks)
        for p in self.parameters():
            p.requires_grad = False
        for name in self.trainable_blocks:
            for p in getattr(self, name).parameters():
                p.requires_grad = True

    @property
    def gradcam_layer(self) -> nn.Module:
        return self.layer4

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            for name in ("stem", "layer1", "layer2", "layer3", "layer4"):
                if name not in self.trainable_blocks:
                    getattr(self, name).eval()
        return self

    def forward(self, pixel_values: torch.Tensor):
        x = self.stem(pixel_values)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))     # (B, 2048, 7, 7)
        pooled = self.pool(x).flatten(1)                               # (B, 2048)
        grid = x.flatten(2).transpose(1, 2)                            # (B, 49, 2048)
        return pooled, grid


class ImageClassifier(nn.Module):
    modality = "image"

    def __init__(self, encoder: ImageEncoder, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(encoder.out_dim, num_classes)

    def forward(self, pixel_values=None, **_):
        pooled, _ = self.encoder(pixel_values)
        return self.head(self.dropout(pooled))
