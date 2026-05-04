import torch.nn as nn
from torchvision import models, transforms

from ..model_registry import register
from ..architecture_bundle import ArchitectureBundle


class ResNet18(nn.Module):
    """ResNet18 for DermaMNIST"""

    def __init__(self, config):
        super(ResNet18, self).__init__()

        self.model = models.resnet18(weights=None)

        self.logits = config["logits"]
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Flatten(), nn.Linear(num_ftrs, self.logits))

        self.layers = list(self.model.children())


class ResNet18Client(nn.Module):
    """ResNet18 Client for DermaMNIST"""

    def __init__(self, config):
        super(ResNet18Client, self).__init__()
        self.logits = config["logits"]
        self.cut_layer = config["cut_layer"]

        self.model = models.resnet18(weights=None)

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Flatten(), nn.Linear(num_ftrs, self.logits))

        self.layers = list(self.model.children())

    def forward(self, x):
        for i, l in enumerate(self.layers):
            if i > self.cut_layer:
                break
            x = l(x)
        return x


class ResNet18Server(nn.Module):
    """ResNet18 Server for DermaMNIST"""

    def __init__(self, config):
        super(ResNet18Server, self).__init__()
        self.logits = config["logits"]
        self.cut_layer = config["cut_layer"]

        self.model = models.resnet18(weights=None)

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Flatten(), nn.Linear(num_ftrs, self.logits))

        self.layers = list(self.model.children())

    def forward(self, x):
        for i, l in enumerate(self.layers):
            if i <= self.cut_layer:
                continue
            x = l(x)
        return x

    def classify(self, x):
        return nn.functional.softmax(self.forward(x))
    
    def change_cut(self, cut_layer):
        self.cut_layer = cut_layer


# DermaMNIST transforms: 28x28 RGB images, resize to 32x32 for ResNet18 compatibility
# Using ImageNet-style normalization (common for RGB medical images)
transformer = transforms.Compose(
    [
        transforms.Resize((32, 32)),  # Resize 28x28 to 32x32 for ResNet18
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        ),  # ImageNet stats
    ]
)

eval_transformer = transforms.Compose(
    [
        transforms.Resize((32, 32)),  # Resize 28x28 to 32x32 for ResNet18
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        ),  # ImageNet stats
    ]
)


@register("ResNet18_DermaMNIST")
def grab():
    return ArchitectureBundle(
        base=ResNet18,
        client=ResNet18Client,
        server=ResNet18Server,
        training_transformer=transformer,
        eval_transformer=eval_transformer,
    )
