import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from ..model_registry import register
from ..architecture_bundle import ArchitectureBundle


class EfficientNet(nn.Module):
    """EfficientNet-B0 for DermaMNIST"""

    def __init__(self, config):
        super(EfficientNet, self).__init__()

        # Load EfficientNet-B0 (smallest variant, good for 28x28 images)
        self.model = models.efficientnet_b0(weights=None)

        self.logits = config["logits"]
        # EfficientNet uses 'classifier' instead of 'fc'
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            self.model.classifier[0], nn.Linear(num_ftrs, self.logits)  # Dropout layer
        )

        # EfficientNet has features (conv layers) and classifier (FC layers)
        # Flatten the structure for split learning
        features_layers = list(self.model.features.children())
        classifier_layers = list(self.model.classifier.children())
        self.layers = features_layers + classifier_layers


class EfficientNetClient(nn.Module):
    """EfficientNet Client for DermaMNIST"""

    def __init__(self, config):
        super(EfficientNetClient, self).__init__()
        self.logits = config["logits"]
        self.cut_layer = config["cut_layer"]

        self.model = models.efficientnet_b0(weights=None)

        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            self.model.classifier[0], nn.Linear(num_ftrs, self.logits)  # Dropout layer
        )

        # EfficientNet has features (conv layers) and classifier (FC layers)
        # Flatten the structure for split learning
        features_layers = list(self.model.features.children())
        classifier_layers = list(self.model.classifier.children())
        self.layers = features_layers + classifier_layers

    def forward(self, x):
        for i, l in enumerate(self.layers):
            if i > self.cut_layer:
                break
            x = l(x)
        return x


class EfficientNetServer(nn.Module):
    """EfficientNet Server for DermaMNIST"""

    def __init__(self, config):
        super(EfficientNetServer, self).__init__()
        self.logits = config["logits"]
        self.cut_layer = config["cut_layer"]

        self.model = models.efficientnet_b0(weights=None)

        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            self.model.classifier[0], nn.Linear(num_ftrs, self.logits)  # Dropout layer
        )

        # EfficientNet has features (conv layers) and classifier (FC layers)
        # Flatten the structure for split learning
        features_layers = list(self.model.features.children())
        classifier_layers = list(self.model.classifier.children())
        self.layers = features_layers + classifier_layers

        # Store the number of feature layers to detect when we transition to classifier
        self.num_features_layers = len(features_layers)

    def forward(self, x):
        for i, l in enumerate(self.layers):
            if i <= self.cut_layer:
                continue

            # If we're transitioning from features to classifier and input is still 4D
            # EfficientNet's classifier expects flattened input after adaptive pooling
            if i == self.num_features_layers and len(x.shape) == 4:
                # Add adaptive pooling and flattening
                x = F.adaptive_avg_pool2d(x, (1, 1))
                x = x.view(
                    x.size(0), -1
                )  # Flatten: [batch, channels, 1, 1] -> [batch, channels]

            x = l(x)
        return x

    def classify(self, x):
        return nn.functional.softmax(self.forward(x))


# DermaMNIST transforms: 28x28 RGB images, resize to 224x224 for EfficientNet
# EfficientNet is designed for ImageNet (224x224), so we resize accordingly
# Using ImageNet-style normalization
transformer = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize 28x28 to 224x224 for EfficientNet
        transforms.RandomCrop(224, padding=16),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        ),  # ImageNet stats
    ]
)

eval_transformer = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize 28x28 to 224x224 for EfficientNet
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        ),  # ImageNet stats
    ]
)


@register("EfficientNet_DermaMNIST")
def grab():
    return ArchitectureBundle(
        base=EfficientNet,
        client=EfficientNetClient,
        server=EfficientNetServer,
        training_transformer=transformer,
        eval_transformer=eval_transformer,
    )
