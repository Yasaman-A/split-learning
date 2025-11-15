import torch.nn as nn
from torchvision import models, transforms

from ..model_registry import register
from ..architecture_bundle import ArchitectureBundle

class SqueezeNet(nn.Module):
    def __init__(self, config):
        super(SqueezeNet, self).__init__()
        self.logits = config['logits']
        self.cut_layer = config['cut_layer']
        self.model = models.squeezenet1_1(weights=None)
        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1)
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, self.logits, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.layers = list(self.model.children())

class SqueezeNetClient(nn.Module):
    def __init__(self, config):
        super(SqueezeNetClient, self).__init__()
        self.logits = config['logits']
        self.cut_layer = config['cut_layer']
        self.model = models.squeezenet1_1(weights=None)
        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1)
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, self.logits, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.layers = list(self.model.children())

    def forward(self, x):
        for i, l in enumerate(self.layers):
            if i > self.cut_layer:
                break
            x = l(x)
        return x

class SqueezeNetServer(nn.Module):
    def __init__(self, config):
        super(SqueezeNetServer, self).__init__()
        self.logits = config['logits']
        self.cut_layer = config['cut_layer']
        self.model = models.squeezenet1_1(weights=None)
        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1)
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, self.logits, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.layers = list(self.model.children())

    def forward(self, x):
        for i, l in enumerate(self.layers):
            if i <= self.cut_layer:
                continue
            x = l(x)
        return x

    def classify(self, x):
        return nn.functional.softmax(self.forward(x))

transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

eval_transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

@register("SqueezeNet_MNIST")
def grab():
    return ArchitectureBundle(
        base=SqueezeNet,
        client=SqueezeNetClient,
        server=SqueezeNetServer,
        training_transformer=transformer,
        eval_transformer=eval_transformer
    )