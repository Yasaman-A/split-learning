import torch.nn as nn
from torchvision import models, transforms


from ..model_registry import register
from ..architecture_bundle import ArchitectureBundle

class ResNet18(nn.Module):
    """docstring for ResNet"""

    def __init__(self, config):
        super(ResNet18, self).__init__()

        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) #MNIST change

        self.logits = 10
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Flatten(),
                                                  nn.Linear(num_ftrs, self.logits))

        self.layers = list(self.model.children())




class ResNet18Client(nn.Module):
            """docstring for ResNet"""

            def __init__(self, config):
                super(ResNet18Client, self).__init__()
                self.logits = config['logits']
                self.cut_layer = config['cut_layer']

                self.model = models.resnet18(weights=None)
                self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) #MNIST change

                num_ftrs = self.model.fc.in_features
                self.model.fc = nn.Sequential(nn.Flatten(),
                                                  nn.Linear(num_ftrs, self.logits))
                
                self.layers = list(self.model.children())

            def forward(self, x):
                for i, l in enumerate(self.layers):
                    if i > self.cut_layer:
                        break
                    x = l(x)
                return x


class ResNet18Server(nn.Module):
    """docstring for ResNet"""

    def __init__(self, config):
        super(ResNet18Server, self).__init__()
        self.logits = config['logits']
        self.cut_layer = config['cut_layer']

        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) #MNIST change
        
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Flatten(),
                                        nn.Linear(num_ftrs, self.logits))
        
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
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

eval_transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

@register("ResNet18_MNIST")
def grab():
    return ArchitectureBundle(
        base = ResNet18,
        client = ResNet18Client,
        server = ResNet18Server,
        training_transformer= transformer,
        eval_transformer= eval_transformer
    )