import torch.nn as nn
from torchvision import models, transforms


from ..model_registry import register
from ..architecture_bundle import ArchitectureBundle


class AlexNet(nn.Module):
    def __init__(self, config):
        super(AlexNet, self).__init__()
        self.logits = config['logits']

        self.model = models.alexnet(weights=None)

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Flatten(),
                                                  nn.Linear(num_ftrs, self.logits))

        self.layers = list(self.model.children())


class AlexNetClient(nn.Module):
    def __init__(self, config):
        super(AlexNetClient, self).__init__()
        self.logits = config['logits']
        self.cut_layer = config['cut_layer']

        self.model = models.alexnet(weights=None)


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



class AlexNetServer(nn.Module):
    def __init__(self, config):
        super(AlexNetServer, self).__init__()
        self.logits = config['logits']
        self.cut_layer = config['cut_layer']

        self.model = models.alexnet(weights=None)


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


train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
        ])

eval_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
        ])

@register("AlexNet_CIFAR10")
def grab():
    return ArchitectureBundle(
        base = AlexNet,
        client = AlexNetClient,
        server = AlexNetServer,
        training_transformer= train_transformer,
        eval_transformer= eval_transformer
    )