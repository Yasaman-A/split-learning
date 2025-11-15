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
    transforms.Resize(256),                          
    transforms.RandomCrop(224),                      
    transforms.RandomHorizontalFlip(),               
    transforms.ToTensor(),                            
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

eval_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

@register("AlexNet_ImageNet")
def grab():
    return ArchitectureBundle(
        base = AlexNet,
        client = AlexNetClient,
        server = AlexNetServer,
        training_transformer= train_transformer,
        eval_transformer= eval_transformer
    )