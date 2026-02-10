import torch.nn as nn
from torchvision import transforms

from ..model_registry import register
from ..architecture_bundle import ArchitectureBundle


class SimpleCNN(nn.Module):
    """Simple CNN for MNIST - Custom architecture"""

    def __init__(self, config):
        super(SimpleCNN, self).__init__()

        # Create a model wrapper for compatibility with averaging code
        self.model = nn.Module()

        # Create a simple CNN architecture
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)  # 14x14 -> 7x7

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)  # 7x7 -> 3x3 (with padding)

        # Flatten and FC layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, config["logits"])

        # Assign all layers to model wrapper for averaging compatibility
        self.model.conv1 = self.conv1
        self.model.bn1 = self.bn1
        self.model.relu1 = self.relu1
        self.model.pool1 = self.pool1
        self.model.conv2 = self.conv2
        self.model.bn2 = self.bn2
        self.model.relu2 = self.relu2
        self.model.pool2 = self.pool2
        self.model.conv3 = self.conv3
        self.model.bn3 = self.bn3
        self.model.relu3 = self.relu3
        self.model.pool3 = self.pool3
        self.model.flatten = self.flatten
        self.model.fc1 = self.fc1
        self.model.relu4 = self.relu4
        self.model.dropout = self.dropout
        self.model.fc2 = self.fc2

        # Store layers in a list for cut_layer functionality
        self.layers = [
            self.conv1,
            self.bn1,
            self.relu1,
            self.pool1,
            self.conv2,
            self.bn2,
            self.relu2,
            self.pool2,
            self.conv3,
            self.bn3,
            self.relu3,
            self.pool3,
            self.flatten,
            self.fc1,
            self.relu4,
            self.dropout,
            self.fc2,
        ]

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = self.dropout(self.relu4(self.fc1(x)))
        x = self.fc2(x)
        return x


class SimpleCNNClient(nn.Module):
    """SimpleCNN Client for MNIST"""

    def __init__(self, config):
        super(SimpleCNNClient, self).__init__()
        self.logits = config["logits"]
        self.cut_layer = config["cut_layer"]

        # Create a model wrapper for compatibility with averaging code
        self.model = nn.Module()

        # Create the same architecture
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, self.logits)

        # Assign all layers to model wrapper for averaging compatibility
        self.model.conv1 = self.conv1
        self.model.bn1 = self.bn1
        self.model.relu1 = self.relu1
        self.model.pool1 = self.pool1
        self.model.conv2 = self.conv2
        self.model.bn2 = self.bn2
        self.model.relu2 = self.relu2
        self.model.pool2 = self.pool2
        self.model.conv3 = self.conv3
        self.model.bn3 = self.bn3
        self.model.relu3 = self.relu3
        self.model.pool3 = self.pool3
        self.model.flatten = self.flatten
        self.model.fc1 = self.fc1
        self.model.relu4 = self.relu4
        self.model.dropout = self.dropout
        self.model.fc2 = self.fc2

        # Store layers in a list for cut_layer functionality
        self.layers = [
            self.conv1,
            self.bn1,
            self.relu1,
            self.pool1,
            self.conv2,
            self.bn2,
            self.relu2,
            self.pool2,
            self.conv3,
            self.bn3,
            self.relu3,
            self.pool3,
            self.flatten,
            self.fc1,
            self.relu4,
            self.dropout,
            self.fc2,
        ]

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i > self.cut_layer:
                break
            x = layer(x)
        return x


class SimpleCNNServer(nn.Module):
    """SimpleCNN Server for MNIST"""

    def __init__(self, config):
        super(SimpleCNNServer, self).__init__()
        self.logits = config["logits"]
        self.cut_layer = config["cut_layer"]

        # Create a model wrapper for compatibility with averaging code
        self.model = nn.Module()

        # Create the same architecture
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, self.logits)

        # Assign all layers to model wrapper for averaging compatibility
        self.model.conv1 = self.conv1
        self.model.bn1 = self.bn1
        self.model.relu1 = self.relu1
        self.model.pool1 = self.pool1
        self.model.conv2 = self.conv2
        self.model.bn2 = self.bn2
        self.model.relu2 = self.relu2
        self.model.pool2 = self.pool2
        self.model.conv3 = self.conv3
        self.model.bn3 = self.bn3
        self.model.relu3 = self.relu3
        self.model.pool3 = self.pool3
        self.model.flatten = self.flatten
        self.model.fc1 = self.fc1
        self.model.relu4 = self.relu4
        self.model.dropout = self.dropout
        self.model.fc2 = self.fc2

        # Store layers in a list for cut_layer functionality
        self.layers = [
            self.conv1,
            self.bn1,
            self.relu1,
            self.pool1,
            self.conv2,
            self.bn2,
            self.relu2,
            self.pool2,
            self.conv3,
            self.bn3,
            self.relu3,
            self.pool3,
            self.flatten,
            self.fc1,
            self.relu4,
            self.dropout,
            self.fc2,
        ]

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i <= self.cut_layer:
                continue
            x = layer(x)
        return x

    def classify(self, x):
        return nn.functional.softmax(self.forward(x))

    def change_cut(self, cut_layer):
        self.cut_layer = cut_layer


# MNIST transforms: grayscale, normalize with MNIST statistics
# MNIST mean=0.1307, std=0.3081
transformer = transforms.Compose(
    [
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

eval_transformer = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)


@register("SimpleCNN_MNIST")
def grab():
    return ArchitectureBundle(
        base=SimpleCNN,
        client=SimpleCNNClient,
        server=SimpleCNNServer,
        training_transformer=transformer,
        eval_transformer=eval_transformer,
    )
