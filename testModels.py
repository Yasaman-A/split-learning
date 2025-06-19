import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights
import yaml
from locale import atoi
from PIL import Image


class ResNet18Client(nn.Module):
    def __init__(self, cut_layer):
        super(ResNet18Client, self).__init__()
        self.cut_layer = cut_layer

        self.model = models.resnet18(weights=None)#ResNet18_Weights.DEFAULT)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Flatten(), nn.Linear(num_ftrs, 10))

        self.layers=list(self.model.children())

    def forward(self, x):
        for i, l in enumerate(self.layers):
            if i > self.cut_layer:
                break
            x = l(x)
        return x


class ResNet18Server(nn.Module):
    def __init__(self, cut_layer):
        super(ResNet18Server, self).__init__()
        self.cut_layer = cut_layer

        self.model = models.resnet18(weights=None)#ResNet18_Weights.DEFAULT)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Flatten(), nn.Linear(num_ftrs, 10))

        self.layers=list(self.model.children())

    def forward(self, x):
        for i, l in enumerate(self.layers):
            # Explain this part
            if i <= self.cut_layer:
                continue
            x = l(x)
        return x 
    
    ''' Classify(x)
    Minor helper function that softmaxes output to use when classifying output.
    Prevents double softmaxing.
    '''
    def classify(self, x):
        x = self.forward(x)
        return nn.functional.softmax(x, dim=1)


classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)



def predict_image_split_model(image_path, client_model, server_model, device="cpu"):
    """
    Predict class for a single image using the split model
    Args:
        image_path: Path to the image file
        client_model: Loaded client model
        server_model: Loaded server model
        device: 'cpu' or 'cuda'
    Returns:
        predicted class name and probability
    """
    # Define the same transform as used for testing
    transform = transforms.Compose(
        [
            transforms.Resize(32),  # CIFAR10 image size
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Load and transform the image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Set models to eval mode
    client_model.eval()
    server_model.eval()


    # Predict
    with torch.no_grad():
        # Forward pass through client model
        client_output = client_model(image)

        # Forward pass through server model
        outputs = server_model.classify(client_output)

        # Get prediction
        probabilities = outputs[0]
        predicted_idx = torch.argmax(probabilities).item()
        predicted_class = classes[predicted_idx]
        confidence = probabilities[predicted_idx].item()

    return predicted_class, confidence


def test_split_model_all_cifar10(
    client_model_path, server_model_path, cut_layer, device="cpu"
):
    # Set device
    device = torch.device(device)

    # Load models
    client_model = ResNet18Client(cut_layer).to(device)
    server_model = ResNet18Server(cut_layer).to(device)

    client_model.load_state_dict(torch.load(client_model_path, map_location=device))
    server_model.load_state_dict(torch.load(server_model_path, map_location=device))

    # Set models to evaluation mode
    client_model.eval()
    server_model.eval()

    # Load test dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=4
    )


    # Test the model
    correct = 0
    total = 0

    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            client_output = client_model(images)
            outputs = server_model.classify(client_output)
            _, predicted = torch.max(outputs, 1)
            
            correct = (predicted == labels).cpu()
            labels_cpu = labels.cpu()

            for cls in range(len(classes)):
                cls_mask = (labels_cpu == cls)
                class_correct[cls] += correct[cls_mask].sum().item()
                class_total[cls] += cls_mask.sum().item()

    accuracy = 100 * sum(class_correct) / sum(class_total)
    print(f"Accuracy on test set: {accuracy:.2f}%")

    # Per class accuracy
    for i in range(10):
        print(
            f"Accuracy of {classes[i]}: {100 * class_correct[i] / class_total[i]:.2f}%"
        )

    return accuracy



def predict_with_split_model(client_model_path, server_model_path, device, cut_layer):

    client_model = ResNet18Client(cut_layer).to(device)
    server_model = ResNet18Server(cut_layer).to(device)
    client_model.load_state_dict(torch.load(client_model_path, map_location=device))
    server_model.load_state_dict(torch.load(server_model_path, map_location=device))

    acc = test_split_model_all_cifar10(
        client_model_path, server_model_path, cut_layer, device=device
    )

    # Example of single image prediction - ideally should be from the classes that clients were traiend on
    image_path = "bird.png"  # Update this path
    predicted_class, confidence = predict_image_split_model(
        image_path, client_model, server_model, device
    )
    print(f"\nPrediction for single image:")
    print(f"Class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}")

    return acc


def plot_convergence_split_model(device):
    import os
    import matplotlib.pyplot as plt

    root = "./"#convergence/v1_med"
    client = ""
    server = ""

    server_fmt = "server_fedAvg_model_r{}_6_5555_cuda_[2, 3, 4, 2, 3, 4]_3_35.pt"
    client_fmt = "client_fedAvg_model_r{}_6_4445_35.pt"

    accuracies = []
    cut_layer = 4
    rounds = 28

    for round_num in range(rounds):
        print("======================================================")
        client_path = os.path.join(root, client, client_fmt.format(round_num))
        server_path = os.path.join(root, server, server_fmt.format(round_num))

        acc = predict_with_split_model(client_path, server_path, device, cut_layer)
        accuracies.append(acc)

    # ---- Plotting ----
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(accuracies)), accuracies, marker='o', label='Accuracy')
    plt.title('Model Accuracy Convergence over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("convergence_plot.png")  # Optional
    plt.show()



class ResNet18Combined(nn.Module):
    def __init__(self):
        super(ResNet18Combined, self).__init__()
        self.model = models.resnet18(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Flatten(), nn.Linear(num_ftrs, 10))
        self.layers = list(self.model.children())
    
    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x
    def classify(self, x):
        x = self.forward(x)
        return nn.functional.softmax(x, dim=1)
    

def predict_with_combined_model(model_path, device):
    model = ResNet18Combined().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()


    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=False, num_workers=4
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=4
    )


    def predict_on_loader(model, loader):
        correct = 0
        class_correct = list(0.0 for i in range(10))
        class_total = list(0.0 for i in range(10))

        with torch.no_grad():
                for data in loader:
                    images, labels = data[0].to(device), data[1].to(device)
                    outputs = model.classify(images)
                    _, predicted = torch.max(outputs, 1)
                    
                    correct = (predicted == labels).cpu()
                    labels_cpu = labels.cpu()

                    for cls in range(len(classes)):
                        cls_mask = (labels_cpu == cls)
                        class_correct[cls] += correct[cls_mask].sum().item()
                        class_total[cls] += cls_mask.sum().item()

        accuracy = 100 * sum(class_correct) / sum(class_total)

        for i in range(10):
            print(
                f"Accuracy of {classes[i]}: {100 * class_correct[i] / class_total[i]:.2f}%"
            )

        return accuracy
    
    accuracy = predict_on_loader(model, trainloader)

    print(f"Accuracy on train set: {accuracy:.2f}%")

    print("====================================")
    accuracy = predict_on_loader(model, testloader)

    print(f"Accuracy on test set: {accuracy:.2f}%")


    #predict single image
    image_path = "bird.png"
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model.classify(image)
        probs = preds[0]
        predicted_idx = torch.argmax(probs).item()
        predicted_class = classes[predicted_idx]
        confidence = probs[predicted_idx].item()

    print(f"\nPrediction for single image:")
    print(f"Class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}")

    return

if __name__ == "__main__":
    
    device = "cuda"
    model_path = "final_aggregate_model.pt"

    #

    plot_convergence_split_model(device)

    predict_with_combined_model(model_path, device)


