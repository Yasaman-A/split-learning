#
#   SL Client
#

import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
from torch.autograd import Variable
import time
import sys
import zmq
import torch
from convert import array_to_bytes, bytes_to_array
import logging

# Create and configure logger
logging.basicConfig(filename="client1_newfile.log",
                    format='%(asctime)s %(message)s',
                    filemode='a')

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.INFO)


if __name__ == '__main__':
    context = zmq.Context()

    #  Socket to talk to server
    socket = context.socket(zmq.REQ)
    url = "tcp://"+sys.argv[1] + ":"+sys.argv[2]
    socket.connect(url)

    if(sys.argv[3] == 'cpu'):
        device = 'cpu'
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    # device = 'cpu'

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # CIFAR10 is a dataset of natural images consisting of 50k training images and 10k test
    # Every image is labelled with one of the following class
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=2)

    # Explain nn.Module and explain the forward and backward pass

    class ResNet18Client(nn.Module):
        """docstring for ResNet"""

        # Explain initialize (listing the neural network architecture and other related parameters)
        def __init__(self, config):
            super(ResNet18Client, self).__init__()
            # Explain this line
            self.cut_layer = config["cut_layer"]

            # Explain this line
            self.model = models.resnet18(pretrained=False)

            self.model = nn.ModuleList(self.model.children())
            self.model = nn.Sequential(*self.model)

        # Explain forward (actually used during the execution of the neural network at runtime)
        def forward(self, x):
            for i, l in enumerate(self.model):
                if i > self.cut_layer:
                    break
                x = l(x)
            return x

    config = {"cut_layer": 3, "logits": 10}
    client_model = ResNet18Client(config).to(device)

    criterion = nn.CrossEntropyLoss()
    client_optimizer = optim.SGD(client_model.parameters(), lr=0.01, momentum=0.9)

    iterations = len(trainloader)
    print(iterations)
    send_iterations = str(iterations).encode()
    socket.send(send_iterations)

    names = socket.recv()
    recv_names = names.decode()
    # print(recv_names)


    log_steps = 50
    num_epochs = 2
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            print(i)
            inputs, labels = data[0].to(device), data[1].to(device)

            client_optimizer.zero_grad()

            # print("LABELS", type(labels))
            bytes_labels = array_to_bytes(labels.cpu())
            socket.send(bytes_labels)
            # print("labels_sent")
            
            ##dummy......
            msgs = socket.recv()
            recv_msgs = msgs.decode()
            # print(recv_names)

            # Client part
            activations = client_model(inputs)
            server_inputs = activations.detach().clone()
            
            # print("inside for for...")
            bytes_server_inputs = array_to_bytes(server_inputs.cpu())
            socket.send(bytes_server_inputs)

            ###################################################################################################
            recv_loss = socket.recv()
            numpy_loss = bytes_to_array(recv_loss)
            loss = torch.from_numpy(numpy_loss)
            loss = loss.to(device)

            client_optimizer.step()

            running_loss += loss.item()

            if i % log_steps == log_steps-1:
                print('[{}, {}] loss: {}'.format(epoch + 1, i + 1, running_loss / log_steps))
                logging.info('[{}, {}] loss: {}'.format(epoch + 1, i + 1, running_loss / log_steps))
                running_loss = 0.0


#################################################################################################################################