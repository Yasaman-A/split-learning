import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
from torch.autograd import Variable
import time
import zmq
import torch
from convert import array_to_bytes, bytes_to_array, ordered_dict_to_bytes, bytes_to_dict

import logging
from sys import getsizeof
# from objsize import get_deep_size

# Create and configure logger
logging.basicConfig(filename="client2_newfile.log",
                    format='%(asctime)s %(message)s',
                    filemode='a')

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.INFO)


if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = 'cpu'
    print(device)

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
    client_optimizer = optim.SGD(
        client_model.parameters(), lr=0.01, momentum=0.9)


    client_num = 1                             ## Very Imp, type manually type client num
    num_epochs = 50                            ## to be manually fixed..
    for epoch in range(num_epochs):
        print("EPOCH NO in client:- ", epoch)
        running_loss = 0.0


        ####################################### NEW CONTEXT #########################
        ####
        context = zmq.Context()

        #  Socket to talk to server
        print("Connecting to hello world server…")
        socket = context.socket(zmq.REQ)
        socket.connect("tcp://localhost:5556")
        # socket.connect("tcp://35.237.244.119:5556")


        iterations = len(trainloader)
        print(iterations)
        send_iterations = str(iterations).encode()
        socket.send(send_iterations)

        # names = socket.recv()
        # recv_names = names.decode()
        # # print(recv_names)

        ############# LOAD WEIGHTS FROM SERVER ################
        print("Waiting for weights....")
        weights = socket.recv()
        print("Weights recieved")

        if client_num == 0 and epoch == 0:
            decoded_msg = weights.decode()
            if decoded_msg == "sagar":
                None

        else:
            # weights = socket.recv()
            numpy_weights = bytes_to_dict(weights)
            client_model.load_state_dict(numpy_weights)
            print("MODEL LOADED")

        #######################################################

        for i, data in enumerate(trainloader, 0):
            ## Extra #####
            if i == 5:
                break
            ##############
            print(i)
            inputs, labels = data[0].to(device), data[1].to(device)

            client_optimizer.zero_grad()

            # print("LABELS", type(labels))
            bytes_labels = array_to_bytes(labels.cpu())
            socket.send(bytes_labels)
            # print("labels_sent")

            ##dummy......
            names = socket.recv()
            recv_names = names.decode()
            # print(recv_names)

            # Client part
            activations = client_model(inputs)
            server_inputs = activations.detach().clone()

            # print("inside for for...")
            bytes_server_inputs = array_to_bytes(server_inputs.cpu())
            socket.send(bytes_server_inputs)
            # print("data_sent")

            ###################################################################################################

            # # Simulation of server part is happening in this portion
            # # Server part
            # server_inputs = Variable(server_inputs, requires_grad=True)
            # outputs = server_model(server_inputs)
            # loss = criterion(outputs, labels)
            # loss.backward()

            # # server optimization
            # server_optimizer.step()
            ################################################################################

            recv_loss = socket.recv()
            numpy_loss = bytes_to_array(recv_loss)
            loss = torch.from_numpy(numpy_loss)
            loss = loss.to(device)
            # print("loss_recieved")

            # Simulation of Client Happening in this portion
            # Client optimization

            # activations.backward(server_inputs.grad)
            client_optimizer.step()

            running_loss += loss.item()

            if i % 200 == 199:
                print('[{}, {}] loss: {}'.format(
                    epoch + 1, i + 1, running_loss / 200))
                logging.info('[{}, {}] loss: {}'.format(
                    epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0


        ######### SAVE WEIGHTS TO SERVER #########

        ## send client weights..
        weights = client_model.state_dict()
        print(type(weights))
        print("SIze of model weights in bytes is:-", getsizeof(weights))
        bytes_weights = ordered_dict_to_bytes(weights)
        # time.sleep(10)
        socket.send(bytes_weights)
        del weights
        del bytes_weights

        names = socket.recv()
        recv_names = names.decode()
        print(recv_names)


        ##########################################


        socket.close()
        context.term()

        # time.sleep(5)
        ####################################### CONTEXT CLOSE #####################


#################################################################################################################################
