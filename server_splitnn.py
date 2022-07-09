#
#   SL Server
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
from convert import bytes_to_array, array_to_bytes


if __name__ == '__main__':
    
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    url = "tcp://*:"+sys.argv[1]
    socket.bind(url)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = 'cpu'
    print(device)


    class ResNet18Server(nn.Module):
        """docstring for ResNet"""

        def __init__(self, config):
            super(ResNet18Server, self).__init__()
            self.logits = config["logits"]
            self.cut_layer = config["cut_layer"]

            self.model = models.resnet18(pretrained=False)
            num_ftrs = self.model.fc.in_features
            # Explain this part
            self.model.fc = nn.Sequential(nn.Flatten(),
                                          nn.Linear(num_ftrs, self.logits))

            self.model = nn.ModuleList(self.model.children())
            self.model = nn.Sequential(*self.model)

        def forward(self, x):
            for i, l in enumerate(self.model):
                # Explain this part
                if i <= self.cut_layer:
                    continue
                x = l(x)
            return nn.functional.softmax(x, dim=1)

    config = {"cut_layer": 3, "logits": 10}
    # client_model = ResNet18Client(config).to(device)
    server_model = ResNet18Server(config).to(device)

    criterion = nn.CrossEntropyLoss()
    # client_optimizer = optim.SGD(client_model.parameters(), lr=0.01, momentum=0.9)
    server_optimizer = optim.SGD(server_model.parameters(), lr=0.01, momentum=0.9)

    iterations = socket.recv()
    recv_iterations = int(iterations.decode())
    print(recv_iterations)

    msg= "Starting the server"
    send_msg = msg.encode()
    socket.send(send_msg) 

    num_epochs = 50
    for epoch in range(num_epochs):
        running_loss = 0.0
        # for i, data in enumerate(trainloader, 0):
        for j in range(recv_iterations):
            print(j)

            server_optimizer.zero_grad()

            recv_labels = socket.recv()
            numpy_labels = bytes_to_array(recv_labels)
            labels = torch.from_numpy(numpy_labels)
            labels = labels.to(device)
            # print("labels_recieved")

            ##dummy......
            socket.send(send_msg) 


            # print("inside for for")
            recv_serv_inputs = socket.recv()
            numpy_server_inputs = bytes_to_array(recv_serv_inputs)
            server_inputs = torch.from_numpy(numpy_server_inputs)
            server_inputs = server_inputs.to(device)
            # print("data_recieved")

            ###################################################################################################

            # Simulation of server part is happening in this portion
            # Server part
            server_inputs = Variable(server_inputs, requires_grad=True)
            outputs = server_model(server_inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # server optimization
            server_optimizer.step()

            transfer_loss = loss.detach().clone()
            bytes_loss = array_to_bytes(transfer_loss.cpu())
            socket.send(bytes_loss)
            # print("loss_sent")

            ################################################################################


##################################################################################################################
