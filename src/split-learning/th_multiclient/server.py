
"""
arg1 --> CONFIG_FILE_PATH
"""

import threading
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
from ..lib import convert
import sys
import yaml

class Runner:
    def __init__(self, config_path) -> None:
        with open(config_path, "r") as yamlfile:
            self.config = yaml.load(yamlfile, Loader=yaml.FullLoader)
            print("Read successful")
    
    def run(self):
        client_total = self.config["client_total"]
        split_port = self.config["split_server"]["server_start_port"]
        device = self.config["device"]
        num_epochs = self.config["epoch"]
        cut_layer = self.config["cut_layer"]


        def worker_routine(url, context, device):
            """ Worker routine """

            # Socket to talk to dispatcher
            # context = zmq.Context()
            socket = context.socket(zmq.REP)

            # socket.connect(worker_url)
            # socket.connect("tcp://*:5555")
            socket.bind(url)

            if(device != 'cpu'):
                device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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

            config = {"cut_layer": cut_layer, "logits": 10}
            # client_model = ResNet18Client(config).to(device)
            server_model = ResNet18Server(config).to(device)

            criterion = nn.CrossEntropyLoss()
            # client_optimizer = optim.SGD(client_model.parameters(), lr=0.01, momentum=0.9)
            server_optimizer = optim.SGD(
                server_model.parameters(), lr=0.01, momentum=0.9)

            iterations = socket.recv()
            recv_iterations = int(iterations.decode())
            print(recv_iterations)

            msg = "Starting the server"
            send_msg = msg.encode()
            socket.send(send_msg)

            # num_epochs = 50
            for epoch in range(num_epochs):
                running_loss = 0.0
                # for i, data in enumerate(trainloader, 0):
                for j in range(recv_iterations):
                    print(j)

                    server_optimizer.zero_grad()

                    recv_labels = socket.recv()
                    numpy_labels = convert.bytes_to_array(recv_labels)
                    labels = torch.from_numpy(numpy_labels)
                    labels = labels.to(device)
                    # print("labels_recieved")

                    ##dummy......
                    socket.send(send_msg)

                    # print("inside for for")
                    recv_serv_inputs = socket.recv()
                    numpy_server_inputs = convert.bytes_to_array(recv_serv_inputs)
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
                    bytes_loss = convert.array_to_bytes(transfer_loss.cpu())
                    socket.send(bytes_loss)
                    # print("loss_sent")
            print("Worker done******************************")


        """ server routine """

        total_threads = client_total
        port_no = int(split_port)
        connection_url = ["tcp://*:" +str(split_port+i) for i in range(client_total)]
        # connection_url = ["tcp://*:5555", "tcp://*:5556"]
        context = zmq.Context()

        thrs = []
        # Launch pool of worker threads
        for i in range(total_threads):  # this defines how many clients can connect
            thread = threading.Thread(target=worker_routine, args=(connection_url[i], context, device))
            thrs.append(thread)
            thread.start()

        for thread in thrs:         ##have to check when it will run all epochs..
            thread.join()

    
        print("All threads ended..")
