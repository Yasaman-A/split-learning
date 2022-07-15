
"""
arg1 --> total number of clients
arg2 --> STARTING_PORT_NO
arg3 --> 'cpu' or 'gpu'
"""

# eg command: python server_splitnn_th_REPREQ.py 2 5555 cpu

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
from convert import bytes_to_array, array_to_bytes
import sys


def worker_routine(url, context):
    """ Worker routine """

    # Socket to talk to dispatcher
    # context = zmq.Context()
    socket = context.socket(zmq.REP)

    # socket.connect(worker_url)
    # socket.connect("tcp://*:5555")
    socket.bind(url)

    ##*****************************************************************************************************************
    ##*****************************************************************************************************************
    ##*****************************************************************************************************************

    ##################################################################################################################

    # from mpi4py import MPI
    # import time
    # import logging

    # logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    #                     datefmt='%d-%m-%Y:%H:%M:%S',
    #                     level=logging.INFO,
    #                     filename='logs.txt')

    # # logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    # logging.info('Code started..')

    if(sys.argv[3] == 'cpu'):
        device = 'cpu'
    else:
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

    config = {"cut_layer": 3, "logits": 10}
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




    ##*****************************************************************************************************************
    ##*****************************************************************************************************************
    ##*****************************************************************************************************************

    print("Worker done******************************")


def main():
    """ server routine """

    total_threads = int(sys.argv[1])
    port_no = int(sys.argv[2])
    connection_url = ["tcp://*:" +str(port_no+i) for i in range(total_threads)]
    # connection_url = ["tcp://*:5555", "tcp://*:5556"]
    context = zmq.Context()

    thrs = []
    # Launch pool of worker threads
    for i in range(total_threads):  # this defines how many clients can connect
        thread = threading.Thread(target=worker_routine, args=(connection_url[i], context))
        thrs.append(thread)
        thread.start()

    for thread in thrs:         ##have to check when it will run all epochs..
        thread.join()

    
    print("All threads ended..")

if __name__ == "__main__":
    main()
