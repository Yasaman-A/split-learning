
"""
arg1 --> CONFIG_FILE_PATH
"""

# eg command: python server_splitnn_th_REPREQ.py 2 5555 cpu

import copy
import threading
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights
import torch.optim as optim
from torch.autograd import Variable
import time
import zmq
import torch
from ..lib import convert
import sys
import random
import yaml
import logging
# from objsize import get_deep_size


class Runner:
    def __init__(self, config_path) -> None:
        with open(config_path, "r") as yamlfile:
            self.config = yaml.load(yamlfile, Loader=yaml.FullLoader)
            print("Read successful")
    
    def run(self):
        client_total = self.config["client_total"]
        split_port = self.config["split_server"]["server_start_port"]
        device = self.config["device"]
        cut_layer = self.config["cut_layer"]
        epochs = self.config["epoch"]
        rnd = self.config["round"]

        if(device != 'cpu'):
                device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


        if (self.config["logging"]):
            # Create and configure logger
            logging.basicConfig(filename="./sf_server_" + str(client_total) + "_" + str(split_port) + "_" + device + "_" + cut_layer + "_" + epochs + "_" + rnd + ".log",
                                format='%(asctime)s %(message)s',
                                filemode='a')
            # Creating an object
            logger = logging.getLogger()
            # Setting the threshold of logger to DEBUG
            logger.setLevel(logging.INFO)
            logging.info('Parameters (SF_SERVER_LOG) ---------- [TOTAL_CLIENTS --> {}, STARTING_SERVER_PORT --> {}, DEVICE_TYPE --> {}, CUT_LAYER --> {}, EPOCHS --> {}, ROUNDS --> {}] ---------- '.format(str(client_total), str(split_port), device, str(cut_layer), str(epochs), str(rnd)))



        def average_weights(w, datasize):
            """
            Returns the average of the weights.
            """

            for i, data in enumerate(datasize):
                for key in w[i].keys():
                    w[i][key] *= data

            w_avg = copy.deepcopy(w[0])

            for key in w_avg.keys():
                for i in range(1, len(w)):
                    w_avg[key] += w[i][key]
                w_avg[key] = torch.div(w_avg[key], float(sum(datasize)))

            return w_avg



        # def worker_routine(url, context, client_no, r):
        #     """ Worker routine """

        #     # Socket to talk to dispatcher
        #     # context = zmq.Context()
        #     socket = context.socket(zmq.REP)

        #     # socket.connect(worker_url)
        #     # socket.connect("tcp://*:5555")
        #     socket.bind(url)


        #     iterations = socket.recv()
        #     recv_iterations = int(iterations.decode())
        #     print(recv_iterations)

        #     msg = "give_datasize_length"
        #     send_msg = msg.encode()
        #     socket.send(send_msg)

        #     recv_dataset_size = socket.recv()
        #     dataset_size = int(recv_dataset_size.decode())
        #     print(dataset_size)

        #     msg = "Starting the server"
        #     send_msg = msg.encode()
        #     socket.send(send_msg)



        #     epoch_start_time = time.time()
        #     running_loss = 0.0
        #     # for i, data in enumerate(trainloader, 0):
        #     for j in range(recv_iterations):
        #         step_start_time = time.time()
        #         print("***CL - {}*** {}".format(client_no, j))

        #         server_optimizer.zero_grad()

        #         recv_labels = socket.recv()
        #         numpy_labels = bytes_to_array(recv_labels)
        #         labels = torch.from_numpy(numpy_labels)
        #         labels = labels.to(device)
        #         # print("labels_recieved")

        #         ##dummy......
        #         socket.send(send_msg)

        #         # print("inside for for")
        #         recv_serv_inputs = socket.recv()
        #         numpy_server_inputs = bytes_to_array(recv_serv_inputs)
        #         server_inputs = torch.from_numpy(numpy_server_inputs)
        #         server_inputs = server_inputs.to(device)
        #         # print("data_recieved")

        #         ###################################################################################################

        #         # Simulation of server part is happening in this portion
        #         # Server part
        #         server_inputs = Variable(server_inputs, requires_grad=True)
        #         outputs = server_model(server_inputs)
        #         loss = criterion(outputs, labels)
        #         loss.backward()

        #         # server optimization
        #         server_optimizer.step()

        #         transfer_loss = loss.detach().clone()
        #         bytes_loss = array_to_bytes(transfer_loss.cpu())
        #         socket.send(bytes_loss)
        #         # print("loss_sent")

        #         step_end_time = time.time()
        #         total_one_step_time = step_end_time - step_start_time
        #         print("***CL - {}***  SERVER_TOTAL_ONE_STEP_TIME = {:.3f}".format(client_no, total_one_step_time))
        #         logging.info('***CL - {}***  SERVER_TOTAL_ONE_STEP_TIME = {:.3f}'.format(client_no, total_one_step_time))


        #         ################################################################################

        #     epoch_end_time = time.time()
        #     total_one_epoch_time = epoch_end_time - epoch_start_time
        #     print("***CL - {}***  SERVER_TOTAL_ONE_EPOCH_TIME = {:.3f}" .format(client_no, total_one_epoch_time))
        #     logging.info('***CL - {}***  SERVER_TOTAL_ONE_EPOCH_TIME = {:.3f}'.format(client_no, total_one_epoch_time))

        #     ##################################################################################################################

        #     ##*****************************************************************************************************************
        #     ##*****************************************************************************************************************
        #     ##*****************************************************************************************************************

        #     print("Worker done******************************")

        #     socket.close()


        def main():
            """ server routine """

            class ResNet18Server(nn.Module):
                """docstring for ResNet"""

                def __init__(self, config):
                    super(ResNet18Server, self).__init__()
                    self.logits = config["logits"]
                    self.cut_layer = config["cut_layer"]

                    # self.model = models.resnet18(pretrained=True)
                    # Newer version of (pretrained=True)
                    self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

                    num_ftrs = self.model.fc.in_features
                    # Explain this part
                    self.model.fc = nn.Sequential(nn.Flatten(),
                                                  nn.Linear(num_ftrs, self.logits))

                    self.model = nn.ModuleList(self.model.children())
                    self.model = nn.Sequential(*self.model)

                def forward(self, x):
                    for i, l in enumerate(self.model):
                        # Explain this part
                        if i <= cut_layer:
                            continue
                        x = l(x)
                    return nn.functional.softmax(x, dim=1)
                
                def change_cut(self, cut_layer):
                    self.cut_layer = cut_layer

            config = {"cut_layer": 3
                      #cut_layer
                      , "logits": 10}
            # client_model = ResNet18Client(config).to(device)
            server_model = ResNet18Server(config).to(device)

            criterion = nn.CrossEntropyLoss()
            # client_optimizer = optim.SGD(client_model.parameters(), lr=0.01, momentum=0.9)
            server_optimizer = optim.SGD(
                server_model.parameters(), lr=0.01, momentum=0.9)



            # total_clients = client_total
            port_no = split_port
            connection_url = ["tcp://*:" +str(port_no+i) for i in range(client_total)]
            # connection_url = ["tcp://*:5555", "tcp://*:5556"]

            num_rounds = rnd
            num_epochs = epochs

            context = zmq.Context()

            sockets = []

            for url in connection_url:
                socket = context.socket(zmq.REP)
                socket.bind(url)
                sockets.append(socket)

            training_start_time = time.time()
            for r in range(num_rounds):
                print("New round started..")

                for epoch in range(num_epochs):
                
                    random.shuffle(sockets)
                    # print("NEW_SHUFFLED_CLIENTS_FOR_THIS_EPOCH --> {}".format(connection_url))
                    # logging.info("NEW_SHUFFLED_CLIENTS_FOR_THIS_EPOCH --> {}".format(connection_url))
                    for cl in range(client_total):
                        client_no = cl + 1

                        """ Worker routine """
                        #use socket for random client n
                        socket = sockets[cl]

                        cut_layer = int(socket.recv().decode())
                        print(cut_layer)
                        server_model.change_cut(cut_layer)
                        socket.send(b'ack')

                        iterations = socket.recv()
                        recv_iterations = int(iterations.decode())
                        print(recv_iterations)

                        msg = "give_datasize_length"
                        send_msg = msg.encode()
                        socket.send(send_msg)

                        recv_dataset_size = socket.recv()
                        dataset_size = int(recv_dataset_size.decode())
                        print(dataset_size)

                        # After each iteration in client there's a message, ? if needed
                        msg = "Starting the server"
                        send_msg = msg.encode()
                        socket.send(send_msg)

                        epoch_start_time = time.time()
                        running_loss = 0.0
                        # for i, data in enumerate(trainloader, 0):
                        for j in range(recv_iterations):
                            step_start_time = time.time()
                            print("***CL - {}*** {}".format(client_no, j))

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


                            #dummy
                            socket.recv()

                            for layer, param in enumerate(server_model.parameters()):   
                                if param.grad is not None:
                                    grad_numpy = param.grad.detach().cpu().numpy()
                                else:
                                    grad_numpy = torch.zeros_like(param).cpu().numpy()
                                
                                grad_bytes = convert.array_to_bytes(grad_numpy)
                                socket.send(grad_bytes)

                                #dummy
                                socket.recv()

                            socket.send(b"ack")

                            step_end_time = time.time()
                            total_one_step_time = step_end_time - step_start_time
                            print("***CL - {}***  SERVER_TOTAL_ONE_STEP_TIME = {:.3f}".format(client_no, total_one_step_time))
                            logging.info(
                                '***CL - {}***  SERVER_TOTAL_ONE_STEP_TIME = {:.3f}'.format(client_no, total_one_step_time))

                            ################################################################################

                        epoch_end_time = time.time()
                        total_one_epoch_time = epoch_end_time - epoch_start_time
                        print("***CL - {}***  SERVER_TOTAL_ONE_EPOCH_TIME = {:.3f}" .format(client_no,
                            total_one_epoch_time))
                        logging.info(
                            '***CL - {}***  SERVER_TOTAL_ONE_EPOCH_TIME = {:.3f}'.format(client_no, total_one_epoch_time))

                        ##################################################################################################################

                        ##*****************************************************************************************************************
                        ##*****************************************************************************************************************
                        ##*****************************************************************************************************************

                        print("Worker done******************************")

                        ##################################################
                        ##################################################

                    print("All clients served..")

                model_save_name = "./server_fedAvg_model_r" + str(r) + "_" + str(client_total) + "_" + str(split_port)  + "_" + device + "_" + str(cut_layer) + "_" + str(epochs) + "_" + str(rnd) + ".pt"
                torch.save(server_model.state_dict(), model_save_name)
                print("MODEL_SAVED.")


            training_end_time = time.time()
            training_time = training_end_time - training_start_time
            print("SERVER_TOTAL_TRAINING_TIME = {:.3f}" .format(training_time))
            logging.info('SERVER_TOTAL_TRAINING_TIME = {:.3f}'.format(training_time))


            print("All rounds ended..")
            
            context.term()

        main()
