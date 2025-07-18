
"""
arg1 --> CONFIG_FILE_PATH
"""

# eg command: python server_splitnn_th_REPREQ.py 2 5555 cpu

import torch.nn as nn
from torchvision import models
import torch.optim as optim
from torch.autograd import Variable
import time
import zmq
import torch
from ..lib import convert
import os
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
        epochs = self.config["epoch"]
        rnd = self.config["round"]

        if(device != 'cpu'):
                device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


        if (self.config["logging"]):
            log_path = os.path.join(
                    self.config.get("log_dir", "./"),
                    f"./sf_server_{client_total}_{split_port}_{device}_"
                    f"_{epochs}_{rnd}.log"
                    )
            logging.basicConfig(
                filename=log_path,
                format='%(asctime)s %(message)s',
                filemode='a'
            )
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            logging.info(
                f"Parameters (SF_SERVER_LOG) ---------- "
                f"[TOTAL_CLIENTS --> {client_total}, STARTING_SERVER_PORT --> {split_port}, "
                f"DEVICE_TYPE --> {device}, "
                f"EPOCHS --> {epochs}, ROUNDS --> {rnd}] ----------"
            )


        def main():
            """ server routine """

            class ResNet18Server(nn.Module):
                """docstring for ResNet"""

                def __init__(self, config):
                    super(ResNet18Server, self).__init__()
                    self.logits = config["logits"]
                    self.cut_layer = config["cut_layer"]

                    self.model = models.resnet18(weights=None)

                    num_ftrs = self.model.fc.in_features
                    self.model.fc = nn.Sequential(nn.Flatten(),
                                                  nn.Linear(num_ftrs, self.logits))

                    self.layers = list(self.model.children())

                def forward(self, x):
                    for i, l in enumerate(self.layers):
                        if i <= cut_layer:
                            continue
                        x = l(x)
                    return x
                
                def change_cut(self, cut_layer):
                    self.cut_layer = cut_layer

            config = {"cut_layer": 1, "logits": 10}
            server_model = ResNet18Server(config).to(device)

            criterion = nn.CrossEntropyLoss()
            server_optimizer = optim.SGD(
                server_model.parameters(), lr=0.01, momentum=0.9)

            port_no = split_port
            connection_url = ["tcp://*:" +str(port_no+i) for i in range(client_total)]

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

                        msg = "Starting the server"
                        send_msg = msg.encode()
                        socket.send(send_msg)

                        epoch_start_time = time.time()
                        running_loss = 0.0
                        # for i, data in enumerate(trainloader, 0):
                        for j in range(recv_iterations):
                            step_start_time = time.time()
                            print("***CL - {}*** {}".format(client_no, j))

                            #recieve labels
                            recv_labels = socket.recv()
                            numpy_labels = convert.bytes_to_array(recv_labels)
                            labels = torch.from_numpy(numpy_labels)
                            labels = labels.to(device)

                            ##dummy......
                            socket.send(send_msg)

                            # get client activations
                            recv_serv_inputs = socket.recv()
                            numpy_server_inputs = convert.bytes_to_array(recv_serv_inputs)
                            server_inputs = torch.from_numpy(numpy_server_inputs)
                            server_inputs = server_inputs.to(device)

                            server_inputs = Variable(server_inputs, requires_grad=True)
                            outputs = server_model(server_inputs)
                            
                            server_optimizer.zero_grad()
                            loss = criterion(outputs, labels)
                            loss.backward()

                            #send gradients back to client
                            transfer_loss = server_inputs.grad.clone().detach()
                            server_optimizer.step()

                            bytes_loss = convert.array_to_bytes(transfer_loss.cpu())
                            socket.send(bytes_loss)


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

                model_save_name = os.path.join(
                    self.config.get("model_dir", "./"),
                    f"./server_fedAvg_model_r{r}_{client_total}_{split_port}_"
                    f"{device}_{epochs}_{rnd}.pt"
                )
                torch.save(server_model.state_dict(), model_save_name)
                print("MODEL_SAVED.")


            training_end_time = time.time()
            training_time = training_end_time - training_start_time
            print("SERVER_TOTAL_TRAINING_TIME = {:.3f}" .format(training_time))
            logging.info('SERVER_TOTAL_TRAINING_TIME = {:.3f}'.format(training_time))


            print("All rounds ended..")
            
            for socket in sockets:
                socket.close()

            context.term()

        main()
