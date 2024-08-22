"""
arg1 --> CONFIG_FILE_PATH
arg2 --> CLIENT_ID
"""

from locale import atoi
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
# from ..utils.convert import array_to_bytes, bytes_to_array, ordered_dict_to_bytes, bytes_to_dict
from ..lib import convert
import sys
from sys import getsizeof
import numpy as np
import pickle
import urllib.request
import os
import yaml
import logging
# from objsize import get_deep_size


class Runner:
    def __init__(self, config_path) -> None:
        self.client_id = -1
        with open(config_path, "r") as yamlfile:
            self.config = yaml.load(yamlfile, Loader=yaml.FullLoader)
            print("Read successful")
    
    def run(self):
        split_address = self.config["split_server"]["server_ip"]
        split_port = self.config["split_server"]["server_start_port"]+self.client_id-1
        fed_port = self.config["fed_server"]["server_start_port"]+self.client_id-1
        log_steps = self.config["log_steps"]
        num_epochs = int(self.config["epoch"])
        output_file = self.config["data_server"]["output_file"]
        rnd = self.config["round"]


        if (self.config["logging"]):
            # Create and configure logger
            logging.basicConfig(filename= str(self.client_id) + "_" + str(config["cut_layer"]) + "_" + str(config["epoch"]) + "_" + str(config["rnd"]) + "_" + str(config["batch_size"])+ "_" + config["device"]+".log",
                                format='%(asctime)s %(message)s',
                                filemode='a')
            # Creating an object
            logger = logging.getLogger()
            # Setting the threshold of logger to DEBUG
            logger.setLevel(logging.INFO)


        if(self.config["device"] == 'cpu'):
            device = 'cpu'
        else:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(device)

        transform = transforms.Compose(
            [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # CIFAR10 is a dataset of natural images consisting of 50k training images and 10k test
        # Every image is labelled with one of the following class
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        batch_size = self.config["batch_size"]

        ## Dataloader Splitting....
        if (self.config["split_type"] == 'n'):
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                    download=True, transform=transform)

            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                      shuffle=True, num_workers=2)
            print('trainloader:' + str(len(trainloader)))
            datasetsize_used = len(trainset)
        elif (self.config["split_type"] == 's'):
            if os.path.exists(output_file+str(self.client_id)):
                os.remove(output_file+str(self.client_id))
            print(self.config["data_server"]["server_address"]+"/"+output_file)
            urllib.request.urlretrieve(self.config["data_server"]["server_address"]+"/"+output_file, output_file+str(self.client_id))
            with open(output_file+str(self.client_id), 'rb') as handle:
                trainloaders = pickle.load(handle)
                trainloader = trainloaders[self.client_id]
            datasetsize_used = len(trainloader.dataset)

        else:
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                    download=True, transform=transform)
            dataset_size = len(trainset)                         # 50k images
            total_indices = list(range(dataset_size))
            list_of_indices = np.array_split(
                np.array(total_indices), int(self.config["split_type"]))
            [l.tolist() for l in list_of_indices]

            use_indices = list_of_indices[self.client_id]
            datasetsize_used = len(use_indices)
            print('use_indices:' + str(use_indices))

            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=2, sampler=use_indices)  # shuffle=True (mutually exclusive with sampler)
            print('trainloader:' + str(len(trainloader)))
        # exit()

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
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
                # self.model = models.resnet18(pretrained=True)
                # Newer version of (pretrained=True)
                self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

                self.model = nn.ModuleList(self.model.children())
                self.model = nn.Sequential(*self.model)

            # Explain forward (actually used during the execution of the neural network at runtime)
            def forward(self, x):
                for i, l in enumerate(self.model):
                    if i > self.cut_layer:
                        break
                    x = l(x)
                return x

        config = {"cut_layer": int(self.config["cut_layer"]), "logits": 10}
        client_model = ResNet18Client(config).to(device)

        criterion = nn.CrossEntropyLoss()
        client_optimizer = optim.SGD(
            client_model.parameters(), lr=0.01, momentum=0.9)


        training_start_time = time.time()
        num_rounds = rnd
    
        for r in range(num_rounds):
            if r > 0:
                client_model.load_state_dict(global_numpy_weights)
                print("GLOBAL_CLIENT_WEIGHTS_LOADED")
                del global_numpy_weights



            context = zmq.Context()

            #  Socket to talk to server
            print("Connecting to split server…")
            socket = context.socket(zmq.REQ)

            url = split_address + ":"+ str(split_port)
            socket.connect(url)
            # socket.connect("tcp://35.237.244.119:5555")


            iterations = len(trainloader)
            print(iterations)
            send_iterations = str(iterations).encode()
            socket.send(send_iterations)

            names = socket.recv()
            recv_names = names.decode()
            # print(recv_names)

            print(datasetsize_used)
            send_dataset_size = str(datasetsize_used).encode()
            socket.send(send_dataset_size)

            names = socket.recv()
            recv_names = names.decode()

            # send_iterations = str(iterations).encode()
            # socket.send(send_iterations)
            # exit()

            # log_steps = config[log_steps]



            for epoch in range(num_epochs):
                epoch_start_time = time.time()
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    step_start_time = time.time()
                    print(r, epoch, i)
                    inputs, labels = data[0].to(device), data[1].to(device)

                    client_optimizer.zero_grad()

                    # print("LABELS", type(labels))
                    bytes_labels = convert.array_to_bytes(labels.cpu())
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
                    bytes_server_inputs = convert.array_to_bytes(server_inputs.cpu())
                    server_work_time_start = time.time()
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
                    server_work_time_end = time.time()
                    numpy_loss = convert.bytes_to_array(recv_loss)
                    loss = torch.from_numpy(numpy_loss)
                    loss = loss.to(device)
                    # print("loss_recieved")

                    # Simulation of Client Happening in this portion
                    # Client optimization

                    # activations.backward(server_inputs.grad)
                    client_optimizer.step()

                    running_loss += loss.item()

                    if i % log_steps == log_steps-1:
                        print('[{}, {}] loss: {:.3f}'.format(
                            epoch + 1, i + 1, running_loss / log_steps))
                        logging.info('[{}, {}] loss:  {:.3f}'.format(
                            epoch + 1, i + 1, running_loss / log_steps))
                        running_loss = 0.0

                    step_end_time = time.time()
                    total_one_step_time = step_end_time - step_start_time
                    server_work_time = server_work_time_end - server_work_time_start
                    print("CLIENT_TOTAL_ONE_STEP_TIME = {:.3f}, SERVER_WORK_TIME = {:.3f}" .format(
                        total_one_step_time, server_work_time))
                    logging.info("CLIENT_TOTAL_ONE_STEP_TIME = {:.3f}    , SERVER_WORK_TIME = {:.3f}" .format(
                        total_one_step_time, server_work_time))

                epoch_end_time = time.time()
                total_one_epoch_time = epoch_end_time - epoch_start_time
                print("CLIENT_TOTAL_ONE_EPOCH_TIME = ", total_one_epoch_time)
                logging.info('CLIENT_TOTAL_ONE_EPOCH_TIME = {:.3f}'.format(
                    total_one_epoch_time))


            socket.close()
            context.term()

            # model_save_name = "./client_thread_model_r" + str(r) + "_" + str(self.client_id) + "_" + str(split_port) + "_" + config["device"] + "_" + config["cut_layer"] + "_" + config["epoch"] + "_" + config["split_type"] + "_" + str(self.client_id) + "_" + config["batch_size"] + "_" + config["round"] + "_" + str(fed_port) + ".pt"
            # Consistent use of self.config (resolving KeyError: 'device')
            model_save_name = ("./client_thread_model_r" + str(r) + "_" + str(self.client_id) + "_" + str(split_port) + "_" + 
                               self.config.get("device", "cpu") + "_" + str(self.config["cut_layer"]) + "_" + 
                               str(self.config["epoch"]) + "_" + self.config["split_type"] + "_" + str(self.client_id) + "_" + 
                               str(self.config["batch_size"]) + "_" + str(self.config["round"]) + "_" + str(fed_port) + ".pt")

            torch.save(client_model.state_dict(), model_save_name)
            print("***TH - {}***  MODEL_SAVED." .format(self.client_id))



            ############################################################
            ########### Sending model to fedServer #####################
            ############################################################

            context1 = zmq.Context()

            #  Socket to talk to server
            print("Connecting to fed_avg server to aggregate weights …")
            socket1 = context1.socket(zmq.REQ)
            url = config["fed_server"]["server_ip"] + ":"+ fed_port
            socket1.connect(url)
            # socket.connect("tcp://35.237.244.119:5555")

            weights = client_model.state_dict()
            # print(type(weights))
            print("SIze of model weights (before) in bytes is:-", getsizeof(weights))
            bytes_weights = convert.ordered_dict_to_bytes(weights)
            print("SIze of model weights (after) in bytes is:-",
                  getsizeof(bytes_weights))
            # time.sleep(10)
            socket1.send(bytes_weights)

            ## dummy recv
            names = socket1.recv()
            recv_names = names.decode()

            ## send dataset size for weighted avg
            socket1.send(send_dataset_size)

            ## dummy recv
            names = socket1.recv()
            recv_names = names.decode()

            del weights
            del bytes_weights

            socket1.close()
            context1.term()

            #############################################################
            ######### Recieving global model from fedServer #############
            #############################################################

            context2 = zmq.Context()

            #  Socket to talk to server
            print("Connecting to fed_avg server to recv global weights…")
            socket2 = context2.socket(zmq.REQ)
            url = config["split_server"]["server_ip"] + ":"+ fed_port
            socket2.connect(url)
            # socket.connect("tcp://35.237.244.119:5555")

            msg = "send_global_weights"
            send_msg = msg.encode()
            socket2.send(send_msg)

            global_weights = socket2.recv()
            print("Global weights recieved from fedServer")
            print("SIze of global model weights (before) in bytes is:-", getsizeof(global_weights))
            global_numpy_weights = convert.bytes_to_dict(global_weights)
            print("SIze of global model weights (after) in bytes is:-", getsizeof(global_numpy_weights))

            socket2.close()
            context2.term()



        training_end_time = time.time()
        training_time = training_end_time - training_start_time
        print("CLIENT_TOTAL_TRAINING_TIME = ", training_time)
        logging.info('CLIENT_TOTAL_TRAINING_TIME = {:.3f}'.format(training_time))

        # model_save_name = "./client_thread_model_" + sys.argv[4] + "_" + sys.argv[2] + "_" + sys.argv[3] + "_" + \
        #     sys.argv[5] + "_" + sys.argv[6] + "_" + sys.argv[7] + \
        #     "_" + sys.argv[8] + "_" + sys.argv[9] + ".pt"
        # torch.save(client_model.state_dict(), model_save_name)
        # print("MODEL_SAVED.")

#################################################################################################################################
