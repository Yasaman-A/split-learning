
"""
arg1 --> total number of clients
arg2 --> STARTING_PORT_NO
arg3 --> 'cpu' or 'gpu'
arg4 --> cut_layer(same for all) or 'list' at the end
arg5 --> epochs
arg6 --> round
arg7..x --> cut layers for all clients
"""

# eg command: python server_splitnn_th_REPREQ.py 2 5555 cpu

import copy
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
from convert import array_to_bytes, bytes_to_array
import sys
from custom_model_avg import server_custom_avg_model

import logging
# from objsize import get_deep_size

# Create and configure logger
logging.basicConfig(filename="./cc_sf_server_" + sys.argv[1] + "_" + sys.argv[2] + "_" + sys.argv[3] + "_" + sys.argv[4] + "_" + sys.argv[5] + "_" + sys.argv[6] + ".log",
                    format='%(asctime)s %(message)s',
                    filemode='a')

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.INFO)


logging.info('Parameters (CC_SF_SERVER_LOG) ---------- [TOTAL_CLIENTS --> {}, STARTING_SERVER_PORT --> {}, DEVICE_TYPE --> {}, CUT_LAYER --> {}, EPOCHS --> {}, ROUNDS --> {}] ---------- '.format(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]))


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



def worker_routine(url, context, thread_no, r, cut_layer):
    """ Worker routine """

    global server_global_weights
    global server_weights
    global datasetsize_server

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

            self.model = models.resnet18(pretrained=True)
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

    if (sys.argv[4] == 'list'):
        config = {"cut_layer": cut_layer, "logits": 10}

    else:
        config = {"cut_layer": int(sys.argv[4]), "logits": 10}


    # client_model = ResNet18Client(config).to(device)
    server_model = ResNet18Server(config).to(device)

    criterion = nn.CrossEntropyLoss()
    # client_optimizer = optim.SGD(client_model.parameters(), lr=0.01, momentum=0.9)
    server_optimizer = optim.SGD(
        server_model.parameters(), lr=0.01, momentum=0.9)

    if r > 0:
        server_model.load_state_dict(server_global_weights)
        print("GLOBAL_SERVER_WEIGHTS_LOADED")

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

    num_epochs = int(sys.argv[5])

    round_start_time = time.time()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        # for i, data in enumerate(trainloader, 0):
        for j in range(recv_iterations):
            step_start_time = time.time()
            print("***TH - {}***  {} {}".format(thread_no, epoch, j))

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

            step_end_time = time.time()
            total_one_step_time = step_end_time - step_start_time
            print("***TH - {}***  SERVER_TOTAL_ONE_STEP_TIME = {:.3f}".format(thread_no, total_one_step_time))
            logging.info('***TH - {}***  SERVER_TOTAL_ONE_STEP_TIME = {:.3f}'.format(thread_no, total_one_step_time))


            ################################################################################

        epoch_end_time = time.time()
        total_one_epoch_time = epoch_end_time - epoch_start_time
        print("***TH - {}***  SERVER_TOTAL_ONE_EPOCH_TIME = {:.3f}" .format(thread_no, total_one_epoch_time))
        logging.info('***TH - {}***  SERVER_TOTAL_ONE_EPOCH_TIME = {:.3f}'.format(thread_no, total_one_epoch_time))

        ##################################################################################################################

    round_end_time = time.time()
    round_time = round_end_time - round_start_time
    print("***TH - {}***  SERVER_ROUND_TRAINING_TIME = {:.3f}" .format(thread_no, round_time))
    logging.info('***TH - {}***  SERVER_ROUND_TRAINING_TIME = {:.3f}'.format(thread_no, round_time))

    server_weights.append(server_model.state_dict())
    datasetsize_server.append(dataset_size)

    model_save_name = "./server_thread_model_r" + str(r) + "_" + str(thread_no) + "_" +sys.argv[1] + "_" + sys.argv[2] + "_" + sys.argv[3] + "_" + sys.argv[4] + "_" + sys.argv[5] + ".pt"
    torch.save(server_model.state_dict(), model_save_name)
    print("***TH - {}***  MODEL_SAVED." .format(thread_no))



    ##*****************************************************************************************************************
    ##*****************************************************************************************************************
    ##*****************************************************************************************************************

    print("Worker done******************************")

    socket.close()


def main():
    """ server routine """

    global server_global_weights
    global server_weights
    global datasetsize_server

    server_weights = []
    datasetsize_server = []

    total_threads = int(sys.argv[1])
    port_no = int(sys.argv[2])
    connection_url = ["tcp://*:" +str(port_no+i) for i in range(total_threads)]
    # connection_url = ["tcp://*:5555", "tcp://*:5556"]

    server_cut_layer_list = []
    if(sys.argv[4] == 'list'):
        for cut in range(total_threads):
            server_cut_layer_list.append(int(sys.argv[7+cut]))
        print(server_cut_layer_list)

    num_rounds = int(sys.argv[6])
    context = zmq.Context()

    training_start_time = time.time()
    for r in range(num_rounds):
        print("New round started..")
        thrs = []

        server_weights.clear()
        datasetsize_server.clear()

        # Launch pool of worker threads
        for i in range(total_threads):  # this defines how many clients can connect
            thread = threading.Thread(target=worker_routine, args=(connection_url[i], context, i, r, server_cut_layer_list[i]))
            thrs.append(thread)
            thread.start()

        for thread in thrs:         ##have to check when it will run all epochs..
            thread.join()

        print("Length of server weights:- ", len(server_weights))
        print("Length of dataset:- ", len(datasetsize_server))


        # Server models weighted averaging..
        # server_global_weights = average_weights(server_weights, datasetsize_server)
        server_global_weights = server_custom_avg_model(server_weights, datasetsize_server, server_cut_layer_list)

        model_save_name = "./cc_server_fedAvg_model_r" + str(r) + "_" + sys.argv[1] + "_" + sys.argv[2] + "_" + sys.argv[3] + sys.argv[4] + "_" + sys.argv[5] + "_" + sys.argv[6] + ".pt"
        torch.save(server_global_weights, model_save_name)
        print("MODEL_SAVED.")


        print("All threads ended..")
    print("All rounds ended..")

    training_end_time = time.time()
    training_time = training_end_time - training_start_time
    print("SERVER_TOTAL_TRAINING_TIME = {:.3f}" .format(training_time))
    logging.info('SERVER_TOTAL_TRAINING_TIME = {:.3f}'.format(training_time))

    context.term()

if __name__ == "__main__":
    main()
