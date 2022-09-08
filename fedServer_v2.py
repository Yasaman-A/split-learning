
#######################################################
#################     FED_SERVER    ###################
#######################################################

"""
arg1 --> total number of clients
arg2 --> STARTING_PORT_NO
arg3 --> round
"""

# eg command: python fedServer.py 2 4444

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
from convert import array_to_bytes, bytes_to_array, ordered_dict_to_bytes, bytes_to_dict
import sys
from sys import getsizeof

import logging
# from objsize import get_deep_size

# Create and configure logger
logging.basicConfig(filename="./fed_server_" + sys.argv[1] + "_" + sys.argv[2] + "_" + sys.argv[3] + ".log",
                    format='%(asctime)s %(message)s',
                    filemode='a')

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.INFO)


logging.info('Parameters (FED_SERVER_LOG) ---------- [TOTAL_CLIENTS --> {}, STARTING_SERVER_PORT --> {}, ROUNDS --> {}] ---------- '.format(
    sys.argv[1], sys.argv[2], sys.argv[3]))


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


def get_weights(url, context, thread_no):
    """ Worker routine """

    global client_global_weights
    global client_weights
    global datasetsize_client

    # Socket to talk to dispatcher
    # context = zmq.Context()
    socket = context.socket(zmq.REP)

    # socket.connect(worker_url)
    # socket.connect("tcp://*:5555")
    socket.bind(url)

    ##*****************************************************************************************************************

    print("Waiting for weights from client {}".format(thread_no))
    weights = socket.recv()
    print("Weights recieved from client {}".format(thread_no))
    numpy_weights = bytes_to_dict(weights)
    client_weights.append(numpy_weights)

    msg = "weights_recv"
    send_msg = msg.encode()
    socket.send(send_msg)

    recv_dataset_size = socket.recv()
    dataset_size = int(recv_dataset_size.decode())
    datasetsize_client.append(dataset_size)
    print(dataset_size)

    msg = "dataset_size_recv"
    send_msg = msg.encode()
    socket.send(send_msg)

    ##*****************************************************************************************************************

    print("Worker done******************************")

    socket.close()


def send_weights(url, context, thread_no):
    """ Worker routine """

    global client_global_weights

    # Socket to talk to dispatcher
    # context = zmq.Context()
    socket = context.socket(zmq.REP)

    # socket.connect(worker_url)
    # socket.connect("tcp://*:5555")
    socket.bind(url)

    ##*****************************************************************************************************************
    
    ## dummy recv
    names = socket.recv()
    recv_names = names.decode()

    print("SIze of global model weights (before) in bytes is:-", getsizeof(client_global_weights))
    global_bytes_weights = ordered_dict_to_bytes(client_global_weights)
    print("SIze of global model weights (after) in bytes is:-",
          getsizeof(global_bytes_weights))
    # time.sleep(10)
    socket.send(global_bytes_weights)
    print("Weights send to client {}".format(thread_no))

    ##*****************************************************************************************************************
    print("Worker done******************************")

    socket.close()



def main():
    """ server routine """

    global client_global_weights
    global client_weights
    global datasetsize_client

    client_weights = []
    datasetsize_client = []

    total_threads = int(sys.argv[1])
    port_no = int(sys.argv[2])
    connection_url = ["tcp://*:" + str(port_no+i) for i in range(total_threads)]
    # connection_url = ["tcp://*:5555", "tcp://*:5556"]

    num_rounds = int(sys.argv[3])
    context = zmq.Context()

    for r in range(num_rounds):
        print("New round started..")
        thrs = []
        # Launch pool of worker threads
        for i in range(total_threads):  # this defines how many clients can connect
            thread = threading.Thread(target=get_weights, args=(
                connection_url[i], context, i))
            thrs.append(thread)
            thread.start()

        for thread in thrs:  # have to check when it will run all epochs..
            thread.join()



        # Server models weighted averaging..
        client_global_weights = average_weights(client_weights, datasetsize_client)
        print("Global clients calculated..")
        
        model_save_name = "./client_fedAvg_model_r" + str(r+1) + "_" + sys.argv[1] + "_" + sys.argv[2] + "_" + sys.argv[3] + ".pt"
        torch.save(client_global_weights, model_save_name)
        print("MODEL_SAVED.")


        thrs = []
        # Launch pool of worker threads
        for i in range(total_threads):  # this defines how many clients can connect
            thread = threading.Thread(target=send_weights, args=(
                connection_url[i], context, i))
            thrs.append(thread)
            thread.start()

        for thread in thrs:  # have to check when it will run all epochs..
            thread.join()



        print("All threads ended..")
    print("All rounds ended..")

    context.term()


if __name__ == "__main__":
    main()
