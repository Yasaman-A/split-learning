#######################################################
#################     FED_SERVER    ###################
#######################################################

"""
arg1 --> CONFIG_FILE_PATH
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
from ..lib import convert
import sys
from sys import getsizeof
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
        fed_port = self.config["fed_server"]["server_start_port"]
        rnd = self.config["round"]


        if (self.config["logging"]):
            # Create and configure logger
            logging.basicConfig(filename="./fed_server_" + str(client_total) + "_" + str(fed_port) + "_" + str(rnd) + ".log",
                                format='%(asctime)s %(message)s',
                                filemode='a')
            # Creating an object
            logger = logging.getLogger()
            # Setting the threshold of logger to DEBUG
            logger.setLevel(logging.INFO)
            logging.info('Parameters (FED_SERVER_LOG) ---------- [TOTAL_CLIENTS --> {}, STARTING_SERVER_PORT --> {}, ROUNDS --> {}] ---------- '.format(
                str(client_total), str(fed_port), str(rnd)))



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
            numpy_weights = convert.bytes_to_dict(weights)
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
            time.sleep(10)


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

            print("Size of global model weights (before) in bytes is:", getsizeof(client_global_weights))
            global_bytes_weights = convert.ordered_dict_to_bytes(client_global_weights)
            print("Size of global model weights (after) in bytes is:",
                  getsizeof(global_bytes_weights))
            # time.sleep(10)
            socket.send(global_bytes_weights)
            print("Weights send to client {}".format(thread_no))

            ##*****************************************************************************************************************
            print("Worker done******************************")

            socket.close()
            time.sleep(10)



        def main():
            """ server routine """

            global client_global_weights
            global client_weights
            global datasetsize_client

            client_weights = []
            datasetsize_client = []

            total_threads = client_total
            port_no = fed_port
            connection_url = ["tcp://*:" + str(port_no+i) for i in range(client_total)]
            # connection_url = ["tcp://*:5555", "tcp://*:5556"]

            num_rounds = rnd
            context = zmq.Context()

            for r in range(num_rounds):
                print("New round started..")
                thrs = []
                # Launch pool of worker threads
                for i in range(total_threads):  # this defines how many clients can connect
                    thread = threading.Thread(target=get_weights, args=(
                        connection_url[i], context, i+1))
                    thrs.append(thread)
                    thread.start()

                for thread in thrs:  # have to check when it will run all epochs..
                    thread.join()



                # Server models weighted averaging..
                client_global_weights = average_weights(client_weights, datasetsize_client)
                print("Global clients calculated..")

                model_save_name = "./client_fedAvg_model_r" + str(r) + "_" + str(client_total) + "_" + str(fed_port) + "_" + str(rnd) + ".pt"
                torch.save(client_global_weights, model_save_name)
                print("MODEL_SAVED.")


                thrs = []
                # Launch pool of worker threads
                for i in range(total_threads):  # this defines how many clients can connect
                    thread = threading.Thread(target=send_weights, args=(
                        connection_url[i], context, i+1))
                    thrs.append(thread)
                    thread.start()

                for thread in thrs:  # have to check when it will run all epochs..
                    thread.join()



                print("All threads ended..")
            print("All rounds ended..")

            context.term()

        main()
