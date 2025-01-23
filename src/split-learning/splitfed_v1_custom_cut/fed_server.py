
#######################################################
#################     FED_SERVER    ###################
#######################################################
"""
arg1 --> CONFIG_FILE_PATH
"""

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
# from convert import array_to_bytes, bytes_to_array, ordered_dict_to_bytes, bytes_to_dict
from ..lib import convert
import sys
import yaml
from sys import getsizeof
from .custom_model_avg import client_custom_avg_model
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
            logging.basicConfig(filename="./cc_fed_server_" + str(client_total) + "_" + str(fed_port) + "_" + str(rnd) + ".log",
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
                for i in range(1, len(w)):        ## IMP
                    w_avg[key] += w[i][key]
                w_avg[key] = torch.div(w_avg[key], float(sum(datasize)))

            return w_avg



        def get_weights(url, context, thread_no):
            """ Worker routine """

            global client_global_weights
            global client_weights
            global datasetsize_client
            global client_cut_layer_list

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
            # print(dataset_size)

            msg = "dataset_size_recv"
            send_msg = msg.encode()
            socket.send(send_msg)

            recv_cut_layer = socket.recv()
            cut_layer = int(recv_cut_layer.decode())
            client_cut_layer_list.append(cut_layer)
            # print(cut_layer)

            msg = "cut_layer_recv"
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

            print("Size of global model weights (before) in bytes is:", getsizeof(client_global_weights))
            global_bytes_weights = convert.ordered_dict_to_bytes(client_global_weights)
            print("Size of global model weights (after) in bytes is:",
                  getsizeof(global_bytes_weights))
            # time.sleep(10)
            logging.info('Size of global model weights (before) in bytes is:', getsizeof(client_global_weights))
            logging.info('Size of Size of global model weights (after) in bytes is:', getsizeof(global_bytes_weights))


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
            global client_cut_layer_list

            client_weights = []
            datasetsize_client = []
            client_cut_layer_list = []

            total_threads = int(client_total)
            port_no = int(fed_port)
            connection_url = ["tcp://*:" + str(fed_port+i) for i in range(client_total)]
            # connection_url = ["tcp://*:5555", "tcp://*:5556"]

            num_rounds = rnd
            context = zmq.Context()

            for r in range(num_rounds):
                print("New round started..")
                thrs = []

                client_weights.clear()
                datasetsize_client.clear()
                client_cut_layer_list.clear()

                # Launch pool of worker threads
                for i in range(total_threads):  # this defines how many clients can connect
                    thread = threading.Thread(target=get_weights, args=(
                        connection_url[i], context, i+1))
                    thrs.append(thread)
                    thread.start()

                for thread in thrs:  # have to check when it will run all epochs..
                    thread.join()

                print("Length of client weights:  ", len(client_weights))
                print("Length of dataset: ", len(datasetsize_client))
                print("Length of cut_layers: ", len(client_cut_layer_list))


                # Client models weighted averaging..
                # client_global_weights = average_weights(client_weights, datasetsize_client)
                client_global_weights = client_custom_avg_model(client_weights, datasetsize_client, client_cut_layer_list)
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
