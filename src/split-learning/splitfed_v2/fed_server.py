#######################################################
#################     FED_SERVER    ###################
#######################################################

"""
arg1 --> CONFIG_FILE_PATH
"""

import copy
import threading
import time
import zmq
import torch
from ..lib import convert
from sys import getsizeof
import yaml
import logging
import torch.nn as nn
from torchvision import models
import os


class ResNet18Client(nn.Module):
    """docstring for ResNet"""

    def __init__(self, config):
        super(ResNet18Client, self).__init__()
        self.cut_layer = config["cut_layer"]
        self.logits = config["logits"]

        self.model = models.resnet18(weights=None)

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Flatten(),
                                        nn.Linear(num_ftrs, self.logits))


        self.layers = list(self.model.children())



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
            log_path = os.path.join(
                self.config.get("log_dir", "./"),
                f"./cc_fed_server_{client_total}_{fed_port}_{rnd}.log"
            )
            
            # Create and configure logger
            logging.basicConfig(filename=log_path,
                                format='%(asctime)s %(message)s',
                                filemode='a')
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            logging.info('Parameters (FED_SERVER_LOG) ---------- [TOTAL_CLIENTS --> {}, STARTING_SERVER_PORT --> {}, ROUNDS --> {}] ---------- '.format(
                str(client_total), str(fed_port), str(rnd)))



        def average_weights(state_dicts, datasizes):
            """
            Returns the average of the weights.
            """
            info = ResNet18Client({"cut_layer" : self.config["cut_layer"], "logits": 10})
            
            weights_avg = copy.deepcopy(state_dicts[0])

            for i, data in enumerate(datasizes):
                for key in state_dicts[i].keys():
                    state_dicts[i][key] *= data

            for layer_idx, (layer_name, layer) in enumerate(info.model.named_children()):
                for key in info.state_dict().keys():
                    if key.startswith(f"model.{layer_name}"):
                        weight_value = 0
                        weight_size_sum = 0

                        for state_idx, state_dict in enumerate(state_dicts):
                            if (layer_idx <= self.config["cut_layer"]):
                                weight_value += state_dict[key]
                                weight_size_sum += datasizes[state_idx]
                        
                        if weight_size_sum > 0:
                            weights_avg[key] = weight_value / weight_size_sum
        
            return weights_avg


        def client_worker(sync_params, url, context, thread_no):
            """ Worker routine """

            global client_global_weights
            global client_weights
            global datasetsize_client

            lock, barrier, event = sync_params

            socket = context.socket(zmq.REP)

            socket.bind(url)

            ##*****************************************************************************************************************

            print(f"Waiting for weights from client {thread_no}")
            logging.info(f"Waiting for weights from client {thread_no}")
            weights = socket.recv()
            print(f"Weights recieved from client {thread_no}")
            logging.info(f"Weights recieved from client {thread_no}")
            numpy_weights = convert.bytes_to_dict(weights)
            with lock: client_weights.append(numpy_weights)

            msg = "weights_recv"
            send_msg = msg.encode()
            socket.send(send_msg)

            recv_dataset_size = socket.recv()
            dataset_size = int(recv_dataset_size.decode())
            with lock: datasetsize_client.append(dataset_size)
            print(f"{thread_no} : dataset_size: {dataset_size}")
            logging.info(f"{thread_no} : dataset_size: {dataset_size}")

            barrier.wait()

            event.wait()

            print(f"Size of global model weights (before) in bytes is: {getsizeof(client_global_weights)}")
            logging.info(f"Size of global model weights (before) in bytes is: {getsizeof(client_global_weights)}")

            global_bytes_weights = convert.ordered_dict_to_bytes(client_global_weights)
            
            print(f"Size of global model weights (after) in bytes is: {getsizeof(global_bytes_weights)}")
            logging.info(f"Size of Size of global model weights (after) in bytes is: {getsizeof(global_bytes_weights)}")
            
            socket.send(global_bytes_weights)
            print(f"Weights sent to client {thread_no}")
            logging.info(f"Weights sent to client {thread_no}")

            ##*****************************************************************************************************************
            print(f"Worker {thread_no} done******************************")
            logging.info(f"Worker {thread_no} done******************************")

            socket.close()



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

            num_rounds = rnd
            context = zmq.Context()


            for r in range(num_rounds):
                print("New round started..")
                thrs = []

                client_weights.clear()
                datasetsize_client.clear()

                lock = threading.Lock()
                barrier = threading.Barrier(parties = total_threads + 1)
                event = threading.Event()
                sync_params = (lock, barrier, event)


                # Launch pool of worker threads
                for i in range(total_threads):  # this defines how many clients can connect
                    thread = threading.Thread(target=client_worker, 
                                              args=(
                                                sync_params,
                                                connection_url[i], 
                                                context, 
                                                i+1))
                    thrs.append(thread)
                    thread.start()

                barrier.wait()

                # Server models weighted averaging..
                client_global_weights = average_weights(client_weights, datasetsize_client)
                print("Global clients calculated..")
                logging.info("Global clients calculated..")

                model_save_name = os.path.join(
                    self.config.get("model_dir", "./"),
                    f"client_fedAvg_model_r{r}_{client_total}_{fed_port}_{rnd}.pt"
                )

                torch.save(client_global_weights, model_save_name)
                print("MODEL_SAVED.")
                logging.info("MODEL_SAVED.")

                event.set()

                for no, thread in enumerate(thrs):  # have to check when it will run all epochs..
                    thread.join()
                    logging.info(f"Thread {no} joined")

                logging.info("All threads joined.")

                print("All threads ended..")

                if self.config["device"] != 'cpu': time.sleep(0.5)
                
            print("All rounds ended..")
            logging.info("All rounds ended..")

            context.term()

        main()
