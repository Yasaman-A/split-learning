#######################################################
#################     FED_SERVER    ###################
#######################################################

"""
arg1 --> CONFIG_FILE_PATH
"""

# eg command: python fedServer.py 2 4444

import copy
import threading
import torch.nn as nn
from torchvision import models
import time
import zmq
import torch
from ..lib import convert
from ..lib.transformed_dataset import TransformedDataset
from ..architectures import get_architecture_bundle
from sys import getsizeof
import yaml
import logging
import os
import urllib.request
import pickle
from torchvision import transforms, models
import torchvision.transforms as transforms
import torch.nn as nn
from tqdm.auto import tqdm


class Runner:
    def __init__(self, config_path) -> None:
        with open(config_path, "r") as yamlfile:
            self.config = yaml.load(yamlfile, Loader=yaml.FullLoader)
            print("Read successful")
    
    def run(self):
        client_total = self.config['client_total']
        fed_port = self.config['fed_server']['server_start_port']
        rnd = self.config['round']


        ##################################################################
        #Code to enable ad-hoc testing 
        if(self.config['device'] == 'cpu'):
            device = 'cpu'
        else:
            device = torch.device(
                'cuda') if torch.cuda.is_available() else torch.device('cpu')


        output_file = self.config['data_server']['output_file']
        val_file = output_file.replace(".pkl", "_val.pkl")
        val_file_tmp = f"tmp_fed_{val_file}"

        urllib.request.urlretrieve(
            f"{self.config['data_server']['server_address']}/{val_file}",
            val_file_tmp
            )
        
        with open(val_file_tmp, 'rb') as handle:
            valset = pickle.load(handle)

        output_file = self.config['data_server']['output_file']
        test_file = output_file.replace(".pkl", "_test.pkl")
        test_file_tmp = f"tmp_fed_{test_file}"
        cut_layer = self.config['test_cut_layer']

        urllib.request.urlretrieve(
            f"{self.config['data_server']['server_address']}/{test_file}",
            test_file_tmp
            )

        with open(test_file_tmp, 'rb') as handle:
            testset = pickle.load(handle)

        model_architecture = self.config.get("model_architecture", "ResNet18_CIFAR10")
        arch = get_architecture_bundle(model_architecture)
        logits = self.config.get("logits", 10)

        transformer = arch.eval_transformer
        
        valset = TransformedDataset(valset, transform=transformer)
        testset = TransformedDataset(testset, transform=transformer)
        
        valloader = torch.utils.data.DataLoader(valset,
                                    batch_size=self.config['batch_size'],
                                    shuffle=False,
                                    num_workers=0,
                                    persistent_workers=False
        )

        testloader = torch.utils.data.DataLoader(testset,
                                    batch_size=self.config['batch_size'],
                                    shuffle=False,
                                    num_workers=0,
                                    persistent_workers=False
        )

        ##################################################################

        if (self.config['logging']):
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



        def average_weights(state_dicts, datasizes, cut_layer_list):
            """
            Returns the average of the weights.
            """

            info = arch.client({"cut_layer" : self.config['cut_layer'], "logits": 10})
            
            weights_avg = copy.deepcopy(state_dicts[0])
            
            for i, data in enumerate(datasizes):
                for key in state_dicts[i].keys():
                    state_dicts[i][key] *= data

            for layer_idx, (layer_name, layer) in enumerate(info.model.named_children()):
                for key in info.state_dict().keys():
                    if key.startswith(f"model.{layer_name}"):
                        weight_value = 0
                        weight_size_sum = 0

                        for state_idx, state in enumerate(state_dicts):
                            if (layer_idx <= cut_layer_list[state_idx]):
                                weight_value += state[key]
                                weight_size_sum += datasizes[state_idx]
                        
                        if weight_size_sum > 0:
                            weights_avg[key] = weight_value / weight_size_sum

            return weights_avg


        def client_worker(sync_params, url, context, thread_no):
            """ Worker routine """

            global client_global_weights
            global client_weights
            global cut_layer_list
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

            msg = "dataset_size_recv"
            send_msg = msg.encode()
            socket.send(send_msg)

            recv_cut_layer = int(socket.recv().decode())
            with lock: cut_layer_list.append(recv_cut_layer)

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
            global cut_layer_list

            

            client_weights = []
            datasetsize_client = []
            cut_layer_list = []

            total_threads = client_total
            port_no = fed_port
            connection_url = ["tcp://*:" + str(port_no+i) for i in range(client_total)]

            num_rounds = rnd
            context = zmq.Context()
            terminate = False

            for r in range(num_rounds):
                if terminate:
                    context.term()
                    print("Terminate recieved.")
                    logging.info("Terminate recieved.")
                    break
                print("New round started..")
                thrs = []
                
                client_weights.clear()
                datasetsize_client.clear()
                cut_layer_list.clear()

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
                client_global_weights = average_weights(client_weights, datasetsize_client, cut_layer_list)
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

                if self.config['device'] != 'cpu': time.sleep(0.5)

                #get accuracy of aggregated models
                serv_context = zmq.Context()
                serv_url = f"tcp://*:{fed_port+client_total}"
                serv_socket = serv_context.socket(zmq.REQ)
                serv_socket.bind(serv_url)
                print(f"listening on {serv_url}")

                
                '''
                VAL SET - FOR EARLY STOPPING
                '''

                val_iters = len(valloader)
                send_val_iters = str(val_iters).encode()
                serv_socket.send(send_val_iters)
                serv_socket.recv()
                

                config = {"cut_layer": int(self.config['test_cut_layer']), "logits": 10}
                test_model = arch.client(config).to(device)
                test_model.load_state_dict(client_global_weights)

                bar = tqdm(valloader, desc=f"valset: ", unit='', ascii=True,
                           bar_format='{desc} {n_fmt}/{total_fmt} {percentage:3.0f}%|{bar}| {postfix}')

                test_model.eval()

                with torch.no_grad():
                    for data in bar:
                        inputs, labels = data[0].to(device), data[1].to(device)

                        #send labels
                        bytes_labels = convert.array_to_bytes(labels.cpu())
                        serv_socket.send(bytes_labels)
                        serv_socket.recv()

                        #send activations
                        activations = test_model(inputs)
                        server_inputs = activations.detach().clone()
                        bytes_server_inputs = convert.array_to_bytes(server_inputs.cpu())

                        serv_socket.send(bytes_server_inputs)
                        serv_socket.recv()

                serv_socket.send(b"term?")
                terminate = bool(int(serv_socket.recv().decode()))

                                
                '''
                TEST SET - TRUE ACCURACY
                '''

                #send dataset length
                test_iters = len(testloader)
                send_test_iters = str(test_iters).encode()
                serv_socket.send(send_test_iters)
                serv_socket.recv()

                config = {"cut_layer": int(self.config['test_cut_layer']), "logits": 10}
                test_model = arch.client(config).to(device)
                test_model.load_state_dict(client_global_weights)

                bar = tqdm(testloader, desc=f"testset: ", unit='', ascii=True,
                           bar_format='{desc} {n_fmt}/{total_fmt} {percentage:3.0f}%|{bar}| {postfix}')


                with torch.no_grad():
                    for data in bar:
                        inputs, labels = data[0].to(device), data[1].to(device)

                        #send labels
                        bytes_labels = convert.array_to_bytes(labels.cpu())
                        serv_socket.send(bytes_labels)
                        serv_socket.recv()

                        #send activations
                        activations = test_model(inputs)
                        server_inputs = activations.detach().clone()
                        bytes_server_inputs = convert.array_to_bytes(server_inputs.cpu())

                        serv_socket.send(bytes_server_inputs)
                        serv_socket.recv()


                test_model.train()

                serv_socket.close()
                serv_context.term()


            print("All rounds ended..")
            logging.info("All rounds ended..")
            context.term()

        main()
