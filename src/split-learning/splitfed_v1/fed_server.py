#######################################################
#################     FED_SERVER    ###################
#######################################################
"""
arg1 --> CONFIG_FILE_PATH
"""

# eg command: python fedServer.py 2 4444

import copy
import threading
import time
import zmq
import torch
from ..lib import convert
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



class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, label




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



        transformer = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                        (0.2023, 0.1994, 0.2010))
            ])
        
        
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

        

        class ResNet18Client(nn.Module):

            def __init__(self, config):
                super(ResNet18Client, self).__init__()
                self.logits = config['logits']
                self.cut_layer = cut_layer

                self.model = models.resnet18(weights=None)

                num_ftrs = self.model.fc.in_features
                self.model.fc = nn.Sequential(nn.Flatten(),
                                                  nn.Linear(num_ftrs, self.logits))
                
                self.layers = list(self.model.children())

            def forward(self, x):
                for i, l in enumerate(self.layers):
                    if i > self.cut_layer:
                        break
                    x = l(x)
                return x


        if (self.config['logging']):
            log_path = os.path.join(
                self.config.get("log_dir", "./"),
                f"./fed_server_{client_total}_{fed_port}_{rnd}.log"
            )
            # Create and configure logger
            logging.basicConfig(
                filename=log_path,
                format='%(asctime)s %(message)s',
                filemode='a'
            )
            logger = logging.getLogger()
            # Setting the threshold of logger to DEBUG
            logger.setLevel(logging.INFO)
            logging.info(
                f"Parameters (FED_SERVER_LOG) ---------- [TOTAL_CLIENTS --> {client_total}, "
                f"STARTING_SERVER_PORT --> {fed_port}, ROUNDS --> {rnd}] ----------"
            )


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


        def socket_bind_retry(socket, url, max_retries=10, delay=5):
            retries = 0
            while retries < max_retries:
                try:
                    socket.bind(url)
                    print(f"Success bounding to {url}")
                    return True
                except zmq.ZMQError as e:
                    if e.errno == zmq.EADDRINUSE:
                        retries += 1
                        print(f"Address {url} in use. Retrying... ({retries}/{max_retries})")
                        logging.info(f"Address {url} in use. Retrying... ({retries}/{max_retries})")
                        time.sleep(delay)
                    else:
                        raise e
            return False


        def client_worker(sync_params, url, context, thread_no):
            """ Worker routine """

            global client_global_weights
            global client_weights
            global datasetsize_client

            lock, barrier, event = sync_params

            socket = context.socket(zmq.REP)
            
            if not socket_bind_retry(socket, url):
                print(f"Failed binding to {url}.")
                return

            ##*****************************************************************************************************************

            print("Waiting for weights from client {}".format(thread_no))
            weights = socket.recv()
            print("Weights recieved from client {}".format(thread_no))
            numpy_weights = convert.bytes_to_dict(weights)
            with lock: client_weights.append(numpy_weights)

            msg = "weights_recv"
            send_msg = msg.encode()
            socket.send(send_msg)

            recv_dataset_size = socket.recv()
            dataset_size = int(recv_dataset_size.decode())
            with lock: datasetsize_client.append(dataset_size)
            print(dataset_size)


            barrier.wait() #ensure all threads are done

            event.wait() #wait for server to process model

            print("Size of global model weights (before) in bytes is:", getsizeof(client_global_weights))
            logging.info(f"Size of global model weights (before) in bytes is: {getsizeof(client_global_weights)}")
            global_bytes_weights = convert.ordered_dict_to_bytes(client_global_weights)
            print("Size of global model weights (after) in bytes is:",
                  getsizeof(global_bytes_weights))
            logging.info(f"Size of global model weights (after) in bytes is: {getsizeof(global_bytes_weights)}")

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

            terminate = False

            total_threads = client_total
            port_no = fed_port
            connection_url = ["tcp://*:" + str(fed_port+i) for i in range(client_total)]

            num_rounds = rnd
            context = zmq.Context()

            for r in range(num_rounds):
                if terminate:
                    context.term()
                    print("Terminate recieved.")
                    logging.info("Terminate recieved.")
                    break
                print("New round started..")
                logging.info("New round started..")
                thrs = []

                client_weights.clear()
                datasetsize_client.clear()

                lock = threading.Lock()
                barrier = threading.Barrier(parties = total_threads + 1)
                event = threading.Event()
                sync_params = (lock, barrier, event)

                logging.info("Launching reciever threads...")
                # Launch pool of worker threads
                for i in range(total_threads):  # this defines how many clients can connect
                    thread = threading.Thread(target=client_worker, 
                                              args=(
                                                sync_params,
                                                connection_url[i], 
                                                context, 
                                                i+1)
                        )
                    thrs.append(thread)
                    thread.start()

                barrier.wait()

                print("Length of client weights:", len(client_weights))
                logging.info(f"Length of client weights: {len(client_weights)}")
                print("Length of dataset:", len(datasetsize_client))
                logging.info(f"Length of dataset: {len(datasetsize_client)}")

                # Client models weighted averaging..
                client_global_weights = average_weights(client_weights, datasetsize_client)
                print("Global clients calculated..")
                logging.info("Global clients calculated..")

                model_save_name = os.path.join(
                    self.config.get("model_dir", "./"),
                    f"./client_fedAvg_model_r_{r}_{client_total}_{fed_port}_{rnd}.pt"
                )
                torch.save(client_global_weights, model_save_name)
                print("MODEL_SAVED.")

                event.set()

                for no, thread in enumerate(thrs):
                    thread.join()
                    logging.info(f"Thread {no} joined.")
                
                logging.info("All threads joined.")
                
                if self.config['device'] != "cpu":
                    time.sleep(1) #gpu is too fast for ZMQ; race condition occurs and fed server terminates.

                print("All threads ended..")

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
                

                config = {"cut_layer": int(self.config['cut_layer']), "logits": 10}
                test_model = ResNet18Client(config).to(device)
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

                config = {"cut_layer": int(self.config['cut_layer']), "logits": 10}
                test_model = ResNet18Client(config).to(device)
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

            print("socket closed")

            print("All rounds ended..")
            logging.info("All rounds ended..")

            context.term()

        main()