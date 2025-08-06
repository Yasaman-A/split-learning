
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
import yaml
from sys import getsizeof
from .custom_model_avg import custom_model_avg, combine_fed_avg_models
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
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                        (0.2023, 0.1994, 0.2010))
            ])
        
        testset = TransformedDataset(testset, transform=transformer)
        
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

        config = {"cut_layer": int(cut_layer), "logits": 10}
        model = ResNet18Client(config).to(device)

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

        def client_worker(sync_params, url, context, thread_no):
            global client_global_weights
            global client_weights
            global datasetsize_client
            global client_cut_layer_list

            lock, barrier, event = sync_params

            socket = context.socket(zmq.REP)
            socket.bind(url)

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

            msg = "dataset_size_recv"
            send_msg = msg.encode()
            socket.send(send_msg)

            recv_cut_layer = socket.recv()
            cut_layer = int(recv_cut_layer.decode())
            
            with lock: client_cut_layer_list.append(cut_layer)
            
            barrier.wait() #ensure all threads are done

            event.wait() #wait for server to process model
            
            print(f"Size of global model weights (before) in bytes is: {getsizeof(client_global_weights)}")
            logging.info('Size of global model weights (before) in bytes is: %s', getsizeof(client_global_weights))
            
            global_bytes_weights = convert.ordered_dict_to_bytes(client_global_weights)

            print(f"Size of global model weights (after) in bytes is: {getsizeof(global_bytes_weights)}")
            logging.info('Size of Size of global model weights (after) in bytes is: %s', getsizeof(global_bytes_weights))


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
            global client_cut_layer_list

            client_weights = []
            datasetsize_client = []
            client_cut_layer_list = []

            #client_exposure = []

            terminate = False

            total_threads = int(client_total)
            port_no = int(fed_port)
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
                client_cut_layer_list.clear()

                lock = threading.Lock()
                barrier = threading.Barrier(parties=total_threads + 1)
                event = threading.Event()
                sync_params = (lock, barrier, event)

                logging.info("Launching reciever threads..")
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

                print("Length of client weights:  ", len(client_weights))
                print("Length of dataset: ", len(datasetsize_client))
                print("Length of cut_layers: ", len(client_cut_layer_list))

                client_global_weights, client_exposure = custom_model_avg(False, client_weights, datasetsize_client, client_cut_layer_list)

                print("Global clients calculated..")
                logging.info("Global clients calculated..")

                model_save_name = os.path.join(
                    self.config.get("model_dir", "./"),
                    f"client_fedAvg_model_r{r}_{client_total}_{fed_port}_{rnd}.pt"
                )
                torch.save(client_global_weights, model_save_name)
                print("MODEL_SAVED.")
                logging.info("MODEL SAVED.")
                
                event.set() #workers will send data back to clients

                for no, thread in enumerate(thrs):  # have to check when it will run all epochs..
                    thread.join()
                    logging.info(f"Thread {no} joined.")
                
                logging.info("All threads joined.")
                
                if self.config['device'] != "cpu":
                    time.sleep(1) #gpu is too fast for ZMQ; race condition occurs and fed server terminates.

                print("All threads ended..")

                #get accuracy of aggregated models
                serv_context = zmq.Context()
                serv_url = f"tcp://*:{fed_port+client_total}"
                serv_socket = serv_context.socket(zmq.REQ)
                serv_socket.bind(serv_url)
                print(f"listening on {serv_url}")

                #send dataset length
                test_iters = len(testloader)
                send_test_iters = str(test_iters).encode()
                serv_socket.send(send_test_iters)
                serv_socket.recv()
                
                bar = tqdm(testloader, desc=f"testset: ", unit='', ascii=True,
                           bar_format='{desc} {n_fmt}/{total_fmt} {percentage:3.0f}%|{bar}| {postfix}')

                model.eval()

                with torch.no_grad():
                    for data in bar:
                        inputs, labels = data[0].to(device), data[1].to(device)

                        #send labels
                        bytes_labels = convert.array_to_bytes(labels.cpu())
                        serv_socket.send(bytes_labels)
                        serv_socket.recv()

                        #send activations
                        activations = model(inputs)
                        server_inputs = activations.detach().clone()
                        bytes_server_inputs = convert.array_to_bytes(server_inputs.cpu())

                        serv_socket.send(bytes_server_inputs)
                        serv_socket.recv()

                serv_socket.send(b"term?")
                terminate = bool(int(serv_socket.recv().decode()))

                serv_socket.close()
                serv_context.term()

            print("socket closed")



            print("All rounds ended..")
            logging.info("All rounds ended..")







            #exposure / final aggregation legacy code.

            # bytes_weights = serv_socket.recv()
            # serv_socket.send("a".encode())
            # serv_weights = convert.bytes_to_dict(bytes_weights)
            # print("got weights")

            # bytes_exposure = serv_socket.recv()
            # serv_socket.send("a".encode())
            # serv_exposure = convert.bytes_to_dict(bytes_exposure)
            # print("got exposure")

            # final_model_weights = combine_fed_avg_models(client_global_weights, client_exposure, serv_weights, serv_exposure)
            # model_save_name = f"./final_aggregate_model.pt"
            # model_save_name = os.path.join(self.config.get("model_dir", "./"), "final_aggregate_model.pt")
            # torch.save(final_model_weights, model_save_name)


            context.term()

        main()
