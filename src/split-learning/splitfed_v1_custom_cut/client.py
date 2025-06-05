"""
arg1 --> CONFIG_FILE_PATH
arg2 --> CLIENT_ID
arg3 --> CUT_LAYER
"""

from locale import atoi
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models
import torch.optim as optim
import time
import pickle
import zmq
import torch
from ..lib import convert
from sys import getsizeof
import numpy as np
import urllib.request
import os
import yaml
import logging
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
        self.client_id = -1
        self.input_cut_layer = 0
        with open(config_path, "r") as yamlfile:
            self.config = yaml.load(yamlfile, Loader=yaml.FullLoader)
            print("Read successful")
        
        # Variables to track communication overhead
        self.total_activation_size = 0.0
        self.total_loss_size = 0.0

    def set_extra_options(self, extra):
        self.input_cut_layer = atoi(extra)
    
    def run(self):
        cut_layer = self.input_cut_layer
        split_address = self.config["split_server"]["server_ip"]
        split_port = self.config["split_server"]["server_start_port"]+self.client_id-1
        fed_port = self.config["fed_server"]["server_start_port"]+self.client_id-1
        log_steps = self.config["log_steps"]
        num_epochs = int(self.config["epoch"])
        output_file = self.config["data_server"]["output_file"]
        rnd = self.config["round"]


        initial_loading_start_time = time.time()


         #Initialize Logger
        if (self.config["logging"]):
            logging.basicConfig(
                filename=(
                    f"{self.client_id}_{cut_layer}_"
                    f"{self.config['epoch']}_{self.config['round']}_"
                    f"{self.config['batch_size']}_{self.config['device']}.log"
                ),
                format='%(asctime)s %(message)s',
                filemode='a'
                )
            logger = logging.getLogger()
            # Setting the threshold of logger to DEBUG
            logger.setLevel(logging.INFO)

        if(self.config["device"] == 'cpu'):
            device = 'cpu'
        else:
            device = torch.device(
                'cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(device)

         #Data Preparation

        #transforms for CIFAR-10
        transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
        ])
        batch_size = self.config["batch_size"]

        #Data Splitting
        match self.config["split_type"]:
            case 'n': #No splitting. Use full dataset
                trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                    download=True, transform=transformer)
                sampler = None
                shuffle = True

            case 's': #Use pre-defined split data
                if os.path.exists(output_file+str(self.client_id)):
                    os.remove(output_file+str(self.client_id))

                print(self.config["data_server"]["server_address"]+"/"+output_file)
                urllib.request.urlretrieve(self.config["data_server"]["server_address"]+"/"+output_file, output_file+str(self.client_id))

                with open(output_file+str(self.client_id), 'rb') as handle:
                    datasets = pickle.load(handle)
                    dataset = datasets[self.client_id-1]

                trainset = TransformedDataset(dataset, transformer)
                sampler = None
                shuffle = True

            case '_': #Split into 'split_type' number of blocks.
                trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                    download=True, transform=transformer)
                dataset_size = len(trainset)
                total_indices = list(range(dataset_size))
                list_of_indices = np.array_split(total_indices, int(self.config["split_type"]))
                use_indices = list_of_indices[self.client_id]
                datasetsize_used = len(use_indices)
                print('use_indices:' + str(use_indices))

                sampler = torch.utils.data.SubsetRandomSampler(use_indices)
                shuffle=False


        trainloader = torch.utils.data.DataLoader(trainset, 
                                        batch_size=batch_size,
                                        shuffle=shuffle,
                                        sampler = sampler,
                                        num_workers=2,
                                        persistent_workers=True)
        datasetsize_used = len(trainloader.dataset)



        class ResNet18Client(nn.Module):
            """docstring for ResNet"""

            def __init__(self, config):
                super(ResNet18Client, self).__init__()
                self.logits = config["logits"]
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
        client_model = ResNet18Client(config).to(device)

        client_optimizer = optim.SGD(
            client_model.parameters(), lr=0.01, momentum=0.9)

        training_start_time = time.time()
        num_rounds = rnd

        #Networking Telemetry
        total_sent_to_split = 0
        total_received_from_split = 0
        total_sent_to_fed = 0
        total_received_from_fed = 0
        weights_total_waiting_time = 0


        initial_loading_end_time = time.time()
        initial_loading_total_time = initial_loading_end_time - initial_loading_start_time
        print("CLIENT_INITIAL_LOADING_TIME = ", initial_loading_total_time)
        logging.info('CLIENT_INITIAL_LOADING_TIME = {:.3f}'.format(initial_loading_total_time))


        '''
        ====================================================        
        BEGIN TRAINING
        ====================================================
        '''

        for r in range(num_rounds):
            if r > 0:
                client_model.load_state_dict(global_numpy_weights)
                print("GLOBAL_CLIENT_WEIGHTS_LOADED")
                del global_numpy_weights
            
            round_sent_to_split = 0
            round_received_from_split = 0
            round_sent_to_fed = 0
            round_received_from_fed = 0

            logging.info(f"\n********ROUND {r}********\n")
            round_start_time = time.time()

            
            #Connect to Split Server
            context = zmq.Context()
            print("Connecting to server…")
            socket = context.socket(zmq.REQ)
            url = split_address + ":"+ str(split_port)
            socket.connect(url)
            # socket.connect("tcp://35.237.244.119:5555")


            iterations = len(trainloader)
            print(iterations)
            send_iterations = str(iterations).encode()
            socket.send(send_iterations)
            round_sent_to_split += len(send_iterations)

            names = socket.recv()
            round_received_from_split += len(names)

            print(datasetsize_used)
            send_dataset_size = str(datasetsize_used).encode()
            socket.send(send_dataset_size)
            round_sent_to_split += len(send_dataset_size)

            names = socket.recv()
            round_received_from_split += len(names)


            for epoch in range(num_epochs):
                logging.info(f"\n********EPOCH {epoch}********\n")
                
                epoch_start_time = time.time()
                epoch_sent_to_split = 0
                epoch_received_from_split = 0

                bar = tqdm(trainloader, desc=f"{r} {epoch}", unit='', ascii=True,
                           bar_format='{desc} {n_fmt}/{total_fmt} {percentage:3.0f}%|{bar}| {postfix}')

                for data in bar:
                    step_start_time = time.time()
                    inputs, labels = data[0].to(device), data[1].to(device)

                    bytes_labels = convert.array_to_bytes(labels.cpu())
                    socket.send(bytes_labels)
                    epoch_sent_to_split += len(bytes_labels)


                    ##dummy......
                    names = socket.recv()
                    epoch_received_from_split += len(names)


                    #forward prop and sending activations to server
                    activations = client_model(inputs)
                    server_inputs = activations.detach().clone()
                    bytes_server_inputs = convert.array_to_bytes(server_inputs.cpu())
                    
                    server_work_time_start = time.time()
                    socket.send(bytes_server_inputs)
                    epoch_sent_to_split += len(bytes_server_inputs)

                    #recover gradient from server
                    recv_grad = socket.recv()
                    epoch_received_from_split += len(recv_grad)
                    server_work_time_end = time.time()

                    numpy_grad = convert.bytes_to_array(recv_grad)
                    grad_output = torch.from_numpy(numpy_grad)
                    grad_output = grad_output.to(device)

                    client_optimizer.zero_grad()
                    activations.backward(gradient=grad_output)
                    client_optimizer.step()

                    step_end_time = time.time()
                    total_one_step_time = step_end_time - step_start_time
                    server_work_time = server_work_time_end - server_work_time_start


                    #telemetry
                    step_end_time = time.time()
                    total_one_step_time = step_end_time - step_start_time
                    server_work_time = server_work_time_end - server_work_time_start

                    bar.set_postfix({
                        "step_time": f"{total_one_step_time:.3f}",
                        "server_time": f"{server_work_time:.3f}"
                    })
                    logging.info(
                        f"CLIENT_TOTAL_ONE_STEP_TIME = {total_one_step_time:.3f}    , "
                        f"SERVER_WORK_TIME = {server_work_time:.3f}"
                    )


                    #BATCH OVER

                #Logging and telemetry
                epoch_end_time = time.time()
                total_one_epoch_time = epoch_end_time - epoch_start_time

                print("\nCLIENT_TOTAL_ONE_EPOCH_TIME = ", total_one_epoch_time)
                logging.info(f"\nCLIENT_TOTAL_ONE_EPOCH_TIME = {total_one_epoch_time:.3f}")

                print(f"Total data sent in epoch {epoch} (activations): {epoch_sent_to_split:.2f} bytes")
                logging.info(f"Total data sent in epoch {epoch} (activations): {epoch_sent_to_split:.2f} bytes")
                
                print(f"Total data received in epoch {epoch} (loss): {epoch_received_from_split:.2f} bytes")
                logging.info(f"Total data received in epoch {epoch} (loss): {epoch_received_from_split:.2f} bytes")
                
                total_data_transmitted_epoch = epoch_sent_to_split + epoch_received_from_split
                print(f"Total data transmitted in epoch {epoch}: {total_data_transmitted_epoch:.2f} bytes")
                logging.info(f"Total data transmitted in epoch {epoch}: {total_data_transmitted_epoch:.2f} bytes")

                                
                round_sent_to_split += epoch_sent_to_split
                round_received_from_split += epoch_received_from_split

                #EPOCH OVER


            socket.close()
            context.term()

            model_save_name = (
                f"./cc_client_thread_model_r{r}_{self.client_id}_{split_port}_"
                f"{self.config['device']}_{cut_layer}_{self.config['epoch']}_"
                f"{self.config['split_type']}_{self.client_id}_{self.config['batch_size']}_"
                f"{self.config['round']}_{fed_port}.pt"
            )
            torch.save(client_model.state_dict(), model_save_name)
            print("***TH - {}***  MODEL_SAVED." .format(self.client_id))


            '''
            ====================================================        
            SEND DATA TO FED SERVER
            ====================================================
            '''

            context1 = zmq.Context()

            #  Socket to talk to server
            print("Connecting to fed_avg server to give weights…")
            socket1 = context1.socket(zmq.REQ)
            url = str(self.config["fed_server"]["server_ip"]) + ":"+ str(fed_port)
            socket1.connect(url)

            weights = client_model.state_dict()

            print("Size of model weights (before) in bytes is:", getsizeof(weights))
            bytes_weights = convert.ordered_dict_to_bytes(weights)
            print("Size of model weights (after) in bytes is:", getsizeof(bytes_weights))

            logging.info('Size of model weights (before) in bytes is: %s', (getsizeof(weights)))
            logging.info('Size of model weights (after) in bytes is: %s', (getsizeof(bytes_weights)))
            
            send_weights_start_time = time.time()
            socket1.send(bytes_weights)
            round_sent_to_fed += len(bytes_weights)

            ## dummy recv
            names = socket1.recv()
            round_received_from_fed += len(names)
            send_weights_end_time = time.time()

            send_weights_time = send_weights_end_time - send_weights_start_time
            logging.info("SEND_WEIGHTS_COMMUNICATION_TIME = {:.3f}".format(send_weights_time))
            print("SEND_WEIGHTS_COMMUNICATION_TIME = {:.3f}".format(send_weights_time))

            ## send dataset size for weighted avg
            socket1.send(send_dataset_size)
            round_sent_to_fed += len(send_dataset_size)

            ## dummy recv
            names = socket1.recv()
            round_received_from_fed += len(names)

            ## send cut layer info
            send_cut_layer_size = str(cut_layer).encode()
            socket1.send(send_cut_layer_size)
            round_sent_to_fed += len(send_cut_layer_size)

            ## dummy recv
            names = socket1.recv()
            round_received_from_fed += len(names)

            del weights
            del bytes_weights

            socket1.close()
            context1.term()

            '''
            ====================================================        
            RECEIVE GLOBAL MODEL FROM FED SERVER
            ====================================================
            '''

            context2 = zmq.Context()
            print("Connecting to fed_avg server to recv global weights…")

            weights_waiting_start_time = time.time()
            socket2 = context2.socket(zmq.REQ)
            url = str(self.config["fed_server"]["server_ip"]) + ":"+ str(fed_port)
            socket2.connect(url)

            msg = "send_global_weights" #TODO: Minimize this message to reduce slight message size overhead
            send_msg = msg.encode()
            socket2.send(send_msg)
            round_sent_to_fed += len(send_msg)

            global_weights = socket2.recv()
            round_received_from_fed += len(global_weights)

            weights_waiting_end_time = time.time()
            weights_waiting_time = weights_waiting_end_time - weights_waiting_start_time
            weights_total_waiting_time += weights_waiting_time

            #Receive weights communication time
            print(f"CLIENT_WEIGHTS_WAITING_TIME = {weights_waiting_time}")
            logging.info(f"CLIENT_WEIGHTS_WAITING_TIME = {weights_waiting_time:.3f}")

            
            print("Global weights recieved from fedServer")
            print("Size of global model weights (before) in bytes is:", getsizeof(global_weights))
            
            global_numpy_weights = convert.bytes_to_dict(global_weights)
            print("Size of global model weights (after) in bytes is:", getsizeof(global_numpy_weights))

            round_sent_to_servers = round_sent_to_fed + round_sent_to_split
            round_rcvd_from_servers = round_received_from_fed + round_received_from_split
            round_total = round_sent_to_servers + round_rcvd_from_servers

            # At the end of each round, log total data sent and received
            print("======== Round Networking Summary ========")
            print(f"Data sent to Split Server: {round_sent_to_split} bytes")
            print(f"Data sent to Fed Server: {round_sent_to_fed} bytes")
            print(f"Total Sent: {round_sent_to_servers}")
            print(f"Data received from Split Server: {round_received_from_split} bytes")
            print(f"Data received from Fed Server: {round_received_from_fed} bytes")
            print(f"Total Received: {round_rcvd_from_servers}")
            print(f"===== \nTotal Transmitted this Round: {round_total}")
            
            logging.info("======== Round Networking Summary ========")
            logging.info(f"Data sent to Split Server: {round_sent_to_split} bytes")
            logging.info(f"Data sent to Fed Server: {round_sent_to_fed} bytes")
            logging.info(f"Total Sent: {round_sent_to_servers}")
            logging.info(f"Data received from Split Server: {round_received_from_split} bytes")
            logging.info(f"Data received from Fed Server: {round_received_from_fed} bytes")
            logging.info(f"Total Received: {round_rcvd_from_servers}")
            logging.info(f"===== \nTotal Transmitted this Round: {round_total}")
            
            total_sent_to_split += round_sent_to_split
            total_sent_to_fed += round_sent_to_fed
            total_received_from_split += round_received_from_split
            total_received_from_fed += round_received_from_fed

            socket2.close()
            context2.term()

            #END ROUND



        training_end_time = time.time()
        training_time = training_end_time - training_start_time
        print(f"CLIENT_TOTAL_TRAINING_TIME = {training_time}")
        logging.info(f"CLIENT_TOTAL_TRAINING_TIME = {training_time:.3f}")

        total_sent_to_servers = total_sent_to_split + total_sent_to_fed
        total_rcvd_from_servers = total_received_from_split + total_received_from_fed
        total_data_transmitted = total_sent_to_servers + total_rcvd_from_servers

        print("======== Training Networking Summary ========")
        print("Sent Data:")
        print(f"Sent to Split Server: {total_sent_to_split} bytes.")
        print(f"Sent to Fed Server: {total_sent_to_fed} bytes.")
        print(f"Combined Total: {total_sent_to_servers} bytes.")
        print("Received Data:")
        print(f"Received from Split Server: {total_received_from_split} bytes.")
        print(f"Received from Fed Server: {total_received_from_fed} bytes.")
        print(f"Combined Total: {total_rcvd_from_servers} bytes.")
        print("========")
        print(f"Total Transmitted Data: {total_data_transmitted} bytes.")

        logging.info("======== Training Networking Summary ========")
        logging.info("Sent Data:")
        logging.info(f"Sent to Split Server: {total_sent_to_split} bytes.")
        logging.info(f"Sent to Fed Server: {total_sent_to_fed} bytes.")
        logging.info(f"Combined Total: {total_sent_to_servers} bytes.")
        logging.info("Received Data:")
        logging.info(f"Received from Split Server: {total_received_from_split} bytes.")
        logging.info(f"Received from Fed Server: {total_received_from_fed} bytes.")
        logging.info(f"Combined Total: {total_rcvd_from_servers} bytes.")
        logging.info("========")
        logging.info(f"Total Transmitted Data: {total_data_transmitted} bytes.")