"""
arg1 --> CONFIG_FILE_PATH
arg2 --> CLIENT_ID
"""

import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models
import torch.optim as optim
import time
import zmq
import torch
from ..lib import convert
import os
import urllib.request
import pickle
from sys import getsizeof
import numpy as np
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
        with open(config_path, "r") as yamlfile:
            self.config = yaml.load(yamlfile, Loader=yaml.FullLoader)
            print("Read successful")

    def run(self):
        split_address = self.config['split_server']['server_ip']
        split_port = self.config['split_server']['server_start_port']+self.client_id-1
        fed_port = self.config['fed_server']['server_start_port']+self.client_id-1
        #log_steps = self.config['log_steps']
        num_epochs = int(self.config['epoch'])
        output_file = self.config['data_server']['output_file']
        rnd = self.config['round']
        self.cut_layer = self.config['cut_layer']


        if (self.config['logging']):
            log_path = os.path.join(
                self.config.get("log_dir", ",/"),
                f"{self.client_id}_{self.config['cut_layer']}_"
                f"{self.config['epoch']}_{self.config['round']}_"
                f"{self.config['batch_size']}_{self.config['device']}.log"
            )
            logging.basicConfig(
                filename= log_path,
                format='%(asctime)s %(message)s',
                filemode='a'
                )
            logger = logging.getLogger()
            # Setting the threshold of logger to DEBUG
            logger.setLevel(logging.INFO)

        if(self.config['device'] == 'cpu'):
            device = 'cpu'
        else:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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
        batch_size = self.config['batch_size']

        #Data Splitting
        match self.config['split_type']:
            case 'n': #No splitting. Use full dataset
                trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                    download=True, transform=transformer)
                sampler = None
                shuffle = True

            case 's': #Use pre-defined split data
                if os.path.exists(output_file+str(self.client_id)):
                    os.remove(output_file+str(self.client_id))

                print(self.config['data_server']['server_address']+"/"+output_file)
                urllib.request.urlretrieve(self.config['data_server']['server_address']+"/"+output_file, output_file+str(self.client_id))

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
                list_of_indices = np.array_split(total_indices, int(self.config['split_type']))
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
                self.cut_layer = config['cut_layer']
                self.logits = config['logits']

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
            

        config = {"cut_layer": self.config['cut_layer'], "logits": 10}
        client_model = ResNet18Client(config).to(device)

        client_optimizer = optim.SGD(
            client_model.parameters(), lr=0.01, momentum=0.9)

        #client_optimizer = optim.Adam(client_model.parameters(), lr=0.001)

        training_start_time = time.time()
        num_rounds = rnd
        
        
        for r in range(num_rounds):

            if r > 0:
                client_model.load_state_dict(global_numpy_weights)
                print("GLOBAL_CLIENT_WEIGHTS_LOADED")
                logging.info("GLOBAL CLIENT WEIGHTS LOADED")
                del global_numpy_weights

            for epoch in range(num_epochs):
                context = zmq.Context()

                print("Connecting to server…")
                socket = context.socket(zmq.REQ)
                url = split_address + ":"+ str(split_port)
                socket.connect(url)
                
                #send cut layer of this model
                socket.send(str(config['cut_layer']).encode())

                socket.recv()

                iterations = len(trainloader)
                send_iterations = str(iterations).encode()
                socket.send(send_iterations)

                socket.recv()

                send_dataset_size = str(datasetsize_used).encode()
                socket.send(send_dataset_size)

                socket.recv()


                epoch_start_time = time.time()

                bar = tqdm(trainloader, desc=f"{r} {epoch}", unit='', ascii=True,
                           bar_format='{desc} {n_fmt}/{total_fmt} {percentage:3.0f}%|{bar}| {postfix}')
                for data in bar:
                    step_start_time = time.time()
                    inputs, labels = data[0].to(device), data[1].to(device)
                    
                    #send labels to server
                    bytes_labels = convert.array_to_bytes(labels.cpu())
                    socket.send(bytes_labels)

                    socket.recv()

                    # Forward prop and sending activations to server
                    activations = client_model(inputs)
                    server_inputs = activations.detach().clone()
                    bytes_server_inputs = convert.array_to_bytes(server_inputs.cpu())
                    
                    server_work_time_start = time.time()
                    socket.send(bytes_server_inputs)

                    recv_grad = socket.recv()
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


                    bar.set_postfix({
                        "step_time": f"{total_one_step_time:.3f}",
                        "server_time": f"{server_work_time:.3f}"
                    })
                    logging.info(
                        f"CLIENT_TOTAL_ONE_STEP_TIME = {total_one_step_time:.3f}    , "
                        f"SERVER_WORK_TIME = {server_work_time:.3f}"
                    )
                    
                    #BATCH OVER

                epoch_end_time = time.time()
                total_one_epoch_time = epoch_end_time - epoch_start_time
                print("CLIENT_TOTAL_ONE_EPOCH_TIME = ", total_one_epoch_time)
                logging.info('CLIENT_TOTAL_ONE_EPOCH_TIME = {:.3f}'.format(
                    total_one_epoch_time))


                socket.close()
                context.term()

            ############################################################
            ########### Sending model to fedServer #####################
            ############################################################

            context1 = zmq.Context()

            #  Socket to talk to server
            print("Connecting to fed_avg server to give weights…")
            socket1 = context1.socket(zmq.REQ)
            url = self.config['fed_server']['server_ip'] + ":" + str(fed_port)
            socket1.connect(url)

            weights = client_model.state_dict()

            print("Size of model weights (before) in bytes is:", getsizeof(weights))
            bytes_weights = convert.ordered_dict_to_bytes(weights)
            print("Size of model weights (after) in bytes is:", getsizeof(bytes_weights))

            socket1.send(bytes_weights)
            socket1.recv()

            ## send dataset size for weighted avg
            socket1.send(send_dataset_size)

            #recieve federated model
            global_weights = socket1.recv()

            print("Size of global model weights (before) in bytes is:", getsizeof(global_weights))
            global_numpy_weights = convert.bytes_to_dict(global_weights)
            print("Size of global model weights (after) in bytes is:", getsizeof(global_numpy_weights))

            socket1.close()
            context1.term()

            #END ROUND



        training_end_time = time.time()
        training_time = training_end_time - training_start_time
        print("CLIENT_TOTAL_TRAINING_TIME = ", training_time)
        logging.info('CLIENT_TOTAL_TRAINING_TIME = {:.3f}'.format(training_time))


#################################################################################################################################
