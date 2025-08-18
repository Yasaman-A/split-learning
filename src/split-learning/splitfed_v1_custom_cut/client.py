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
        split_address = self.config['split_server']['server_ip']
        split_port = self.config['split_server']['server_start_port']+self.client_id-1
        fed_port = self.config['fed_server']['server_start_port']+self.client_id-1
        log_steps = self.config['log_steps']
        num_epochs = int(self.config['epoch'])
        output_file = self.config['data_server']['output_file']
        rnd = self.config['round']


        metrics = {
            #global time metrics
            "running time"        : 0, 
            "initial loading time": 0,
            "round init time"     : 0,
            "total training time" : 0,
            "server work time"    : 0,
            "testing time"        : 0,
            "fed wait time"       : 0,
            
            #global networking metrics
            "sent to split"  : 0,
            "recv from split": 0,
            "sent to fed"    : 0,
            "recv from fed"  : 0,


            #Per round and epoch time/networking metrics

            "round": {
                "running time"    : 0,
                "training time"   : 0,
                "server work time": 0,
                "testing time"    : 0,
                "fed wait time"   : 0,
                "init time"       : 0,

                "sent to split"  : 0,
                "recv from split": 0,
                "sent to fed"    : 0,
                "recv from fed"  : 0,
            },
            "epoch": {
                "running time"     : 0,
                "training time"    : 0,
                "server work time" : 0,
                "testing time"     : 0,

                "sent to split"  : 0,
                "recv from split": 0,
            },

            "best acc"  : 0,
            "best model": "",
        }


        initial_loading_start_time = time.perf_counter()


         #Initialize Logger
        if (self.config['logging']):
            log_path = os.path.join(
                self.config.get("log_dir", "./"),
                f"{self.client_id}_{cut_layer}_"
                f"{self.config['epoch']}_{self.config['round']}_"
                f"{self.config['batch_size']}_{self.config['device']}.log"
            )
            logging.basicConfig(
                filename=log_path,
                format='%(asctime)s %(message)s',
                filemode='a'
                )
            logger = logging.getLogger()
            # Setting the threshold of logger to DEBUG
            logger.setLevel(logging.INFO)

        if(self.config['device'] == 'cpu'):
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
        batch_size = self.config['batch_size']

        #Data Splitting
        match self.config['split_type']:
            case 'n': #No splitting. Use full dataset
                trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                    download=True, transform=transformer)
                sampler = None
                shuffle = True

                testset = trainset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                    download=True, transform=transformer)

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

                test_file = output_file.replace('.pkl', '_test.pkl')
                test_file_tmp = f"tmp_{self.client_id}_{test_file}"

                if os.path.exists(test_file_tmp):
                    os.remove(test_file_tmp)
                print(f"{self.config['data_server']['server_address']}/{test_file}")

                urllib.request.urlretrieve(
                    f"{self.config['data_server']['server_address']}/{test_file}",
                    test_file_tmp
                    )

                with open(test_file_tmp, 'rb') as handle:
                    testset = pickle.load(handle)
                
                testset = TransformedDataset(testset, transformer)

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

                testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transformer)


        trainloader = torch.utils.data.DataLoader(trainset, 
                                        batch_size=batch_size,
                                        shuffle=shuffle,
                                        sampler = sampler,
                                        num_workers=2,
                                        persistent_workers=True)
        
        testloader = torch.utils.data.DataLoader(testset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=0,
                                            persistent_workers=False
        )
        datasetsize_used = len(trainloader.dataset)



        class ResNet18Client(nn.Module):
            """docstring for ResNet"""

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
        client_model = ResNet18Client(config).to(device)

        client_optimizer = optim.SGD(
            client_model.parameters(), lr=0.01, momentum=0.9)


        num_rounds = rnd

        #Networking Telemetry


        initial_loading_end_time = time.perf_counter()
        metrics['initial loading time'] = initial_loading_end_time - initial_loading_start_time
        print(f"CLIENT_INITIAL_LOADING_TIME = {metrics['initial loading time']}")
        logging.info(f"CLIENT_INITIAL_LOADING_TIME = {metrics['initial loading time']}")

        running_time_start = time.perf_counter()

        '''
        ====================================================        
        BEGIN TRAINING
        ====================================================
        '''

        for r in range(num_rounds):
            metrics['round']['running time']    = 0
            metrics['round']['training time']   = 0
            metrics['round']['server work time']= 0
            metrics['round']['testing time']    = 0
            metrics['round']['fed wait time']   = 0
            metrics['round']['init time']       = 0

            metrics['round']['sent to split']   = 0
            metrics['round']['recv from split'] = 0
            metrics['round']['sent to fed']     = 0
            metrics['round']['recv from fed']   = 0

            round_running_start = time.perf_counter()
            round_init_start = time.perf_counter()
            
            if r > 0:
                client_model.load_state_dict(global_numpy_weights)
                print("GLOBAL_CLIENT_WEIGHTS_LOADED")
                del global_numpy_weights
            


            logging.info(f"\n********ROUND {r}********\n")

            
            #Connect to Split Server
            context = zmq.Context()
            print("Connecting to server…")
            socket = context.socket(zmq.REQ)
            url = split_address + ":"+ str(split_port)
            socket.connect(url)
            # socket.connect("tcp://35.237.244.119:5555")


            socket.send(b"term?")
            term = bool(int(socket.recv().decode()))
            if term: 
                print("Terminate recieved.")
                logging.info("Terminate recieved.")
                socket.close()
                context.term()
                break

            iterations = len(trainloader)
            print(iterations)
            send_iterations = str(iterations).encode()
            socket.send(send_iterations)
            metrics['round']['sent to split'] += len(send_iterations)

            names = socket.recv()
            metrics['round']['recv from split'] += len(names)

            print(datasetsize_used)
            send_dataset_size = str(datasetsize_used).encode()
            socket.send(send_dataset_size)
            metrics['round']['sent to split'] += len(send_dataset_size)

            names = socket.recv()
            metrics['round']['recv from split'] += len(names)

            #send length of test set
            test_iters = len(testloader)
            send_test_iters = str(test_iters).encode()
            socket.send(send_test_iters)

            socket.recv()

            round_init_end = time.perf_counter()
            metrics['round']['init time'] = round_init_end - round_init_start


            for epoch in range(num_epochs):
                logging.info(f"\n********EPOCH {epoch}********\n")
                
                metrics['epoch']['running time']     = 0
                metrics['epoch']['training time']    = 0
                metrics['epoch']['testing time']     = 0
                metrics['epoch']['server work time'] = 0
                
                metrics['epoch']['sent to split'] = 0
                metrics['epoch']['recv from split'] = 0

                epoch_running_start = time.perf_counter()
                
                bar = tqdm(trainloader, desc=f"{r} {epoch}", unit='', ascii=True,
                           bar_format='{desc} {n_fmt}/{total_fmt} {percentage:3.0f}%|{bar}| {postfix}')

                epoch_train_start = time.perf_counter()
                for data in bar:
                    step_start_time = time.perf_counter()
                    inputs, labels = data[0].to(device), data[1].to(device)

                    bytes_labels = convert.array_to_bytes(labels.cpu())
                    socket.send(bytes_labels)
                    metrics['epoch']['sent to split'] += len(bytes_labels)


                    ##dummy......
                    names = socket.recv()
                    metrics['epoch']['recv from split'] += len(names)


                    #forward prop and sending activations to server
                    activations = client_model(inputs)
                    server_inputs = activations.detach().clone()
                    bytes_server_inputs = convert.array_to_bytes(server_inputs.cpu())
                    
                    server_work_time_start = time.perf_counter()
                    socket.send(bytes_server_inputs)
                    metrics['epoch']['sent to split'] += len(bytes_server_inputs)

                    #recover gradient from server
                    recv_grad = socket.recv()
                    metrics['epoch']['recv from split'] += len(recv_grad)
                    server_work_time_end = time.perf_counter()

                    numpy_grad = convert.bytes_to_array(recv_grad)
                    grad_output = torch.from_numpy(numpy_grad)
                    grad_output = grad_output.to(device)

                    client_optimizer.zero_grad()
                    activations.backward(gradient=grad_output)
                    client_optimizer.step()

                    step_end_time = time.perf_counter()
                    total_one_step_time = step_end_time - step_start_time


                    #telemetry
                    step_end_time = time.perf_counter()
                    total_one_step_time = step_end_time - step_start_time
                    server_work_time = server_work_time_end - server_work_time_start

                    metrics['epoch']['server work time'] += server_work_time

                    bar.set_postfix({
                        "step_time": f"{total_one_step_time:.3f}",
                        "server_time": f"{server_work_time:.3f}"
                    })
                    logging.info(
                        f"CLIENT_TOTAL_ONE_STEP_TIME = {total_one_step_time:.3f}    , "
                        f"SERVER_WORK_TIME = {server_work_time:.3f}"
                    )
                    
                    

                    #BATCH OVER

                epoch_train_end = time.perf_counter() 

                #======= TEST SET ========
                # epoch_test_start = time.perf_counter()

                # bar = tqdm(testloader, desc=f"testset: ", unit='', ascii=True,
                #            bar_format='{desc} {n_fmt}/{total_fmt} {percentage:3.0f}%|{bar}| {postfix}')
                # client_model.eval()

                # with torch.no_grad():
                #     for data in bar:
                #         inputs, labels = data[0].to(device), data[1].to(device)
                        
                #         #send labels to server
                #         bytes_labels = convert.array_to_bytes(labels.cpu())
                #         socket.send(bytes_labels)

                #         ##dummy......
                #         names = socket.recv()

                #         #forward prop and sending activations to server
                #         activations = client_model(inputs)
                #         server_inputs = activations.detach().clone()
                #         bytes_server_inputs = convert.array_to_bytes(server_inputs.cpu())                    
                        
                #         socket.send(bytes_server_inputs)

                #         socket.recv()
                
                # socket.send("acc".encode())
                # accuracy = float(socket.recv().decode())

                # if accuracy > metrics['best acc']:
                #     metrics['best acc'] = accuracy
                #     metrics['best model'] = f"r{r}e{epoch}"

                # client_model.train()                
                
                # epoch_test_end = time.perf_counter()
                epoch_running_end = time.perf_counter()
            
                metrics['epoch']['running time']  = epoch_running_end - epoch_running_start
                metrics['epoch']['training time'] = epoch_train_end - epoch_train_start
                #metrics['epoch']['testing time']  = epoch_test_end - epoch_test_start

                metrics['round']['training time']    += metrics['epoch']['training time']
                metrics['round']['server work time'] += metrics['epoch']['server work time']
                metrics['round']['testing time']     += metrics['epoch']['testing time']


                

                #Logging and telemetry
                epoch_running_str  = f"Running time: {metrics['epoch']['running time']}"
                epoch_training_str = f"Training time: {metrics['epoch']['training time']}"
                epoch_testing_str  = f"Testing time: {metrics['epoch']['testing time']}"

                epoch_sent_str = f"Number of bytes sent (activations): {metrics['epoch']['sent to split']}"
                epoch_recv_str = f"Number of bytes recieved (loss gradient): {metrics['epoch']['recv from split']}"

                print(f"========= Client R{r} E{epoch} Statistics ==========")
                print("Time statistics:")
                print(epoch_running_str)
                print(epoch_training_str)
                print(epoch_testing_str)
                print("Training networking statistics")
                print(epoch_sent_str)
                print(epoch_recv_str)
                print("==================================")

                logging.info(f"========= Client R{r} E{epoch} Statistics ==========")
                logging.info("Time statistics:")
                logging.info(epoch_running_str)
                logging.info(epoch_training_str)
                logging.info(epoch_testing_str)
                logging.info("Training networking statistics")
                logging.info(epoch_sent_str)
                logging.info(epoch_recv_str)
                logging.info("==================================")

                metrics['round']['sent to split']   += metrics['epoch']['sent to split']
                metrics['round']['recv from split'] += metrics['epoch']['recv from split']
                #EPOCH OVER


            socket.close()
            context.term()

            model_save_name = os.path.join(
                self.config.get("model_dir", "./"),
                f"cc_client_thread_model_r{r}_{self.client_id}_{split_port}_"
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
            url = str(self.config['fed_server']['server_ip']) + ":"+ str(fed_port)
            socket1.connect(url)

            weights = client_model.state_dict()

            print("Size of model weights (before) in bytes is:", getsizeof(weights))
            bytes_weights = convert.ordered_dict_to_bytes(weights)
            print("Size of model weights (after) in bytes is:", getsizeof(bytes_weights))

            logging.info('Size of model weights (before) in bytes is: %s', (getsizeof(weights)))
            logging.info('Size of model weights (after) in bytes is: %s', (getsizeof(bytes_weights)))
            
            send_weights_time_start = time.perf_counter()
            socket1.send(bytes_weights)
            metrics['round']['sent to fed'] += len(bytes_weights)

            ## dummy recv
            names = socket1.recv()
            metrics['round']['recv from fed'] += len(names)
            send_weights_time_end = time.perf_counter()

            send_weights_time = send_weights_time_end - send_weights_time_start
            logging.info("SEND_WEIGHTS_COMMUNICATION_TIME = {:.3f}".format(send_weights_time))
            print("SEND_WEIGHTS_COMMUNICATION_TIME = {:.3f}".format(send_weights_time))

            ## send dataset size for weighted avg
            socket1.send(send_dataset_size)
            metrics['round']['sent to fed'] += len(send_dataset_size)

            ## dummy recv
            names = socket1.recv()
            metrics['round']['recv from fed'] += len(names)

            ## send cut layer info
            send_cut_layer_size = str(cut_layer).encode()
            socket1.send(send_cut_layer_size)
            metrics['round']['sent to fed'] += len(send_cut_layer_size)

            del weights
            del bytes_weights

            '''
            ====================================================        
            RECEIVE GLOBAL MODEL FROM FED SERVER
            ====================================================
            '''

            weights_waiting_time_start = time.perf_counter()
            
            global_weights = socket1.recv()
            metrics['round']['recv from fed'] += len(global_weights)

            weights_waiting_time_end = time.perf_counter()


            socket1.close()
            context1.term()

            round_running_end = time.perf_counter()
            metrics['round']['running time'] = round_running_end - round_running_start
            metrics['round']['fed wait time'] = weights_waiting_time_end - weights_waiting_time_start


            
            print("Global weights recieved from fedServer")
            print("Size of global model weights (before) in bytes is:", getsizeof(global_weights))
            
            global_numpy_weights = convert.bytes_to_dict(global_weights)
            print("Size of global model weights (after) in bytes is:", getsizeof(global_numpy_weights))

            round_sent_to_servers = metrics['round']['sent to fed'] + metrics['round']['sent to split']
            round_rcvd_from_servers = metrics['round']['recv from fed'] + metrics['round']['recv from split']
            round_total = round_sent_to_servers + round_rcvd_from_servers

            round_running_str     = f"Running time: {metrics['round']['running time']}"
            round_training_str    = f"Training time: {metrics['round']['training time']}"
            round_server_work_str = f"Server work time: {metrics['round']['server work time']}"
            round_testing_str     = f"Testing time: {metrics['round']['testing time']}"
            round_fed_wait_str    = f"Fed wait time: {metrics['round']['fed wait time']}"
            round_init_str        = f"Round Initialization Time: {metrics['round']['init time']}"

            round_sent_to_split_str   = f"Data sent to Split Server: {metrics['round']['sent to split']} bytes"
            round_sent_to_fed_str     = f"Data sent to Fed Server: {metrics['round']['sent to fed']} bytes"
            round_total_sent_str      = f"Total Sent: {round_sent_to_servers} bytes"
            
            round_recv_from_split_str = f"Data received from Split Server: {metrics['round']['recv from split']} bytes"
            round_recv_from_fed_str   = f"Data received from Fed Server: {metrics['round']['recv from fed']} bytes"
            round_total_recv_str      = f"Total Received: {round_rcvd_from_servers} bytes"

            round_total_trans_str     = f"===== \nTotal Transmitted this Round: {round_total} byes"

            print(f"======== Round {r} Summary ========")
            print("Time statistics:")
            print(round_running_str)
            print(round_training_str)
            print(round_server_work_str)
            print(round_testing_str)
            print(round_fed_wait_str)
            print(round_init_str)
            print("Networking statistics:")
            print(round_sent_to_split_str)
            print(round_sent_to_fed_str)
            print(round_total_sent_str)
            print(round_recv_from_split_str)
            print(round_recv_from_fed_str)
            print(round_total_recv_str)
            print(round_total_trans_str)


            logging.info(f"======== Round {r} Summary ========")
            logging.info("Time statistics:")
            logging.info(round_running_str)
            logging.info(round_training_str)
            logging.info(round_server_work_str)
            logging.info(round_testing_str)
            logging.info(round_fed_wait_str)
            logging.info(round_init_str)
            logging.info("Networking statistics:")
            logging.info(round_sent_to_split_str)
            logging.info(round_sent_to_fed_str)
            logging.info(round_total_sent_str)
            logging.info(round_recv_from_split_str)
            logging.info(round_recv_from_fed_str)
            logging.info(round_total_recv_str)
            logging.info(round_total_trans_str)

            
            metrics['round init time']     += metrics['round']['init time']
            metrics['total training time'] += metrics['round']['training time']
            metrics['server work time']    += metrics['round']['server work time']
            metrics['testing time']        += metrics['round']['testing time']
            metrics['fed wait time']       += metrics['round']['fed wait time']
            
            metrics['sent to split']   += metrics['round']['sent to split']
            metrics['recv from split'] += metrics['round']['sent to fed']
            metrics['sent to fed']     += metrics['round']['recv from split']
            metrics['recv from fed']   += metrics['round']['recv from fed']
           
            


            #END ROUND



        running_time_end = time.perf_counter()
        metrics['running time'] = running_time_end - running_time_start

        total_sent_to_servers = metrics['sent to split'] + metrics['recv from split']
        total_rcvd_from_servers = metrics['sent to fed'] + metrics['recv from fed']
        total_data_transmitted = total_sent_to_servers + total_rcvd_from_servers

        total_running_str         = f"Running time: {metrics['running time']}"
        total_init_load_str       = f"Initial loading time: {metrics['initial loading time']}"
        total_round_init_str      = f"Round init time: {metrics['round init time']}"
        total_total_train_str     = f"Total training time: {metrics['total training time']}"
        total_server_work_str     = f"Server work time: {metrics['server work time']}"
        total_testing_str         = f"Testing time: {metrics['testing time']}"
        total_fed_wait_str        = f"Fed wait time: {metrics['fed wait time']}"

        total_sent_split_str      = f"Data sent to Split Server: {metrics['sent to split']} bytes"
        total_recv_split_str      = f"Data received from Split Server: {metrics['recv from split']} bytes"
        total_send_str            = f"Total Sent: {total_sent_to_servers} bytes"
        total_sent_fed_str        = f"Data sent to Fed Server: {metrics['sent to fed']} bytes"
        total_recv_fed_str        = f"Data received from Fed Server: {metrics['recv from fed']} bytes"
        total_recv_str            = f"Total Received: {total_rcvd_from_servers} bytes"
        total_trans_str           = f"Total Transmitted: {total_data_transmitted} bytes"

        best_acc_str               = f"Best client-side model accuracy: {metrics['best acc']}"
        best_model_str             = f"Best client-side model identifier: {metrics['best model']}"


        print("======== Global Summary ========")
        print("Time statistics:")
        print(total_running_str)
        print(total_init_load_str)
        print(total_round_init_str)
        print(total_total_train_str)
        print(total_server_work_str)
        print(total_testing_str)
        print(total_fed_wait_str)
        print("Networking statistics:")
        print(total_sent_split_str)
        print(total_recv_split_str)
        print(total_send_str)
        print(total_sent_fed_str)
        print(total_recv_fed_str)
        print(total_recv_str)
        print(total_trans_str)
        print("Model statistics:")
        print(best_acc_str)
        print(best_model_str)


        logging.info("======== Global Summary ========")
        logging.info("Time statistics:")
        logging.info(total_running_str)
        logging.info(total_init_load_str)
        logging.info(total_round_init_str)
        logging.info(total_total_train_str)
        logging.info(total_server_work_str)
        logging.info(total_testing_str)
        logging.info(total_fed_wait_str)
        logging.info("Networking statistics:")
        logging.info(total_sent_split_str)
        logging.info(total_recv_split_str)
        logging.info(total_send_str)
        logging.info(total_sent_fed_str)
        logging.info(total_recv_fed_str)
        logging.info(total_recv_str)
        logging.info(total_trans_str)
        logging.info("Model statistics:")
        logging.info(best_acc_str)
        logging.info(best_model_str)