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
from ..lib.transformed_dataset import TransformedDataset
from ..lib.metrics import Metrics
from ..architectures import get_architecture_bundle
from sys import getsizeof
import numpy as np
import urllib.request
import os
import yaml
import logging
from datetime import datetime
from tqdm.auto import tqdm
class ShadesOfGray(object):
    """
    Implements Shades of Gray color constancy (Minkowski Norm p=6).
    This normalizes lighting artifacts which are common in dermoscopy images.
    """
    def __init__(self, power=6):
        self.power = power

    def __call__(self, img):
        # Ensure img is a tensor for calculation
        if not isinstance(img, torch.Tensor):
            t_img = transforms.functional.to_tensor(img)
        else:
            t_img = img
            
        # 1. Calculate the Minkowski norm (estimation of the illuminant)
        # Flatten spatial dims: (C, H, W) -> (C, H*W)
        c, h, w = t_img.shape
        img_flat = t_img.view(c, -1)
        
        # Power p, mean, then root p
        illum = img_flat.pow(self.power).mean(dim=1).pow(1.0/self.power)
        
        # 2. Normalize the image by the illuminant
        # Shape handling for broadcasting (C, 1, 1)
        illum = illum.view(c, 1, 1)
        
        # Avoid division by zero
        normalized = t_img / (illum + 1e-8)
        
        # 3. Optional: Scale so the mean matches a standard gray (e.g., 0.5) 
        # or clip to [0, 1]. For neural nets, standardizing the mean is usually enough.
        # We clip to ensure validity.
        normalized = torch.clamp(normalized, 0, 1)
        
        # If the input was a PIL image, we usually return a Tensor here anyway 
        # because this transform sits in a chain.
        return normalized

class Runner:
    def __init__(self, config_path) -> None:
        self.client_id = -1
        self.input_cut_layer = 0
        with open(config_path, "r") as yamlfile:
            self.config = yaml.load(yamlfile, Loader=yaml.FullLoader)
            print("Read successful")

    def set_extra_options(self, extra):
        self.input_cut_layer = atoi(extra)

    def run(self):
        cut_layer = self.input_cut_layer
        split_address = self.config["split_server"]["server_ip"]
        split_port = (
            self.config["split_server"]["server_start_port"] + self.client_id - 1
        )
        fed_port = self.config["fed_server"]["server_start_port"] + self.client_id - 1
        log_steps = self.config["log_steps"]
        num_epochs = int(self.config["epoch"])
        output_file = self.config["data_server"]["output_file"]
        rnd = self.config["round"]
        test_last_model = self.config.get("test_last_model", False)

        # Load architecture
        model_architecture = self.config.get("model_architecture", "ResNet18_CIFAR10")
        arch = get_architecture_bundle(model_architecture)
        logits = self.config.get("logits", 7)#10)
        mu = float(self.config.get("fedprox_mu", 0.0))

        metrics = Metrics()
        
        with metrics.initial_loading_timer():

            # Initialize Logger
            if self.config["logging"]:
                log_dir = self.config.get("log_dir", "./logs")
                run_id_path = os.path.join(log_dir, ".run_id")
                # Wait up to 30s for server.py to write .run_id
                for _ in range(30):
                    if os.path.exists(run_id_path):
                        break
                    time.sleep(1)
                if os.path.exists(run_id_path):
                    with open(run_id_path) as f:
                        run_tag = f.read().strip()
                else:
                    # Fallback if server.py never wrote .run_id
                    loss_fn = self.config.get("loss_function", "CE")
                    dataset = os.path.splitext(self.config["data_server"]["output_file"])[0]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                    run_tag = f"{model_architecture}_{loss_fn}_{dataset}_{timestamp}"

                run_dir = os.path.join(log_dir, run_tag)
                os.makedirs(run_dir, exist_ok=True)
                new_log_path = os.path.join(run_dir, f"client_{self.client_id}.log")
                old_log_path = os.path.join(
                    log_dir,
                    f"{self.client_id}_{cut_layer}_"
                    f"{self.config['epoch']}_{self.config['round']}_"
                    f"{self.config['batch_size']}_{self.config['device']}.log",
                )
                formatter = logging.Formatter("%(asctime)s %(message)s")
                logger = logging.getLogger()
                logger.setLevel(logging.INFO)
                for path, mode in [(new_log_path, "w"), (old_log_path, "a")]:
                    h = logging.FileHandler(path, mode=mode)
                    h.setFormatter(formatter)
                    logger.addHandler(h)
    
            if self.config["device"] == "cpu":
                device = "cpu"
            else:
                device = (
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
            print(device)
    
            # Data Preparation
            transformer = arch.training_transformer
            transformer_eval = arch.eval_transformer
            batch_size = self.config["batch_size"]
            # Data Splitting
            match self.config["split_type"]:
                case "n":  # No splitting. Use full dataset
                    trainset = torchvision.datasets.CIFAR10(
                        root="./data", train=True, download=True, transform=transformer
                    )
                    sampler = None
                    shuffle = True

                    if test_last_model:
                        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transformer_eval)

    
                case "s":  # Use pre-defined split data
                    if os.path.exists(output_file + str(self.client_id)):
                        os.remove(output_file + str(self.client_id))

                    print(self.config["data_server"]["server_address"] + "/" + output_file)
                    urllib.request.urlretrieve(
                        self.config["data_server"]["server_address"] + "/" + output_file,
                        output_file + str(self.client_id),
                    )

                    with open(output_file + str(self.client_id), "rb") as handle:
                        datasets = pickle.load(handle)
                        dataset = datasets[self.client_id - 1]

                    trainset = TransformedDataset(dataset, transformer)
                    sampler = None
                    shuffle = True

                    if test_last_model:
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

                        testset = TransformedDataset(testset, transformer_eval)
            
                case "_":  # Split into 'split_type' number of blocks.
                    trainset = torchvision.datasets.CIFAR10(
                        root="./data", train=True, download=True, transform=transformer
                    )
                    dataset_size = len(trainset)
                    total_indices = list(range(dataset_size))
                    list_of_indices = np.array_split(
                        total_indices, int(self.config["split_type"])
                    )
                    use_indices = list_of_indices[self.client_id]
                    datasetsize_used = len(use_indices)
                    print("use_indices:" + str(use_indices))
    
                    sampler = torch.utils.data.SubsetRandomSampler(use_indices)
                    shuffle = False

                    if test_last_model:
                        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transformer_eval)
   
            trainloader = torch.utils.data.DataLoader(
                trainset,
                batch_size=batch_size,
                shuffle=shuffle,
                sampler=sampler,
                num_workers=0,
                drop_last=True,
                persistent_workers=False,
            )

            if test_last_model:
                test_loader = torch.utils.data.DataLoader(testset,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    num_workers=0,
                                                    drop_last=True,
                                                    persistent_workers=False
                )
    
            datasetsize_used = len(trainloader.dataset)
    
            model_config = {"cut_layer": int(cut_layer), "logits": logits}
            client_model = arch.client(model_config).to(device)
    
            client_optimizer = optim.SGD(
                client_model.parameters(),
                lr=0.01,
                momentum=0.9,
            )
            # client_scheduler = optim.lr_scheduler.StepLR(
            #     client_optimizer, step_size=15, gamma=0.1
            # )

            num_rounds = rnd
    
        
            
            #END INIT_TIMER
   
        out = f"CLIENT_INITIAL_LOADING_TIME = {metrics.overall.initial_loading_time}"
        print(out)
        logging.info(out)

        """
        ====================================================        
        BEGIN TRAINING
        ====================================================
        """
        
        with metrics.overall_running_timer():
            
            for r in range(num_rounds):
                with metrics.round_running_timer(): 
                    with metrics.round_init_timer():
                        if r > 0:
                            client_model.load_state_dict(global_numpy_weights)
                            print("GLOBAL_CLIENT_WEIGHTS_LOADED")
                            del global_numpy_weights
            
                        logging.info(f"********ROUND {r}********\n")
            
                        # Connect to Split Server
                        context = zmq.Context()
                        print("Connecting to server…")
                        socket = context.socket(zmq.REQ)
                        socket.setsockopt(zmq.LINGER, 0)
                        url = split_address + ":" + str(split_port)
                        socket.connect(url)
            
                        socket.send(b"term?")
                        term = bool(int(socket.recv().decode()))
                        if term:
                            print("Terminate recieved.")
                            logging.info("Terminate recieved.")
                            if test_last_model:
                                test_client(device, client_model, socket, test_loader)
                            socket.close()
                            context.term()
                            break
            
                        iterations = len(trainloader)
                        print(iterations)
                        send_iterations = str(iterations).encode()
                        socket.send(send_iterations)
                        metrics.round.sent_to_split += len(send_iterations)
            
                        names = socket.recv()
                        metrics.round.recv_from_split += len(names)
            
                        print(datasetsize_used)
                        send_dataset_size = str(datasetsize_used).encode()
                        socket.send(send_dataset_size)
                        metrics.round.sent_to_split += len(send_dataset_size)
            
                        names = socket.recv()
                        metrics.round.recv_from_split += len(names)
            
                        #END ROUND_INIT_TIMER

                    # FedProx: snapshot weights at round start for proximal term
                    w0_client = {n: p.data.clone() for n, p in client_model.named_parameters()} if (mu > 0 and r > 0) else None

                    for epoch in range(num_epochs):
                        logging.info(f"********EPOCH {epoch}********\n")
                        
                        with metrics.epoch_running_timer():
                            bar = tqdm(
                                trainloader,
                                desc=f"{r} {epoch}",
                                unit="",
                                ascii=True,
                                bar_format="{desc} {n_fmt}/{total_fmt} {percentage:3.0f}%|{bar}| {postfix}",
                                )
                            with metrics.epoch_training_timer():
                                for data in bar:
                                    with metrics.step_timer():
                                        inputs, labels = data[0].to(device), data[1].to(device)
                    
                                        bytes_labels = convert.array_to_bytes(labels.cpu())
                                        socket.send(bytes_labels)
                                        metrics.epoch.sent_to_split += len(bytes_labels)
                    
                                        ##dummy......
                                        names = socket.recv()
                                        metrics.epoch.recv_from_split += len(names)
                    
                                        # forward prop and sending activations to server
                                        activations = client_model(inputs)
                                        server_inputs = activations.detach().clone()
                                        bytes_server_inputs = convert.array_to_bytes(server_inputs.cpu())
                    
                                        with metrics.server_timer():
                                            socket.send(bytes_server_inputs)
                                            metrics.epoch.sent_to_split += len(bytes_server_inputs)
                    
                                            # recover gradient from server
                                            recv_grad = socket.recv()
                                            metrics.epoch.recv_from_split += len(recv_grad)
                                            #END SERVER_TIMER

                                        numpy_grad = convert.bytes_to_array(recv_grad)
                                        grad_output = torch.from_numpy(numpy_grad)
                                        grad_output = grad_output.to(device)
                    
                                        client_optimizer.zero_grad()
                                        activations.backward(gradient=grad_output)
                                        if w0_client is not None:
                                            for n, p in client_model.named_parameters():
                                                if p.grad is not None:
                                                    p.grad.data.add_(mu * (p.data - w0_client[n]))
                                        client_optimizer.step()
                                        #END EPOCH_STEP_TIMER
                
                                    bar.set_postfix(
                                        {
                                            "step_time": f"{metrics.last_step_time:.3f}",
                                            "server_time": f"{metrics.last_server_work_time:.3f}",
                                        }
                                    )
                                    logging.info(
                                        f"CLIENT_TOTAL_ONE_STEP_TIME = {metrics.last_step_time:.3f}    , "
                                        f"SERVER_WORK_TIME = {metrics.last_server_work_time:.3f}"
                                    )
                                    # BATCH OVER
                                #END EPOCH_TRAIN_TIMER
                            #END EPOCH_RUNNING_TIMER
                        
                        metrics.reportEpoch(r, epoch, logger)
                        # EPOCH OVER

                    if test_last_model and (r == num_rounds - 1):
                        test_client(device, client_model, socket, test_loader)

                    socket.close()
                    context.term()
        
                    # model_save_name = os.path.join(
                    #     self.config.get("model_dir", "./"),
                    #     f"cc_client_thread_model_r{r}_{self.client_id}_{split_port}_"
                    #     f"{self.config['device']}_{cut_layer}_{self.config['epoch']}_"
                    #     f"{self.config['split_type']}_{self.client_id}_{self.config['batch_size']}_"
                    #     f"{self.config['round']}_{fed_port}.pt",
                    # )
                    # torch.save(client_model.state_dict(), model_save_name)
                    print("***TH - {}***  MODEL_SAVED.".format(self.client_id))
        
                    """
                    ====================================================        
                    SEND DATA TO FED SERVER
                    ====================================================
                    """
        
                    context1 = zmq.Context()
        
                    #  Socket to talk to server
                    print("Connecting to fed_avg server to give weights…")
                    socket1 = context1.socket(zmq.REQ)
                    url = str(self.config["fed_server"]["server_ip"]) + ":" + str(fed_port)
                    socket1.connect(url)
        
                    weights = client_model.state_dict()
        
                    out = f"Size of model weights (before) in bytes is: {getsizeof(weights)}"
                    print(out)
                    logging.info(out)
                    
                    bytes_weights = convert.ordered_dict_to_bytes(weights)
                    
                    out = f"Size of model weights (after) in bytes is: {getsizeof(bytes_weights)}"
                    print(out) 
                    logging.info(out)
        
                    with metrics.weights_sending_timer():
                        socket1.send(bytes_weights)
                        metrics.round.sent_to_fed += len(bytes_weights)
        
                        ## dummy recv
                        names = socket1.recv()
                        metrics.round.recv_from_fed += len(names)
                        #END WEIGHTS_SENDING_TIMER
        
                    out = "SEND_WEIGHTS_COMMUNICATION_TIME = {:.3f}".format(metrics.round.send_weights_time)
                    logging.info(out)
                    print(out)

                    ## send dataset size for weighted avg
                    socket1.send(send_dataset_size)
                    metrics.round.sent_to_fed += len(send_dataset_size)
        
                    ## dummy recv
                    names = socket1.recv()
                    metrics.round.recv_from_fed += len(names)
        
                    ## send cut layer info
                    send_cut_layer_size = str(cut_layer).encode()
                    socket1.send(send_cut_layer_size)
                    metrics.round.sent_to_fed += len(send_cut_layer_size)
        
                    del weights
                    del bytes_weights
        
                    """
                    ====================================================        
                    RECEIVE GLOBAL MODEL FROM FED SERVER
                    ====================================================
                    """
        
                    with metrics.weights_receiving_timer():
                        global_weights = socket1.recv()
                        metrics.round.recv_from_fed += len(global_weights)
                        #END WEIGHTS_RECEIVING_TIMER
        
                    socket1.close()
                    context1.term()
                    #END ROUND_RUNNING_TIMER
 
                print("Global weights recieved from fedServer")
                print(
                    "Size of global model weights (before) in bytes is:",
                    getsizeof(global_weights),
                )
    
                global_numpy_weights = convert.bytes_to_dict(global_weights)
                print(
                    "Size of global model weights (after) in bytes is:",
                    getsizeof(global_numpy_weights),
                )
                
                metrics.reportRound(r, logger)
                # client_scheduler.step()
                # END ROUND
                

            #Test last set of client accuracies against the test set
            #END OVERALL_RUNNING_TIMER

        metrics.reportOverall(logger)





def test_client(device, client_model, socket, test_loader):
    bar = tqdm(test_loader, desc=f"testset: ", unit='', ascii=True,
               bar_format='{desc} {n_fmt}/{total_fmt} {percentage:3.0f}%|{bar}| {postfix}')
    client_model.eval()

    socket.send(str(len(test_loader)).encode())
    socket.recv()

    with torch.no_grad():
        for data in bar:
            inputs, labels = data[0].to(device), data[1].to(device)

            #send labels to server
            bytes_labels = convert.array_to_bytes(labels.cpu())
            socket.send(bytes_labels)

            ##dummy......
            names = socket.recv()

            #forward prop and sending activations to server
            activations = client_model(inputs)
            server_inputs = activations.detach().clone()
            bytes_server_inputs = convert.array_to_bytes(server_inputs.cpu())

            socket.send(bytes_server_inputs)

            socket.recv()

    socket.send("acc".encode())
    accuracy = float(socket.recv().decode())

    out = f"Final client model accuracy on test set: {accuracy}"
    print(out)
    logging.info(out)

    client_model.train()