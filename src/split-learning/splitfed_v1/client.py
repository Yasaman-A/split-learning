"""
arg1 --> CONFIG_FILE_PATH
arg2 --> CLIENT_ID
"""
import logging
import os
from sys import getsizeof
import yaml

from tqdm.auto import tqdm
import torch
from torch import optim
import zmq

from ..architectures import get_architecture_bundle
from ..lib import convert
from ..lib.data_prep import DataPrep
from ..lib.metrics import Metrics


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
        num_epochs = int(self.config['epoch'])
        rnd = self.config['round']

        model_architecture = self.config.get("model_architecture", "ResNet18_CIFAR10")
        arch = get_architecture_bundle(model_architecture)

        metrics = Metrics()

        with metrics.initial_loading_timer():

            #Initialize Logger
            if self.config['logging']:
                log_path = os.path.join(
                    self.config.get("log_dir", "./"),
                    f"{self.client_id}_{self.config['cut_layer']}_"
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

            if self.config['device'] == 'cpu':
                device = 'cpu'
            else:
                device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            print(device)

            model_config = {"cut_layer": int(self.config['cut_layer']), "logits": 10}
            client_model = arch.client(model_config).to(device)
            client_optimizer = optim.SGD(client_model.parameters(), lr=0.01, momentum=0.9)

            data_prepper = DataPrep(self.config, arch, self.client_id)
            trainloader = data_prepper.get_training_loader()
            datasetsize_used = len(trainloader)

            num_rounds = rnd

            #END INIT_TIMER

        out = f"CLIENT_INITIAL_LOADING_TIME = {metrics.overall.initial_loading_time}"
        print(out)
        logging.info(out)




        #====================================================        
        #BEGIN TRAINING
        #====================================================

        with metrics.overall_running_timer():

            for r in range(num_rounds):
                with metrics.round_running_timer():
                    with metrics.round_init_timer():
                        if r > 0:
                            client_model.load_state_dict(global_numpy_weights)
                            print("GLOBAL_CLIENT_WEIGHTS_LOADED")
                            del global_numpy_weights

                        logging.info(f"********ROUND {r}********\n")

                        #connect to Split Server
                        context = zmq.Context()
                        print("Connecting to split server…")
                        socket = context.socket(zmq.REQ)
                        url = split_address + ":"+ str(split_port)
                        socket.connect(url)

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

                    for epoch in range(num_epochs):
                        logging.info(f"********EPOCH {epoch}********\n")

                        with metrics.epoch_running_timer():
                            loading_bar = tqdm(
                                trainloader,
                                desc=f"{r} {epoch}",
                                unit='',
                                ascii=True,
                                bar_format='{desc} {n_fmt}/{total_fmt} {percentage:3.0f}%|{bar}| {postfix}'
                                )
                            with metrics.epoch_training_timer():
                                for data in loading_bar:
                                    with metrics.step_timer():
                                        inputs, labels = data[0].to(device), data[1].to(device)

                                        bytes_labels = convert.array_to_bytes(labels.cpu())
                                        socket.send(bytes_labels)
                                        metrics.epoch.sent_to_split += len(bytes_labels)

                                        ##dummy......
                                        names = socket.recv()
                                        metrics.epoch.recv_from_split += len(names)

                                        #forward prop and sending activations to server
                                        activations = client_model(inputs)
                                        server_inputs = activations.detach().clone()
                                        bytes_server_inputs = convert.array_to_bytes(server_inputs.cpu())

                                        with metrics.server_timer():
                                            socket.send(bytes_server_inputs)
                                            metrics.epoch.sent_to_split += len(bytes_server_inputs)

                                            #recover gradient from server
                                            recv_grad = socket.recv()
                                            metrics.epoch.recv_from_split += len(recv_grad)
                                            #END SERVER_TIMER

                                        numpy_grad = convert.bytes_to_array(recv_grad)
                                        grad_output = torch.from_numpy(numpy_grad)
                                        grad_output = grad_output.to(device)

                                        client_optimizer.zero_grad()
                                        activations.backward(gradient=grad_output)
                                        client_optimizer.step()
                                        #END EPOCH_STEP_TIMER

                                    loading_bar.set_postfix({
                                        "step_time": f"{metrics.last_step_time:.3f}",
                                        "server_time": f"{metrics.last_server_work_time:.3f}"
                                    })
                                    logging.info(
                                        f"CLIENT_TOTAL_ONE_STEP_TIME = {metrics.last_step_time:.3f}    , "
                                        f"SERVER_WORK_TIME = {metrics.last_server_work_time:.3f}"
                                    )
                                    #BATCH OVER
                                #END_EPOCH_TRAIN_TIMER
                            #END EPOCH_RUNNING_TIMER

                        metrics.reportEpoch(r, epoch, logger)
                        #EPOCH OVER

                    socket.close()
                    context.term()


                    model_save_name = os.path.join(
                        self.config.get("model_dir", "./"),
                        f"./client_thread_model_r{r}_{self.client_id}_{split_port}_"
                        f"{self.config.get('device', 'cpu')}_{self.config['cut_layer']}_"
                        f"{self.config['epoch']}_{self.config['split_type']}_{self.client_id}_"
                        f"{self.config['batch_size']}_{self.config['round']}_{fed_port}.pt"
                    )

                    torch.save(client_model.state_dict(), model_save_name)
                    print("***TH - {self.client_id}***  MODEL_SAVED.")



                    #====================================================        
                    #SEND DATA TO FED SERVER
                    #====================================================


                    context1 = zmq.Context()

                    #  Socket to talk to server
                    print("Connecting to fed_avg server to give weights…")
                    socket1 = context1.socket(zmq.REQ)
                    url = str(self.config['fed_server']['server_ip']) + ":"+ str(fed_port)
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

                    out = f"SEND_WEIGHTS_COMMUNICATION_TIME = {metrics.round.send_weights_time:.3f}"
                    print(out)
                    logging.info(out)

                    ## send dataset size for weighted avg
                    socket1.send(send_dataset_size)
                    metrics.round.sent_to_fed += len(send_dataset_size)

                    del weights
                    del bytes_weights


                    #====================================================        
                    #RECEIVE GLOBAL MODEL FROM FED SERVER
                    #====================================================

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
                #END ROUND
            #END OVERALL_RUNNING_TIMER


        metrics.reportOverall(logger)
