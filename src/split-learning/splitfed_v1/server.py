"""
arg1 --> CONFIG_FILE_PATH
"""
import copy
import threading
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from torch.autograd import Variable
import time
import zmq
import torch
import yaml
import logging
from ..lib import convert


class Runner:
    def __init__(self, config_path) -> None:
        with open(config_path, "r") as yamlfile:
            self.config = yaml.load(yamlfile, Loader=yaml.FullLoader)
            print("Read successful")
    
    def run(self):
        client_total = self.config["client_total"]
        split_port = self.config["split_server"]["server_start_port"]
        device = self.config["device"]
        cut_layer = self.config["cut_layer"]
        epochs = self.config["epoch"]
        rnd = self.config["round"]

        if(self.config["device"] == 'cpu'):
            device = 'cpu'
        else:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        #Initialize Logger
        if (self.config["logging"]):
            logging.basicConfig(
                filename=(
                    f"./sf_server_{client_total}_{split_port}_{device}_"
                    f"{cut_layer}_{epochs}_{rnd}.log"
                ),
                format='%(asctime)s %(message)s',
                filemode='a'
            )
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            logging.info(
                f"Parameters (SF_SERVER_LOG) ---------- "
                f"[TOTAL_CLIENTS --> {client_total}, STARTING_SERVER_PORT --> {split_port}, "
                f"DEVICE_TYPE --> {device}, CUT_LAYER --> {cut_layer}, "
                f"EPOCHS --> {epochs}, ROUNDS --> {rnd}] ----------"
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

        def worker_routine(url, context, thread_no, r):
            """ Worker routine """
            global server_global_weights
            global server_weights
            global datasetsize_server

            # Socket to talk to dispatcher
            socket = context.socket(zmq.REP)
            socket.bind(url)


            '''
            Model Definition
            '''
            class ResNet18Server(nn.Module):
                """docstring for ResNet"""

                def __init__(self, config):
                    super(ResNet18Server, self).__init__()
                    self.logits = config["logits"]
                    self.cut_layer = config["cut_layer"]

                    self.model = models.resnet18(weights=None)
                    
                    num_ftrs = self.model.fc.in_features
                    self.model.fc = nn.Sequential(nn.Flatten(),
                                                  nn.Linear(num_ftrs, self.logits))
                    
                    self.layers = list(self.model.children())


                def forward(self, x):
                    for i, l in enumerate(self.layers):
                        if i <= self.cut_layer:
                            continue
                        x = l(x)
                    return x
                
                def classify(self, x):
                    return nn.functional.softmax(self.forward(x))

            #initialization...
            model_config = {"cut_layer": cut_layer, "logits": 10}
            server_model = ResNet18Server(model_config).to(device)

            criterion = nn.CrossEntropyLoss()
            server_optimizer = optim.SGD(
                server_model.parameters(), lr=0.01, momentum=0.9)

            #load aggregated server weights
            if r > 0:
                server_model.load_state_dict(server_global_weights)
                print("GLOBAL_SERVER_WEIGHTS_LOADED")

            iterations = socket.recv()
            recv_iterations = int(iterations.decode())
            print(recv_iterations)

            msg = "give_datasize_length"
            send_msg = msg.encode()
            socket.send(send_msg)

            recv_dataset_size = socket.recv()
            dataset_size = int(recv_dataset_size.decode())
            print(dataset_size)

            msg = "Starting the server"
            send_msg = msg.encode()
            socket.send(send_msg)

            num_epochs = epochs

            round_start_time = time.time()

            for epoch in range(num_epochs):
                epoch_start_time = time.time()
                
                for j in range(recv_iterations):
                    step_start_time = time.time()
                    print(f"***TH - {thread_no}***  {epoch} {j}")

                    #recieve labels
                    recv_labels = socket.recv()
                    numpy_labels = convert.bytes_to_array(recv_labels)
                    labels = torch.from_numpy(numpy_labels)
                    labels = labels.to(device)

                    ##dummy......
                    socket.send(send_msg)

                    #get client activations
                    recv_serv_inputs = socket.recv()
                    numpy_server_inputs = convert.bytes_to_array(recv_serv_inputs)
                    server_inputs = torch.from_numpy(numpy_server_inputs)
                    server_inputs = server_inputs.to(device)

                    #forward pass
                    server_inputs = Variable(server_inputs, requires_grad=True)
                    outputs = server_model(server_inputs)

                    server_optimizer.zero_grad()
                    loss = criterion(outputs, labels)
                    loss.backward()

                    #Send gradients back to client.
                    transfer_loss = server_inputs.grad.clone().detach()
                        #only contains grad for client layers
                    server_optimizer.step()

                    bytes_loss = convert.array_to_bytes(transfer_loss.cpu())
                    socket.send(bytes_loss)
                    
                    #telemetry
                    step_end_time = time.time()
                    total_one_step_time = step_end_time - step_start_time
                    print(f"***TH - {thread_no}***  SERVER_TOTAL_ONE_STEP_TIME = {total_one_step_time:.3f}, loss: {loss.item():.3f}")
                    logging.info(f"***TH - {thread_no}***  SERVER_TOTAL_ONE_STEP_TIME = {total_one_step_time:.3f}, loss: {loss.item():.3f}")


                    ################################################################################

                epoch_end_time = time.time()
                total_one_epoch_time = epoch_end_time - epoch_start_time
                print(f"***TH - {thread_no}***  SERVER_TOTAL_ONE_EPOCH_TIME = {total_one_epoch_time:.3f}")
                logging.info(f"***TH - {thread_no}***  SERVER_TOTAL_ONE_EPOCH_TIME = {total_one_epoch_time:.3f}")

                ##################################################################################################################

            round_end_time = time.time()
            round_time = round_end_time - round_start_time
            print(f"***TH - {thread_no}***  SERVER_ROUND_TRAINING_TIME = {round_time:.3f}")
            logging.info(f"***TH - {thread_no}***  SERVER_ROUND_TRAINING_TIME = {round_time:.3f}")

            server_weights.append(server_model.state_dict())
            datasetsize_server.append(dataset_size)

            model_save_name = (
                f"./server_thread_model_r{r}_{thread_no}_{client_total}_"
                f"{split_port}_{device}_{cut_layer}_{epochs}.pt"
            )

            torch.save(server_model.state_dict(), model_save_name)
            print(f"***TH - {thread_no}***  MODEL_SAVED.")

            ##*****************************************************************************************************************
            ##*****************************************************************************************************************
            ##*****************************************************************************************************************

            print("Worker done******************************")

            socket.close()

        def main():
            """ server routine """

            global server_global_weights
            global server_weights
            global datasetsize_server

            server_weights = []
            datasetsize_server = []

            total_threads = client_total
            port_no = split_port
            connection_url = ["tcp://*:" +str(port_no+i) for i in range(total_threads)]

            num_rounds = rnd
            context = zmq.Context()

            training_start_time = time.time()
            for r in range(num_rounds):
                print("New round started..")
                thrs = []

                server_weights.clear()
                datasetsize_server.clear()

                # Launch pool of worker threads
                for i in range(total_threads):  # this defines how many clients can connect
                    thread = threading.Thread(target=worker_routine, args=(connection_url[i], context, i+1, r))
                    thrs.append(thread)
                    thread.start()

                for thread in thrs:         ##have to check when it will run all epochs..
                    thread.join()

                print(f"Length of server weights: {len(server_weights)}")
                print(f"Length of dataset: {len(datasetsize_server)}")


                # Server models weighted averaging..
                server_global_weights = average_weights(server_weights, datasetsize_server)
                model_save_name = (
                    f"./server_fedAvg_model_r{r}_{client_total}_{split_port}_"
                    f"{device}_{cut_layer}_{epochs}_{rnd}.pt"
                )
                torch.save(server_global_weights, model_save_name)
                print("MODEL_SAVED.")


                print("All threads ended..")
            print("All rounds ended..")

            training_end_time = time.time()
            training_time = training_end_time - training_start_time
            print(f"SERVER_TOTAL_TRAINING_TIME = {training_time:.3f}")
            logging.info(f"SERVER_TOTAL_TRAINING_TIME = {training_time:.3f}")

            context.term()

        main()
