"""
arg1 --> CONFIG_FILE_PATH
arg2..x --> cut layers for all clients
"""

"""
arg1 --> total number of clients
arg2 --> STARTING_PORT_NO
arg3 --> 'cpu' or 'gpu'
arg4 --> cut_layer(same for all) or 'list' at the end
arg5 --> epochs
arg6 --> round
arg7..x --> cut layers for all clients
"""

# eg command: python server_splitnn_th_REPREQ.py 2 5555 cpu

#from locale import atoi
import copy
import threading
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from torch.autograd import Variable
import time
import zmq
import torch
from ..lib import convert
from .custom_model_avg import custom_model_avg
import yaml
import logging
# from objsize import get_deep_size


class Runner:
    def __init__(self, config_path) -> None:
        self.server_cut_layer_list = []
        with open(config_path, "r") as yamlfile:
            self.config = yaml.load(yamlfile, Loader=yaml.FullLoader)
            print("Read successful")
    
    def set_extra_options(self, extra):
        self.server_cut_layer_list = [int(x) for x in extra.split(",")] 

    def run(self):
        client_total = self.config["client_total"]
        split_port = self.config["split_server"]["server_start_port"]
        device = self.config["device"]
        #cut_layer = self.config["cut_layer"]
        epochs = self.config["epoch"]
        rnd = self.config["round"]

        if(device != 'cpu'):
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(device)


               #Initialize Logger
        if (self.config["logging"]):
            logging.basicConfig(
                filename=(
                    f"./sf_server_{client_total}_{split_port}_{device}_"
                    f"{self.server_cut_layer_list}_{epochs}_{rnd}.log"
                ),
                format='%(asctime)s %(message)s',
                filemode='a'
            )
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            logging.info(
                f"Parameters (SF_SERVER_LOG) ---------- "
                f"[TOTAL_CLIENTS --> {client_total}, STARTING_SERVER_PORT --> {split_port}, "
                f"DEVICE_TYPE --> {device}, "
                f"EPOCHS --> {epochs}, ROUNDS --> {rnd}] ----------"
            )

        def worker_routine(url, context, thread_no, r, cut_layer):
            """ Worker routine """

            global server_global_weights
            global server_weights
            global datasetsize_server

            socket = context.socket(zmq.REP)

            socket.bind(url)



            class ResNet18Server(nn.Module):
                """docstring for ResNet"""

                def __init__(self, config):
                    super(ResNet18Server, self).__init__()
                    self.logits = config["logits"]
                    self.cut_layer = cut_layer

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

            config = {"cut_layer": cut_layer, "logits": 10}
            server_model = ResNet18Server(config).to(device)

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

                    #receive labels
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
            print("***TH - {}***  MODEL_SAVED." .format(thread_no))



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

            server_global_exposure = {} #for final aggregation

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
                    thread = threading.Thread(target=worker_routine, args=(connection_url[i], context, i+1, r, self.server_cut_layer_list[i]))
                    thrs.append(thread)
                    thread.start()

                for thread in thrs: 
                    thread.join()

                print("Length of server weights:", len(server_weights))
                print("Length of dataset:", len(datasetsize_server))


                # Server models weighted averaging..
                server_global_weights, server_global_exposure = custom_model_avg(True, server_weights, datasetsize_server, self.server_cut_layer_list)

                model_save_name = (
                    f"./server_fedAvg_model_r{r}_{client_total}_{split_port}_"
                    f"{device}_{self.server_cut_layer_list}_{epochs}_{rnd}.pt"
                )

                torch.save(server_global_weights, model_save_name)
                print("MODEL_SAVED.")

                print("All threads ended..")
            print("All rounds ended..")

            print("Sending to fed server...")
            fed_context = zmq.Context()
            fed_url = f"{self.config["fed_server"]["server_ip"]}:{self.config["fed_server"]["server_start_port"] + client_total}"
            fed_socket = fed_context.socket(zmq.REQ)
            fed_socket.connect(fed_url)
            print(f"Connected on {fed_url}")


            bytes_weights = convert.ordered_dict_to_bytes(server_global_weights)
            fed_socket.send(bytes_weights)
            fed_socket.recv()
            print("Weights sent")

            print(server_global_exposure)

            bytes_exposure = convert.ordered_dict_to_bytes(server_global_exposure)
            fed_socket.send(bytes_exposure)
            fed_socket.recv()
            print("exposure sent")

            fed_socket.close()
            fed_context.term()
            print("socket closed)")

            training_end_time = time.time()
            training_time = training_end_time - training_start_time
            print(f"SERVER_TOTAL_TRAINING_TIME = {training_time:.3f}")
            logging.info(f"SERVER_TOTAL_TRAINING_TIME = {training_time:.3f}")

            context.term()

        main()
