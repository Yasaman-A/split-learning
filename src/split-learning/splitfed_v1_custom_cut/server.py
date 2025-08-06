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
import os
# from objsize import get_deep_size


class ResNet18Server(nn.Module):
    """docstring for ResNet"""

    def __init__(self, config):
        super(ResNet18Server, self).__init__()
        self.logits = config['logits']
        self.cut_layer = config['cut_layer']

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


class Runner:
    def __init__(self, config_path) -> None:
        self.server_cut_layer_list = []
        with open(config_path, "r") as yamlfile:
            self.config = yaml.load(yamlfile, Loader=yaml.FullLoader)
            print("Read successful")
        self.terminate = False
    
    def set_extra_options(self, extra):
        self.server_cut_layer_list = [int(x) for x in extra.split(",")] 

    def run(self):
        client_total = self.config['client_total']
        split_port = self.config['split_server']['server_start_port']
        device = self.config['device']
        #cut_layer = self.config['cut_layer']
        epochs = self.config['epoch']
        rnd = self.config['round']

        if(device != 'cpu'):
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(device)


               #Initialize Logger
        if (self.config['logging']):
            log_path = os.path.join(
                self.config.get("log_dir", "./"),
                f"./sf_server_{client_total}_{split_port}_{device}_"
                f"{self.server_cut_layer_list}_{epochs}_{rnd}.log"
            )
            logging.basicConfig(
                filename=log_path,
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

            socket.recv()
            if self.terminate: 
                socket.send(b"1")
                socket.close()
                return
            else: socket.send(b"0")


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

            
            #get length of test set
            test_iters = int(socket.recv().decode())
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
                # TEST SET

                server_model.eval()

                correct = 0
                total = 0

                with torch.no_grad():
                    for j in range(test_iters):
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

                        #dummy
                        socket.send(send_msg)

                        #forward pass
                        server_inputs = Variable(server_inputs, requires_grad=True)
                        outputs = server_model(server_inputs)

                        _, predicted = torch.max(outputs.data, 1)
                        correct += (predicted == labels).sum().item()
                        total += labels.size(0)
                    
                    accuracy = 100 * correct / total if total > 0 else 0

                    socket.recv()
                    socket.send(str(accuracy).encode())
                    print(f" ***TH - {thread_no}*** Accuracy on test set for round {r} epoch {epoch}: {accuracy}%")
                    logging.info(f"***TH - {thread_no}*** Accuracy on test set for round {r} epoch {epoch}: {accuracy}%")
                    server_model.train()



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

            model_save_name = os.path.join(
                self.config.get("model_dir", "./"),
                f"server_thread_model_r{r}_{thread_no}_{client_total}_"
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

            #server_global_exposure = {} #for final aggregation

            total_threads = client_total
            port_no = split_port
            connection_url = ["tcp://*:" +str(port_no+i) for i in range(total_threads)]

            num_rounds = rnd
            context = zmq.Context()
            total_eval_time = 0

            best_accuracy = 0
            best_model = ""
            patience = 0

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

                if self.terminate:
                    break

                print("Length of server weights:", len(server_weights))
                print("Length of dataset:", len(datasetsize_server))


                # Server models weighted averaging..
                        #server_global_exposure removed - unneeded.
                server_global_weights, _ = custom_model_avg(True, server_weights, datasetsize_server, self.server_cut_layer_list)

                model_save_name = os.path.join(
                    self.config.get("model_dir", "./"),
                    f"server_fedAvg_model_r{r}_{client_total}_{split_port}_"
                    f"{device}_{self.server_cut_layer_list}_{epochs}_{rnd}.pt"
                )

                torch.save(server_global_weights, model_save_name)
                print("MODEL_SAVED.")

                print("All threads ended..")

                fed_context = zmq.Context()
                fed_url = f"{self.config['fed_server']['server_ip']}:{self.config['fed_server']['server_start_port'] + client_total}"
                fed_socket = fed_context.socket(zmq.REP)
                fed_socket.connect(fed_url)
                print(f"Connected on {fed_url}")

                fed_iters = int(fed_socket.recv().decode())
                fed_socket.send(b"a")
                
                config = {"cut_layer": self.config['test_cut_layer'], "logits": 10}
                fed_model = ResNet18Server(config).to(device)

                correct = 0
                total = 0
                correct_per_class = torch.zeros(config['logits'], dtype=torch.long)
                total_per_class   = torch.zeros(config['logits'], dtype=torch.long)

                eval_time_start = time.perf_counter()

                with torch.no_grad():
                    for j in range(fed_iters):
                        #receive labels
                        recv_labels = fed_socket.recv()
                        numpy_labels = convert.bytes_to_array(recv_labels)
                        labels = torch.from_numpy(numpy_labels)
                        labels = labels.to(device)

                        ##dummy......
                        fed_socket.send("a".encode())

                        #get client activations
                        recv_serv_inputs = fed_socket.recv()
                        numpy_server_inputs = convert.bytes_to_array(recv_serv_inputs)
                        server_inputs = torch.from_numpy(numpy_server_inputs)
                        server_inputs = server_inputs.to(device)

                        #dummy
                        fed_socket.send("a".encode())

                        #forward pass
                        server_inputs = Variable(server_inputs, requires_grad=True)
                        outputs = fed_model(server_inputs)
                        _, predicted = torch.max(outputs.data, 1)

                        correct += (predicted == labels).sum().item()
                        total += labels.size(0)

                        for class_idx in range(config['logits']):
                            mask = (labels == class_idx)
                            total_per_class[class_idx] += mask.sum().item()
                            correct_per_class[class_idx] += (predicted[mask] == class_idx).sum().item()

                accuracy = (correct / total) * 100 if total > 0 else 0
                per_class_accuracy = correct_per_class.float() / total_per_class.clamp(min=1)

                print("Class\tAccuracy")
                logging.info("Class\tAccuracy")
                for i, acc in enumerate(per_class_accuracy):
                    print(f"{i}\t{acc*100:.4f} ({correct_per_class[i]}/{total_per_class[i]})")
                    logging.info(f"{i}\t{acc*100:.4f} ({correct_per_class[i]}/{total_per_class[i]})")
                print(f"Total Accuracy: {accuracy}")
                logging.info(f"Total Accuracy: {accuracy}")


                if accuracy > best_accuracy:
                    print(f"New best model found! New best accuracy = {accuracy}")
                    logging.info(f"New best model found! New best accuracy = {accuracy}")
                    best_accuracy = accuracy
                    patience = 0
                    best_model = model_save_name
                else:
                    patience += 1

                fed_socket.recv()

                if patience >= self.config['patience']:
                    self.terminate = True
                    print("Patience has run out. Ending experiment.")
                    logging.info("Patience has run out. Ending experiment.")
                    fed_socket.send(b"1")
                else: fed_socket.send(b"0")

                fed_socket.close()
                fed_context.term()

                eval_time_end = time.perf_counter()

                eval_time = eval_time_end - eval_time_start
                total_eval_time += eval_time


            print("All rounds ended..")


            ############################ legacy code
            # print("Sending to fed server...")
            # fed_context = zmq.Context()
            # fed_url = f"{self.config['fed_server']['server_ip']}:{self.config['fed_server']['server_start_port'] + client_total}"
            # fed_socket = fed_context.socket(zmq.REQ)
            # fed_socket.connect(fed_url)
            # print(f"Connected on {fed_url}")


            # bytes_weights = convert.ordered_dict_to_bytes(server_global_weights)
            # fed_socket.send(bytes_weights)
            # fed_socket.recv()
            # print("Weights sent")

            #print(server_global_exposure)

            # bytes_exposure = convert.ordered_dict_to_bytes(server_global_exposure)
            # fed_socket.send(bytes_exposure)
            # fed_socket.recv()
            # print("exposure sent")

            # fed_socket.close()
            # fed_context.term()
            # print("socket closed")

            training_end_time = time.time()
            training_time = training_end_time - training_start_time
            print(f"SERVER_TOTAL_TRAINING_TIME = {training_time:.3f}")
            logging.info(f"SERVER_TOTAL_TRAINING_TIME = {training_time:.3f}")

            print(f"best model was: {best_model} with an accuracy of {best_accuracy}.")


            context.term()

        main()
