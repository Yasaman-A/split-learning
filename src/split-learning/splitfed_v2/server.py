
"""
arg1 --> CONFIG_FILE_PATH
"""

# eg command: python server_splitnn_th_REPREQ.py 2 5555 cpu

import torch.nn as nn
from torchvision import models
import torch.optim as optim
from torch.autograd import Variable
import time
import zmq
import torch
from ..lib import convert
from ..architectures import get_architecture_bundle
import random
import yaml
import logging
import os


class Runner:
    def __init__(self, config_path) -> None:
        with open(config_path, "r") as yamlfile:
            self.config = yaml.load(yamlfile, Loader=yaml.FullLoader)
            print("Read successful")
        self.terminate = False
    
    def run(self):
        client_total = self.config['client_total']
        split_port = self.config['split_server']['server_start_port']
        device = self.config['device']
        self.cut_layer = self.config['cut_layer']
        epochs = self.config['epoch']
        rnd = self.config['round']

        if(device != 'cpu'):
                device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


               #Initialize Logger
        if (self.config['logging']):
            log_path = os.path.join(
                    self.config.get("log_dir", "./"),
                    f"./sf_server_{client_total}_{split_port}_{device}_"
                    f"{self.cut_layer}_{epochs}_{rnd}.log"
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
                f"DEVICE_TYPE --> {device}, CUT_LAYER --> {self.cut_layer}, "
                f"EPOCHS --> {epochs}, ROUNDS --> {rnd}] ----------"
            )


        def main():
            """ server routine """
            
            model_architecture = self.config.get("model_architecture", "ResNet18_CIFAR10")
            arch = get_architecture_bundle(model_architecture)
            model_config = {"cut_layer": self.cut_layer, "logits": self.config.get("logits", 10)}
            server_model = arch.server(model_config).to(device)

            criterion = nn.CrossEntropyLoss()
            server_optimizer = optim.SGD(
                server_model.parameters(), lr=0.01, momentum=0.9)

            port_no = split_port
            connection_url = ["tcp://*:" +str(port_no+i) for i in range(client_total)]

            num_rounds = rnd
            num_epochs = epochs

            context = zmq.Context()

            sockets = []

            for url in connection_url:
                socket = context.socket(zmq.REP)
                socket.bind(url)
                sockets.append(socket)

            best_accuracy = 0
            best_model = ""
            patience = 0

            training_start_time = time.time()
            for r in range(num_rounds):
                print("New round started..")

                for epoch in range(num_epochs): 
                    random.shuffle(sockets)
                    for cl in range(client_total):
                        #use socket for random client n
                        client_no = cl + 1
                        socket = sockets[cl]
                        if epoch == 0:
                            socket.recv()
                            if self.terminate:
                                socket.send(b"1")
                                # if test_last_model:
                                #     test_client(device, server_model, socket, thread_no)
                                socket.close()
                                continue
                            else:
                                socket.send(b"0")
                        

                        """ Worker routine """


                        cut_layer = int(socket.recv().decode())
                        print(cut_layer)
                        server_model.change_cut(cut_layer)
                        socket.send(b'ack')

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

                        epoch_start_time = time.time()

                        server_optimizer = optim.SGD(
                            server_model.parameters(), lr=0.01, momentum=0.9)

                        for j in range(recv_iterations):
                            step_start_time = time.time()
                            print("***CL - {}*** {}".format(client_no, j))

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

                            with torch.no_grad():
                                probs = torch.softmax(outputs, dim=1)
                                avg_conf = probs.max(dim=1).values.mean().item()
                                if avg_conf > 0.6:
                                    print(f"[Round: {r}, Epoch: {epoch}] Avg softmax confidence: {avg_conf:.4f}")
                                entropy = -(probs * probs.log()).sum(dim=1).mean().item()
                                if entropy < 1:
                                    print(f"[Round: {r}, Epoch {epoch}] Prediction entropy: {entropy:.4f}")


                            server_optimizer.zero_grad()
                            loss = criterion(outputs, labels)
                            loss.backward()

                            if loss.item() < 0.5:
                                print(f"[Round: {r}, Epoch {epoch}] Loss: {loss.item():.6f}")

                            #send gradients back to client
                            transfer_loss = server_inputs.grad.clone().detach()
                                #only contains grad for client layers
                            server_optimizer.step()

                            
                            bytes_loss = convert.array_to_bytes(transfer_loss.cpu())
                            socket.send(bytes_loss)


                            step_end_time = time.time()
                            total_one_step_time = step_end_time - step_start_time
                            print("***CL - {}***  SERVER_TOTAL_ONE_STEP_TIME = {:.3f}".format(client_no, total_one_step_time))
                            logging.info(
                                '***CL - {}***  SERVER_TOTAL_ONE_STEP_TIME = {:.3f}'.format(client_no, total_one_step_time))

                            #end batch

                        epoch_end_time = time.time()
                        total_one_epoch_time = epoch_end_time - epoch_start_time
                        print("***CL - {}***  SERVER_TOTAL_ONE_EPOCH_TIME = {:.3f}" .format(client_no,
                            total_one_epoch_time))
                        logging.info(
                            '***CL - {}***  SERVER_TOTAL_ONE_EPOCH_TIME = {:.3f}'.format(client_no, total_one_epoch_time))

                        ##################################################################################################################

                        ##*****************************************************************************************************************
                        ##*****************************************************************************************************************
                        ##*****************************************************************************************************************

                        print("Worker done******************************")

                        ##################################################
                        ##################################################
                    if self.terminate: break
                    print("All clients served..")
                if self.terminate: break
                model_save_name = os.path.join(
                    self.config.get("model_dir", "./"),
                    f"./server_fedAvg_model_r{r}_{client_total}_{split_port}_"
                    f"{device}_{self.cut_layer}_{epochs}_{rnd}.pt"
                )
                torch.save(server_model.state_dict(), model_save_name)

                server_global_weights = server_model.state_dict()
                print("MODEL_SAVED.")


                fed_context = zmq.Context()
                fed_url = f"{self.config['fed_server']['server_ip']}:{self.config['fed_server']['server_start_port'] + client_total}"
                fed_socket = fed_context.socket(zmq.REP)
                fed_socket.connect(fed_url)
                print(f"Connected on {fed_url}")

                """
                VAL SET - FOR EARLY STOPING
                """

                fed_iters = int(fed_socket.recv().decode())
                fed_socket.send(b"a")

                test_config = {
                    "cut_layer": self.config["test_cut_layer"],
                    "logits": self.config.get("logits", 10),
                }
                fed_model = arch.server(test_config).to(device)
                fed_model.load_state_dict(server_global_weights)

                correct = 0
                total = 0
                correct_per_class = torch.zeros(test_config["logits"], dtype=torch.long)
                total_per_class = torch.zeros(test_config["logits"], dtype=torch.long)
                confusion_matrix = torch.zeros(
                    test_config["logits"], test_config["logits"], dtype=torch.int64
                )

                fed_model.eval()
                with torch.no_grad():
                    for j in range(fed_iters):
                        # receive labels
                        recv_labels = fed_socket.recv()
                        numpy_labels = convert.bytes_to_array(recv_labels)
                        labels = torch.from_numpy(numpy_labels)
                        labels = labels.to(device)

                        ##dummy......
                        fed_socket.send("a".encode())

                        # get client activations
                        recv_serv_inputs = fed_socket.recv()
                        numpy_server_inputs = convert.bytes_to_array(recv_serv_inputs)
                        server_inputs = torch.from_numpy(numpy_server_inputs)
                        server_inputs = server_inputs.to(device)

                        # dummy
                        fed_socket.send("a".encode())

                        # forward pass
                        server_inputs = Variable(server_inputs, requires_grad=True)
                        outputs = fed_model(server_inputs)
                        _, predicted = torch.max(outputs.data, 1)

                        # Flatten to 1D tensors to ensure consistent indexing
                        predicted = predicted.view(-1)
                        labels = labels.view(-1)

                        correct += (predicted == labels).sum().item()
                        total += labels.size(0)

                        for class_idx in range(test_config["logits"]):
                            mask = labels == class_idx
                            total_per_class[class_idx] += mask.sum().item()
                            if mask.any():  # Only index if mask has True values
                                correct_per_class[class_idx] += (
                                    (predicted[mask] == class_idx).sum().item()
                                )

                        for t, p in zip(labels, predicted):
                            confusion_matrix[t.long(), p.long()] += 1

                accuracy = (correct / total) * 100 if total > 0 else 0
                per_class_accuracy = correct_per_class.float() / total_per_class.clamp(
                    min=1
                )

                fed_model.train()

                print("Class\tAccuracy")
                logging.info("Class\tAccuracy")
                for i, acc in enumerate(per_class_accuracy):
                    print(
                        f"{i}\t{acc*100:.4f} ({correct_per_class[i]}/{total_per_class[i]})"
                    )
                    logging.info(
                        f"{i}\t{acc*100:.4f} ({correct_per_class[i]}/{total_per_class[i]})"
                    )
                print(f"Total Accuracy on VAL set for {r}: {accuracy}")
                logging.info(f"Total Accuracy on VAL set for {r}: {accuracy}")

                print("Confusion Matrix -- Val (rows=true, cols=pred):")
                logging.info("Confusion Matrix -- Val (rows=true, cols=pred):")
                for i in range(test_config["logits"]):
                    row = " ".join(
                        f"{confusion_matrix[i, j].item():4d}"
                        for j in range(test_config["logits"])
                    )
                    print(row)
                    logging.info(row)

                if accuracy > best_accuracy:
                    print(
                        f"New best model found! New best accuracy at round {r} = {accuracy}"
                    )
                    logging.info(
                        f"New best model found! New best accuracy at round {r} = {accuracy}"
                    )
                    best_accuracy = accuracy
                    patience = 0
                    best_model = model_save_name
                else:
                    patience += 1

                fed_socket.recv()

                if patience >= self.config["patience"]:
                    self.terminate = True
                    print("Patience has run out. Ending experiment.")
                    logging.info("Patience has run out. Ending experiment.")
                    fed_socket.send(b"1")
                else:
                    fed_socket.send(b"0")

                """ 
                TEST SET - TRUE ACCURACY
                """

                fed_iters = int(fed_socket.recv().decode())
                fed_socket.send(b"a")

                test_config = {
                    "cut_layer": self.config["test_cut_layer"],
                    "logits": self.config.get("logits", 10),
                }
                fed_model = arch.server(test_config).to(device)
                fed_model.load_state_dict(server_global_weights)

                correct = 0
                total = 0
                correct_per_class = torch.zeros(test_config["logits"], dtype=torch.long)
                total_per_class = torch.zeros(test_config["logits"], dtype=torch.long)
                confusion_matrix = torch.zeros(
                    test_config["logits"], test_config["logits"], dtype=torch.int64
                )

                eval_time_start = time.perf_counter()
                fed_model.eval()
                with torch.no_grad():
                    for j in range(fed_iters):
                        # receive labels
                        recv_labels = fed_socket.recv()
                        numpy_labels = convert.bytes_to_array(recv_labels)
                        labels = torch.from_numpy(numpy_labels)
                        labels = labels.to(device)

                        ##dummy......
                        fed_socket.send("a".encode())

                        # get client activations
                        recv_serv_inputs = fed_socket.recv()
                        numpy_server_inputs = convert.bytes_to_array(recv_serv_inputs)
                        server_inputs = torch.from_numpy(numpy_server_inputs)
                        server_inputs = server_inputs.to(device)

                        # dummy
                        fed_socket.send("a".encode())

                        # forward pass
                        server_inputs = Variable(server_inputs, requires_grad=True)
                        outputs = fed_model(server_inputs)
                        _, predicted = torch.max(outputs.data, 1)

                        # Flatten to 1D tensors to ensure consistent indexing
                        predicted = predicted.view(-1)
                        labels = labels.view(-1)

                        correct += (predicted == labels).sum().item()
                        total += labels.size(0)

                        for class_idx in range(test_config["logits"]):
                            mask = labels == class_idx
                            total_per_class[class_idx] += mask.sum().item()
                            if mask.any():  # Only index if mask has True values
                                correct_per_class[class_idx] += (
                                    (predicted[mask] == class_idx).sum().item()
                                )

                        for t, p in zip(labels, predicted):
                            confusion_matrix[t.long(), p.long()] += 1

                accuracy = (correct / total) * 100 if total > 0 else 0
                per_class_accuracy = correct_per_class.float() / total_per_class.clamp(
                    min=1
                )

                fed_model.train()

                print("Class\tAccuracy")
                logging.info("Class\tAccuracy")
                for i, acc in enumerate(per_class_accuracy):
                    print(
                        f"{i}\t{acc*100:.4f} ({correct_per_class[i]}/{total_per_class[i]})"
                    )
                    logging.info(
                        f"{i}\t{acc*100:.4f} ({correct_per_class[i]}/{total_per_class[i]})"
                    )
                print(f"Total Accuracy on TEST set for {r}: {accuracy}")
                logging.info(f"Total Accuracy on TEST setfor {r}: {accuracy}")

                print("Confusion Matrix -- Test (rows=true, cols=pred):")
                logging.info("Confusion Matrix -- Test (rows=true, cols=pred):")
                for i in range(test_config["logits"]):
                    row = " ".join(
                        f"{confusion_matrix[i, j].item():4d}"
                        for j in range(test_config["logits"])
                    )
                    print(row)
                    logging.info(row)

                fed_socket.close()
                fed_context.term()


            training_end_time = time.time()
            training_time = training_end_time - training_start_time
            print("SERVER_TOTAL_TRAINING_TIME = {:.3f}" .format(training_time))
            logging.info('SERVER_TOTAL_TRAINING_TIME = {:.3f}'.format(training_time))


            print("All rounds ended..")
            
            for socket in sockets:
                socket.close()

            context.term()

        main()
