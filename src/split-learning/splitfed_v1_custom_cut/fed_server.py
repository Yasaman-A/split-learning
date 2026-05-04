#######################################################
#################     FED_SERVER    ###################
#######################################################
"""
arg1 --> CONFIG_FILE_PATH
"""
import logging
import os
import threading
import time
from sys import getsizeof

import torch
import yaml
import zmq
from tqdm.auto import tqdm

from .custom_model_avg import custom_model_avg
from ..lib import convert
from ..lib.data_prep import DataPrep
from ..architectures import get_architecture_bundle


class Runner:
    def __init__(self, config_path) -> None:
        with open(config_path, "r") as yamlfile:
            self.config = yaml.load(yamlfile, Loader=yaml.FullLoader)
            print("Read successful")

    def run(self):
        client_total = self.config["client_total"]
        fed_port = self.config["fed_server"]["server_start_port"]
        rnd = self.config["round"]

        if self.config["device"] == "cpu":
            device = "cpu"
        else:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        model_architecture = self.config.get("model_architecture")
        if model_architecture is None:
            raise ValueError("Error: No model architecture specified")
        arch = get_architecture_bundle(model_architecture)

        data_prepper = DataPrep(self.config, arch)
        valloader = data_prepper.get_eval_loader("validation")
        testloader = data_prepper.get_eval_loader("testing")


        if self.config["logging"]:
            log_path = os.path.join(
                self.config.get("log_dir", "./"),
                f"./cc_fed_server_{client_total}_{fed_port}_{rnd}.log",
            )
            # Create and configure logger
            logging.basicConfig(
                filename=log_path, format="%(asctime)s %(message)s", filemode="a"
            )
            # Creating an object
            logger = logging.getLogger()
            # Setting the threshold of logger to DEBUG
            logger.setLevel(logging.INFO)
            logging.info(
                "Parameters (FED_SERVER_LOG) ---------- [TOTAL_CLIENTS --> {}, STARTING_SERVER_PORT --> {}, ROUNDS --> {}] ---------- ".format(
                    str(client_total), str(fed_port), str(rnd)
                )
            )

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
            with lock:
                client_weights.append(numpy_weights)

            msg = "weights_recv"
            send_msg = msg.encode()
            socket.send(send_msg)

            recv_dataset_size = socket.recv()
            dataset_size = int(recv_dataset_size.decode())
            with lock:
                datasetsize_client.append(dataset_size)

            msg = "dataset_size_recv"
            send_msg = msg.encode()
            socket.send(send_msg)

            recv_cut_layer = socket.recv()
            cut_layer = int(recv_cut_layer.decode())

            with lock:
                client_cut_layer_list.append(cut_layer)

            barrier.wait()  # ensure all threads are done

            event.wait()  # wait for server to process model

            print(
                f"Size of global model weights (before) in bytes is: {getsizeof(client_global_weights)}"
            )
            logging.info(
                "Size of global model weights (before) in bytes is: %s",
                getsizeof(client_global_weights),
            )

            global_bytes_weights = convert.ordered_dict_to_bytes(client_global_weights)

            print(
                f"Size of global model weights (after) in bytes is: {getsizeof(global_bytes_weights)}"
            )
            logging.info(
                "Size of Size of global model weights (after) in bytes is: %s",
                getsizeof(global_bytes_weights),
            )

            socket.send(global_bytes_weights)
            print(f"Weights sent to client {thread_no}")
            logging.info(f"Weights sent to client {thread_no}")

            ##*****************************************************************************************************************
            print(f"Worker {thread_no} done******************************")
            logging.info(f"Worker {thread_no} done******************************")

            socket.close()

        def main():
            """server routine"""

            global client_global_weights
            global client_weights
            global datasetsize_client
            global client_cut_layer_list

            client_weights = []
            datasetsize_client = []
            client_cut_layer_list = []

            # client_exposure = []

            terminate = False

            total_threads = int(client_total)
            port_no = int(fed_port)
            connection_url = [
                "tcp://*:" + str(fed_port + i) for i in range(client_total)
            ]

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
                for i in range(
                    total_threads
                ):  # this defines how many clients can connect
                    thread = threading.Thread(
                        target=client_worker,
                        args=(sync_params, connection_url[i], context, i + 1),
                    )
                    thrs.append(thread)
                    thread.start()

                barrier.wait()

                print("Length of client weights:  ", len(client_weights))
                print("Length of dataset: ", len(datasetsize_client))
                print("Length of cut_layers: ", len(client_cut_layer_list))

                client_global_weights, client_exposure = custom_model_avg(
                    False,
                    client_weights,
                    datasetsize_client,
                    client_cut_layer_list,
                    arch.base,
                    {"logits": self.config['logits']},
                )

                print("Global clients calculated..")
                logging.info("Global clients calculated..")

                model_save_name = os.path.join(
                    self.config.get("model_dir", "./"),
                    f"client_fedAvg_model_r{r}_{client_total}_{fed_port}_{rnd}.pt",
                )
                torch.save(client_global_weights, model_save_name)
                print("MODEL_SAVED.")
                logging.info("MODEL SAVED.")

                event.set()  # workers will send data back to clients

                for no, thread in enumerate(
                    thrs
                ):  # have to check when it will run all epochs..
                    thread.join()
                    logging.info(f"Thread {no} joined.")

                logging.info("All threads joined.")

                if self.config["device"] != "cpu":
                    time.sleep(
                        1
                    )  # gpu is too fast for ZMQ; race condition occurs and fed server terminates.

                print("All threads ended..")

                # get accuracy of aggregated models
                serv_context = zmq.Context()
                serv_url = f"tcp://*:{fed_port+client_total}"
                serv_socket = serv_context.socket(zmq.REQ)
                serv_socket.bind(serv_url)
                print(f"listening on {serv_url}")

                #VAL SET - FOR EARLY STOPPING

                eval_step(valloader, serv_socket, arch, device, self.config, "validation")

                serv_socket.send(b"term?")
                terminate = bool(int(serv_socket.recv().decode()))

                #TEST SET - TRUE ACCURACY

                eval_step(testloader, serv_socket, arch, device, self.config, "testing")

                serv_socket.close()
                serv_context.term()

            print("socket closed")

            print("All rounds ended..")
            logging.info("All rounds ended..")

            context.term()

        main()

def eval_step(loader, serv_socket, arch, device, config, eval_type):
    """ Does a pass of the eval data over a sample aggregated model."""
    iters = len(loader)
    send_iters = str(iters).encode()
    serv_socket.send(send_iters)
    serv_socket.recv()

    config = {"cut_layer": int(config["test_cut_layer"]), "logits": config['logits']}
    test_model = arch.client(config).to(device)
    test_model.load_state_dict(client_global_weights)

    progress_bar = tqdm(
        loader,
        desc=f"{eval_type}: ",
        unit="",
        ascii=True,
        bar_format="{desc} {n_fmt}/{total_fmt} {percentage:3.0f}%|{bar}| {postfix}",
    )

    test_model.eval()

    with torch.no_grad():
        for data in progress_bar:
            inputs, labels = data[0].to(device), data[1].to(device)

            # send labels
            bytes_labels = convert.array_to_bytes(labels.cpu())
            serv_socket.send(bytes_labels)
            serv_socket.recv()

            # send activations
            activations = test_model(inputs)
            server_inputs = activations.detach().clone()
            bytes_server_inputs = convert.array_to_bytes(
                server_inputs.cpu()
            )

            serv_socket.send(bytes_server_inputs)
            serv_socket.recv()
