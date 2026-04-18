#######################################################
#################     FED_SERVER    ###################
#######################################################
"""
arg1 --> CONFIG_FILE_PATH
"""

# eg command: python fedServer.py 2 4444

import copy
import logging
import os
import threading
import time
from sys import getsizeof

import zmq
import torch
import yaml
from tqdm.auto import tqdm

from ..architectures import get_architecture_bundle
from ..lib import convert
from ..lib.data_prep import DataPrep
from ..lib.transformed_dataset import TransformedDataset


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
            device = (torch.device("cuda")
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
                f"./fed_server_{client_total}_{fed_port}_{rnd}.log",
            )
            # Create and configure logger
            logging.basicConfig(
                filename=log_path, format="%(asctime)s %(message)s", filemode="a"
            )
            logger = logging.getLogger()
            # Setting the threshold of logger to DEBUG
            logger.setLevel(logging.INFO)
            logging.info(
                f"Parameters (FED_SERVER_LOG) ---------- [TOTAL_CLIENTS --> {client_total}, "
                f"STARTING_SERVER_PORT --> {fed_port}, ROUNDS --> {rnd}] ----------"
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

        def socket_bind_retry(socket, url, max_retries=10, delay=5):
            retries = 0
            while retries < max_retries:
                try:
                    socket.bind(url)
                    print(f"Success bounding to {url}")
                    return True
                except zmq.ZMQError as e:
                    if e.errno == zmq.EADDRINUSE:
                        retries += 1
                        print(
                            f"Address {url} in use. Retrying... ({retries}/{max_retries})"
                        )
                        logging.info(
                            f"Address {url} in use. Retrying... ({retries}/{max_retries})"
                        )
                        time.sleep(delay)
                    else:
                        raise e
            return False

        def client_worker(sync_params, url, context, thread_no):
            """Worker routine"""

            global client_global_weights
            global client_weights
            global datasetsize_client

            lock, barrier, event = sync_params

            socket = context.socket(zmq.REP)

            if not socket_bind_retry(socket, url):
                print(f"Failed binding to {url}.")
                return

            ##*****************************************************************************************************************

            print(f"Waiting for weights from client {thread_no}")
            weights = socket.recv()
            print(f"Weights recieved from client {thread_no}")
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
            print(dataset_size)

            barrier.wait()  # ensure all threads are done

            event.wait()  # wait for server to process model

            print(
                "Size of global model weights (before) in bytes is:",
                getsizeof(client_global_weights),
            )
            logging.info(
                f"Size of global model weights (before) in bytes is: {getsizeof(client_global_weights)}"
            )
            global_bytes_weights = convert.ordered_dict_to_bytes(client_global_weights)
            print(
                "Size of global model weights (after) in bytes is:",
                getsizeof(global_bytes_weights),
            )
            logging.info(
                f"Size of global model weights (after) in bytes is: {getsizeof(global_bytes_weights)}"
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

            client_weights = []
            datasetsize_client = []

            terminate = False

            total_threads = client_total
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

                lock = threading.Lock()
                barrier = threading.Barrier(parties=total_threads + 1)
                event = threading.Event()
                sync_params = (lock, barrier, event)

                logging.info("Launching reciever threads...")
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

                print("Length of client weights:", len(client_weights))
                logging.info(f"Length of client weights: {len(client_weights)}")
                print("Length of dataset:", len(datasetsize_client))
                logging.info(f"Length of dataset: {len(datasetsize_client)}")

                # Client models weighted averaging..
                client_global_weights = average_weights(
                    client_weights, datasetsize_client
                )
                print("Global clients calculated..")
                logging.info("Global clients calculated..")

                model_save_name = os.path.join(
                    self.config.get("model_dir", "./"),
                    f"./client_fedAvg_model_r_{r}_{client_total}_{fed_port}_{rnd}.pt",
                )
                torch.save(client_global_weights, model_save_name)
                print("MODEL_SAVED.")

                event.set()

                for no, thread in enumerate(thrs):
                    thread.join()
                    logging.info(f"Thread {no} joined.")

                logging.info("All threads joined.")

                if self.config["device"] != "cpu":
                    time.sleep(
                        1
                    )  # gpu is too fast for ZMQ; race condition occurs and fed server terminates.

                print("All threads ended..")

                serv_context = zmq.Context()
                serv_url = f"tcp://*:{fed_port+client_total}"
                serv_socket = serv_context.socket(zmq.REQ)
                serv_socket.bind(serv_url)
                print(f"listening on {serv_url}")

                #VAL SET - FOR EARLY STOPPING
                eval_step(valloader, serv_socket, arch, device, self.config, "validation")

                #Early stop chekc
                serv_socket.send(b"term?")
                terminate = bool(int(serv_socket.recv().decode()))

                #Test set - True accuracy
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

    config = {"cut_layer": int(config["cut_layer"]), "logits": config['logits']}
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
