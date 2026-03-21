#######################################################
#################     FED_SERVER    ###################
#######################################################
"""
arg1 --> CONFIG_FILE_PATH
"""

import copy
import threading
import time
import zmq
import torch
from ..lib import convert
from ..architectures import get_architecture_bundle
import yaml
from sys import getsizeof
from .custom_model_avg import custom_model_avg, combine_fed_avg_models
import logging
import os
from datetime import datetime
import urllib.request
import pickle
from torchvision import transforms, models
import torchvision.transforms as transforms
import torch.nn as nn
from tqdm.auto import tqdm

# --- ADD THIS CLASS TO fed_server.py ---
# --- ADD THIS TO fed_server.py (After imports) ---
class ShadesOfGray(object):
    """
    Implements Shades of Gray color constancy (Minkowski Norm p=6).
    """
    def __init__(self, power=6):
        self.power = power

    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            t_img = transforms.functional.to_tensor(img)
        else:
            t_img = img
            
        c, h, w = t_img.shape
        img_flat = t_img.view(c, -1)
        illum = img_flat.pow(self.power).mean(dim=1).pow(1.0/self.power)
        illum = illum.view(c, 1, 1)
        normalized = t_img / (illum + 1e-8)
        normalized = torch.clamp(normalized, 0, 1)
        return normalized
    
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
        with open(config_path, "r") as yamlfile:
            self.config = yaml.load(yamlfile, Loader=yaml.FullLoader)
            print("Read successful")

    def run(self):
        client_total = self.config["client_total"]
        fed_port = self.config["fed_server"]["server_start_port"]
        rnd = self.config["round"]

        ##################################################################
        # Code to enable ad-hoc testing
        if self.config["device"] == "cpu":
            device = "cpu"
        else:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        output_file = self.config["data_server"]["output_file"]
        val_file = output_file.replace(".pkl", "_val.pkl")
        val_file_tmp = f"tmp_fed_{val_file}"

        urllib.request.urlretrieve(
            f"{self.config['data_server']['server_address']}/{val_file}", val_file_tmp
        )

        with open(val_file_tmp, "rb") as handle:
            valset = pickle.load(handle)

        output_file = self.config["data_server"]["output_file"]
        test_file = output_file.replace(".pkl", "_test.pkl")
        test_file_tmp = f"tmp_fed_{test_file}"
        cut_layer = self.config["test_cut_layer"]

        urllib.request.urlretrieve(
            f"{self.config['data_server']['server_address']}/{test_file}", test_file_tmp
        )

        with open(test_file_tmp, "rb") as handle:
            testset = pickle.load(handle)

        # Load architecture
        model_architecture = self.config.get("model_architecture", "ResNet18_CIFAR10")
        arch = get_architecture_bundle(model_architecture)
        logits = self.config.get("logits", 10)

        transformer = arch.eval_transformer

        valset = TransformedDataset(valset, transform=transformer)
        testset = TransformedDataset(testset, transform=transformer)

        valloader = torch.utils.data.DataLoader(
            valset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=0,
            persistent_workers=False,
        )

        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=0,
            persistent_workers=False,
        )

        ##################################################################

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
            new_log_path = os.path.join(run_dir, "cc_fed_server.log")
            old_log_path = os.path.join(
                log_dir, f"cc_fed_server_{client_total}_{fed_port}_{rnd}.log"
            )
            formatter = logging.Formatter("%(asctime)s %(message)s")
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            for path, mode in [(new_log_path, "w"), (old_log_path, "a")]:
                h = logging.FileHandler(path, mode=mode)
                h.setFormatter(formatter)
                logger.addHandler(h)
            logging.info(
                "Parameters (FED_SERVER_LOG) ---------- [TOTAL_CLIENTS --> {}, STARTING_SERVER_PORT --> {}, ROUNDS --> {}] ---------- ".format(
                    str(client_total), str(fed_port), str(rnd)
                )
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
                for i in range(1, len(w)):  ## IMP
                    w_avg[key] += w[i][key]
                w_avg[key] = torch.div(w_avg[key], float(sum(datasize)))

            return w_avg

        def client_worker(sync_params, url, context, thread_no):
            global client_global_weights
            global client_weights
            global datasetsize_client
            global client_cut_layer_list

            lock, barrier, event = sync_params

            socket = context.socket(zmq.REP)
            socket.setsockopt(zmq.LINGER, 10000)  # wait up to 10s for global model delivery before closing
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
                    {"logits": logits},
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
                serv_socket.setsockopt(zmq.LINGER, 0)
                serv_socket.bind(serv_url)
                print(f"listening on {serv_url}")

                """
                VAL SET - FOR EARLY STOPPING
                """

                val_iters = len(valloader)
                send_val_iters = str(val_iters).encode()
                serv_socket.send(send_val_iters)
                serv_socket.recv()

                test_config = {
                    "cut_layer": int(self.config["test_cut_layer"]),
                    "logits": logits,
                }
                test_model = arch.client(test_config).to(device)
                test_model.load_state_dict(client_global_weights)

                bar = tqdm(
                    valloader,
                    desc=f"valset: ",
                    unit="",
                    ascii=True,
                    bar_format="{desc} {n_fmt}/{total_fmt} {percentage:3.0f}%|{bar}| {postfix}",
                )

                test_model.eval()

                with torch.no_grad():
                    for data in bar:
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

                serv_socket.send(b"term?")
                terminate = bool(int(serv_socket.recv().decode()))

                """
                TEST SET - TRUE ACCURACY
                """

                # send dataset length
                test_iters = len(testloader)
                send_test_iters = str(test_iters).encode()
                serv_socket.send(send_test_iters)
                serv_socket.recv()

                test_config = {
                    "cut_layer": int(self.config["test_cut_layer"]),
                    "logits": logits,
                }
                test_model = arch.client(test_config).to(device)
                test_model.load_state_dict(client_global_weights)

                bar = tqdm(
                    testloader,
                    desc=f"testset: ",
                    unit="",
                    ascii=True,
                    bar_format="{desc} {n_fmt}/{total_fmt} {percentage:3.0f}%|{bar}| {postfix}",
                )

                with torch.no_grad():
                    for data in bar:
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

                test_model.train()

                serv_socket.close()
                serv_context.term()

            print("socket closed")

            print("All rounds ended..")
            logging.info("All rounds ended..")

            # exposure / final aggregation legacy code.

            # bytes_weights = serv_socket.recv()
            # serv_socket.send("a".encode())
            # serv_weights = convert.bytes_to_dict(bytes_weights)
            # print("got weights")

            # bytes_exposure = serv_socket.recv()
            # serv_socket.send("a".encode())
            # serv_exposure = convert.bytes_to_dict(bytes_exposure)
            # print("got exposure")

            # final_model_weights = combine_fed_avg_models(client_global_weights, client_exposure, serv_weights, serv_exposure)
            # model_save_name = f"./final_aggregate_model.pt"
            # model_save_name = os.path.join(self.config.get("model_dir", "./"), "final_aggregate_model.pt")
            # torch.save(final_model_weights, model_save_name)

            context.term()

        main()
