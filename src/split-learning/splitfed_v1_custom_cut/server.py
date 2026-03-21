"""
arg1 --> CONFIG_FILE_PATH
arg2..x --> cut layers for all clients
"""

import threading
import torch.nn as nn
from torchvision import models
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import time
import zmq
import torch
from ..lib import convert
from .custom_model_avg import custom_model_avg
from ..architectures import get_architecture_bundle
import yaml
import logging
import os
from datetime import datetime

# --- MiT Loss Definition ---
class MiTLoss(nn.Module):
    """
    MiT Loss: Temperature Scaling + Entropy Regularization
    """
    def __init__(self, weight=None, temperature=1.0, alpha_ent=0.1):
        super(MiTLoss, self).__init__()
        # Initialize CE with class weights (CRITICAL for Class Imbalance)
        self.ce = nn.CrossEntropyLoss(weight=weight)
        
        # FIX: Set T=1.0 by default. 
        # T > 1.0 should only be used for Post-Training Calibration, not during training.
        self.T = temperature 
        self.alpha_ent = alpha_ent

    def forward(self, logits, targets):
        #No Scaling (T=1.0) ensures gradients flow naturally
        scaled_logits = logits / self.T
        
        # Cross Entropy (Accuracy)
        ce_loss = self.ce(scaled_logits, targets)
        
        # Entropy Regularization
        # Helps the model decide on difficult boundary cases (like Melanoma vs Nevus)
        probs = F.softmax(scaled_logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
        
        return ce_loss + (self.alpha_ent * entropy)
    


class FocalLoss(nn.Module):
    """
    Focal Loss: FL(pt) = -(1-pt)^gamma * log(pt)
    Focuses training on hard/misclassified examples.
    Helps with minority class collapse (e.g. Class 3 DF in DermaMNIST).
    """
    def __init__(self, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

# class FocalLoss(nn.Module):
#     """
#     Focal Loss with optional class weights (alpha).
#     FL(pt) = -alpha_t * (1-pt)^gamma * log(pt)
#     Combines per-class weighting with hard-example focusing.
#     """
#     def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha  # class weights tensor
#         self.gamma = gamma
#         self.reduction = reduction

#     def forward(self, inputs, targets):
#         ce_loss = F.cross_entropy(inputs, targets,
#                                   weight=self.alpha,
#                                   reduction='none')
#         pt = torch.exp(-ce_loss)
#         focal_loss = (1 - pt) ** self.gamma * ce_loss
#         if self.reduction == 'mean':
#             return focal_loss.mean()
#         return focal_loss.sum()
# --- Server Runner ---

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
        client_total = self.config["client_total"]
        split_port = self.config["split_server"]["server_start_port"]
        device = self.config["device"]
        epochs = self.config["epoch"]
        rnd = self.config["round"]
        muted = bool(self.config.get("muted", False))
        print(f"muted set to: {muted}")
        test_last_model = self.config.get("test_last_model", False)

        # Load architecture
        model_architecture = self.config.get("model_architecture", "ResNet18_CIFAR10")
        arch = get_architecture_bundle(model_architecture)
        logits = self.config.get("logits", 10)
        tau_init = self.config.get("tau_init", 20.0)
        mu = float(self.config.get("fedprox_mu", 0.0))

        if device != "cpu":
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        print(device)

        # Build run folder (used for both logs and model weights)
        loss_fn = self.config.get("loss_function", "CE")
        dataset = os.path.splitext(self.config["data_server"]["output_file"])[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        run_tag = f"{model_architecture}_{loss_fn}_{dataset}_{timestamp}"
        log_dir = self.config.get("log_dir", "./logs")
        run_dir = os.path.join(log_dir, run_tag)
        os.makedirs(run_dir, exist_ok=True)

        # Write run_id so fed_server and clients resolve the same folder
        run_id_path = os.path.join(log_dir, ".run_id")
        with open(run_id_path, "w") as f:
            f.write(run_tag)

        # Initialize Logger
        if self.config["logging"]:

            new_log_path = os.path.join(run_dir, "sf_server.log")
            old_log_path = os.path.join(
                log_dir,
                f"sf_server_{client_total}_{split_port}_{device}_"
                f"{self.server_cut_layer_list}_{epochs}_{rnd}.log",
            )
            formatter = logging.Formatter("%(asctime)s %(message)s")
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            for path, mode in [(new_log_path, "w"), (old_log_path, "a")]:
                h = logging.FileHandler(path, mode=mode)
                h.setFormatter(formatter)
                logger.addHandler(h)
            logging.info(
                f"Parameters (SF_SERVER_LOG) ---------- "
                f"[MODEL --> {model_architecture}, LOSS --> {loss_fn}, DATASET --> {dataset}, "
                f"TOTAL_CLIENTS --> {client_total}, STARTING_SERVER_PORT --> {split_port}, "
                f"DEVICE_TYPE --> {device}, "
                f"EPOCHS --> {epochs}, ROUNDS --> {rnd}] ----------"
            )

        def worker_routine(url, context, thread_no, r, cut_layer):
            """Worker routine"""

            global server_global_weights
            global server_weights
            global datasetsize_server

            socket = context.socket(zmq.REP)
            socket.setsockopt(zmq.LINGER, 0)
            socket.bind(url)

            model_config = {"cut_layer": cut_layer, "logits": logits, "tau_init": tau_init}
            server_model = arch.server(model_config).to(device)

            if loss_fn == "CE":
                criterion = nn.CrossEntropyLoss()
            elif loss_fn.startswith("FocalLoss_g"):
                gamma = float(loss_fn.split("g")[1])
                criterion = FocalLoss(gamma=gamma)
            else:
                criterion = nn.CrossEntropyLoss()

            # DR loss: hook to capture the pre-ETF features (input to ETFClassifier)
            # These are the L2-normalised-ready 512-dim features, used to compute
            # ||normalize(h) - M_y||^2 = 2 - 2·cosine_sim(h, M_y) directly.
            _dr_etf_clf = None
            _hook_handle = None
            _pre_etf = [None]
            if loss_fn == "DR":
                from ..architectures.models.etf_classifier import ETFClassifier
                _dr_etf_clf = next(
                    (m for m in server_model.modules() if isinstance(m, ETFClassifier)), None
                )
                if _dr_etf_clf is None:
                    raise ValueError("loss_function='DR' requires an ETFClassifier in the server model")
                def _capture(module, inp, out):
                    _pre_etf[0] = inp[0]
                _hook_handle = _dr_etf_clf.register_forward_hook(_capture)

            # current_lr = 0.01 if r < 15 else 0.001
            server_optimizer = optim.SGD(
                server_model.parameters(),
                lr=0.01,
                momentum=0.9,
            )

            # load aggregated server weights
            w0_server = None
            if r > 0 or (r == 0 and server_global_weights is not None):
                if server_global_weights is not None:
                    server_model.load_state_dict(server_global_weights)
                    print(f"GLOBAL_SERVER_WEIGHTS_LOADED (Round {r})")
                    if mu > 0:
                        w0_server = {n: p.data.clone() for n, p in server_model.named_parameters()}

            socket.recv()
            if self.terminate:
                socket.send(b"1")
                if test_last_model:
                    test_client(device, server_model, socket, thread_no)
                socket.close()
                return
            else:
                socket.send(b"0")

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

            round_start_time = time.perf_counter()
            for epoch in range(num_epochs):
                epoch_start_time = time.perf_counter()

                for j in range(recv_iterations):
                    step_start_time = time.perf_counter()
                    if not muted:
                        print(f"***TH - {thread_no}*** {epoch} {j}")

                    # receive labels
                    recv_labels = socket.recv()
                    numpy_labels = convert.bytes_to_array(recv_labels)
                    labels = torch.from_numpy(numpy_labels)
                    labels = labels.to(device)

                    ##dummy......
                    socket.send(send_msg)

                    # get client activations
                    recv_serv_inputs = socket.recv()
                    numpy_server_inputs = convert.bytes_to_array(recv_serv_inputs)
                    server_inputs = torch.from_numpy(numpy_server_inputs)
                    server_inputs = server_inputs.to(device)

                    # forward pass
                    server_inputs = Variable(server_inputs, requires_grad=True)
                    outputs = server_model(server_inputs)

                    server_optimizer.zero_grad()
                    if loss_fn == "DR" and _pre_etf[0] is not None:
                        # Dot-regression loss: 2 - 2·cosine_sim(h, M_y)
                        # _pre_etf[0] is the (B, feat_in) input to ETFClassifier captured by hook
                        x_norm = F.normalize(_pre_etf[0], dim=1)
                        M_y = _dr_etf_clf.etf_vec[labels.long().view(-1)]
                        loss = (2 - 2 * (x_norm * M_y).sum(dim=1)).mean()
                    else:
                        # Squeeze labels for MedMNIST
                        loss = criterion(outputs, labels.long().squeeze())
                        if mu > 0 and w0_server is not None:
                            for n, p in server_model.named_parameters():
                                loss = loss + (mu / 2.0) * torch.norm(p - w0_server[n]) ** 2
                    loss.backward()

                    # Send gradients back to client.
                    transfer_loss = server_inputs.grad.clone().detach()
                    server_optimizer.step()

                    bytes_loss = convert.array_to_bytes(transfer_loss.cpu())
                    socket.send(bytes_loss)

                    # telemetry
                    step_end_time = time.perf_counter()
                    total_one_step_time = step_end_time - step_start_time
                    if not muted:
                        print(
                            f"***TH - {thread_no}*** SERVER_TOTAL_ONE_STEP_TIME = {total_one_step_time:.3f}, loss: {loss.item():.3f}"
                        )
                    logging.info(
                        f"***TH - {thread_no}*** SERVER_TOTAL_ONE_STEP_TIME = {total_one_step_time:.3f}, loss: {loss.item():.3f}"
                    )

                epoch_end_time = time.perf_counter()
                total_one_epoch_time = epoch_end_time - epoch_start_time
                print(
                    f"***TH - {thread_no}*** SERVER_TOTAL_ONE_EPOCH_TIME = {total_one_epoch_time:.3f}"
                )
                logging.info(
                    f"***TH - {thread_no}*** SERVER_TOTAL_ONE_EPOCH_TIME = {total_one_epoch_time:.3f}"
                )

            if (test_last_model and r == rnd - 1):
                test_client(device, server_model, socket, thread_no)
         
            round_end_time = time.perf_counter()
            round_time = round_end_time - round_start_time
            print(
                f"***TH - {thread_no}*** SERVER_ROUND_TRAINING_TIME = {round_time:.3f}"
            )
            logging.info(
                f"***TH - {thread_no}*** SERVER_ROUND_TRAINING_TIME = {round_time:.3f}"
            )

            if _hook_handle is not None:
                _hook_handle.remove()

            server_weights.append(server_model.state_dict())
            datasetsize_server.append(dataset_size)

            # model_save_name = os.path.join(
            #     self.config.get("model_dir", "./"),
            #     f"server_thread_model_r{r}_{thread_no}_{client_total}_"
            #     f"{split_port}_{device}_{cut_layer}_{epochs}.pt",
            # )
            # torch.save(server_model.state_dict(), model_save_name)
            print("***TH - {}*** MODEL_SAVED.".format(thread_no))

            print("Worker done******************************")
            socket.close()

        def main():
            """server routine"""

            global server_global_weights
            global server_weights
            global datasetsize_server

            server_weights = []
            datasetsize_server = []

            total_threads = client_total
            port_no = split_port
            connection_url = [
                "tcp://*:" + str(port_no + i) for i in range(total_threads)
            ]

            num_rounds = rnd
            context = zmq.Context()
            
            best_accuracy = 0
            best_val_round = -1
            best_model = ""
            patience = 0
            best_test_accuracy = 0
            best_test_round = -1
            best_val_per_class = None
            best_val_correct = None
            best_val_total = None
            best_val_cm = None
            best_test_per_class = None
            best_test_correct = None
            best_test_total = None
            best_test_cm = None

            server_global_weights = None

            training_start_time = time.perf_counter()
            for r in range(num_rounds):
                print(f"New round {r} started..")
                thrs = []

                server_weights.clear()
                datasetsize_server.clear()

                # Launch pool of worker threads
                for i in range(total_threads):
                    thread = threading.Thread(
                        target=worker_routine,
                        args=(
                            connection_url[i],
                            context,
                            i + 1,
                            r,
                            self.server_cut_layer_list[i],
                        ),
                    )
                    thrs.append(thread)
                    thread.start()

                for thread in thrs:
                    thread.join()

                if self.terminate:
                    break

                # Server models weighted averaging..
                server_global_weights, _ = custom_model_avg(
                    True,
                    server_weights,
                    datasetsize_server,
                    self.server_cut_layer_list,
                    arch.base,
                    {"logits": logits},
                )

                model_save_name = os.path.join(run_dir, f"server_model_r{r}.pt")

                torch.save(server_global_weights, model_save_name)
                print("MODEL_SAVED.")
                print("All threads ended..")

                fed_context = zmq.Context()
                fed_url = f"{self.config['fed_server']['server_ip']}:{self.config['fed_server']['server_start_port'] + client_total}"
                fed_socket = fed_context.socket(zmq.REP)
                fed_socket.setsockopt(zmq.LINGER, 0)
                fed_socket.connect(fed_url)
                print(f"Connected on {fed_url}")

                # =============================================================
                #  VALIDATION STEP
                # =============================================================
                fed_iters = int(fed_socket.recv().decode())
                fed_socket.send(b"a")

                test_config = {
                    "cut_layer": self.config["test_cut_layer"],
                    "logits": logits,
                    "tau_init": tau_init,
                }
                fed_model = arch.server(test_config).to(device)
                fed_model.load_state_dict(server_global_weights)

                correct = 0
                total = 0
                
                # Setup metrics containers
                correct_per_class = torch.zeros(test_config["logits"], dtype=torch.long)
                total_per_class = torch.zeros(test_config["logits"], dtype=torch.long)
                confusion_matrix = torch.zeros(
                    test_config["logits"], test_config["logits"], dtype=torch.int64
                )
                
                # Validation Loop
                fed_model.eval()
                with torch.no_grad():
                    for j in range(fed_iters):
                        recv_labels = fed_socket.recv()
                        numpy_labels = convert.bytes_to_array(recv_labels)
                        labels = torch.from_numpy(numpy_labels).to(device)
                        fed_socket.send("a".encode())

                        recv_serv_inputs = fed_socket.recv()
                        numpy_server_inputs = convert.bytes_to_array(recv_serv_inputs)
                        server_inputs = torch.from_numpy(numpy_server_inputs).to(device)
                        fed_socket.send("a".encode())

                        outputs = fed_model(server_inputs)
                        _, predicted = torch.max(outputs.data, 1)

                        predicted = predicted.view(-1)
                        labels = labels.view(-1)

                        correct += (predicted == labels).sum().item()
                        total += labels.size(0)
                        
                        # Update Per-Class Metrics
                        for class_idx in range(test_config["logits"]):
                            mask = labels == class_idx
                            total_per_class[class_idx] += mask.sum().item()
                            if mask.any():
                                correct_per_class[class_idx] += (
                                    (predicted[mask] == class_idx).sum().item()
                                )

                        # Update Confusion Matrix
                        for t, p in zip(labels, predicted):
                            confusion_matrix[t.long(), p.long()] += 1

                accuracy = (correct / total) * 100 if total > 0 else 0
                per_class_accuracy = correct_per_class.float() / total_per_class.clamp(min=1)
                
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
                    print(f"New best model found! New best accuracy at round {r} = {accuracy}")
                    logging.info(f"New best model found! New best accuracy at round {r} = {accuracy}")
                    best_accuracy = accuracy
                    best_val_round = r
                    patience = 0
                    best_model = model_save_name
                    best_val_per_class = per_class_accuracy.clone()
                    best_val_correct = correct_per_class.clone()
                    best_val_total = total_per_class.clone()
                    best_val_cm = confusion_matrix.clone()
                else:
                    patience += 1

                fed_socket.recv() # Wait for client sync

                if patience >= self.config["patience"]:
                    self.terminate = True
                    print("Patience has run out. Ending experiment.")
                    logging.info("Patience has run out. Ending experiment.")
                    fed_socket.send(b"1")
                else:
                    fed_socket.send(b"0")

                # =============================================================
                # TEST SET
                # =============================================================
                fed_iters = int(fed_socket.recv().decode())
                fed_socket.send(b"a")
                
                # Reset metrics for Test
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
                        recv_labels = fed_socket.recv()
                        numpy_labels = convert.bytes_to_array(recv_labels)
                        labels = torch.from_numpy(numpy_labels).to(device)
                        fed_socket.send("a".encode())

                        recv_serv_inputs = fed_socket.recv()
                        numpy_server_inputs = convert.bytes_to_array(recv_serv_inputs)
                        server_inputs = torch.from_numpy(numpy_server_inputs).to(device)
                        fed_socket.send("a".encode())
                        
                        outputs = fed_model(server_inputs)
                        _, predicted = torch.max(outputs.data, 1)

                        predicted = predicted.view(-1)
                        labels = labels.view(-1)

                        correct += (predicted == labels).sum().item()
                        total += labels.size(0)
                        
                        # Update Per-Class Metrics
                        for class_idx in range(test_config["logits"]):
                            mask = labels == class_idx
                            total_per_class[class_idx] += mask.sum().item()
                            if mask.any():
                                correct_per_class[class_idx] += (
                                    (predicted[mask] == class_idx).sum().item()
                                )

                        # Update Confusion Matrix
                        for t, p in zip(labels, predicted):
                            confusion_matrix[t.long(), p.long()] += 1
                
                accuracy = (correct / total) * 100 if total > 0 else 0
                per_class_accuracy = correct_per_class.float() / total_per_class.clamp(min=1)
                
                fed_model.train()
                
                # --- PRINTING TEST LOGS ---
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
                logging.info(f"Total Accuracy on TEST set for {r}: {accuracy}")

                if accuracy > best_test_accuracy:
                    best_test_accuracy = accuracy
                    best_test_round = r
                    print(f"New best TEST result at round {r} = {accuracy}")
                    logging.info(f"New best TEST result at round {r} = {accuracy}")
                    best_test_per_class = per_class_accuracy.clone()
                    best_test_correct = correct_per_class.clone()
                    best_test_total = total_per_class.clone()
                    best_test_cm = confusion_matrix.clone()

                print("Confusion Matrix -- Test (rows=true, cols=pred):")
                logging.info("Confusion Matrix -- Test (rows=true, cols=pred):")
                for i in range(test_config["logits"]):
                    row = " ".join(
                        f"{confusion_matrix[i, j].item():4d}"
                        for j in range(test_config["logits"])
                    )
                    print(row)
                    logging.info(row)
                
                # Close socket and allow time for ports to recycle
                fed_socket.close()
                fed_context.term()
                print(f"Round {r} finished. Sleeping for 2 seconds to release ports...")
                time.sleep(2) # Safety pause to prevent ZMQ Address in Use errors

            training_end_time = time.perf_counter()
            training_time = training_end_time - training_start_time
            print(f"SERVER_TOTAL_TRAINING_TIME = {training_time:.3f}")
            print(f"BEST_VAL: round {best_val_round}, accuracy = {best_accuracy}")
            print(f"BEST_TEST: round {best_test_round}, accuracy = {best_test_accuracy}")
            logging.info(f"BEST_VAL: round {best_val_round}, accuracy = {best_accuracy}")
            logging.info(f"BEST_TEST: round {best_test_round}, accuracy = {best_test_accuracy}")

            def log_per_class(label, rnd_no, pc, correct, total, cm):
                header = f"=== {label} (Round {rnd_no}) Per-Class ==="
                print(header); logging.info(header)
                for i, acc in enumerate(pc):
                    line = f"{i}\t{acc*100:.4f} ({correct[i]}/{total[i]})"
                    print(line); logging.info(line)
                cm_header = f"Confusion Matrix -- {label} (rows=true, cols=pred):"
                print(cm_header); logging.info(cm_header)
                for i in range(len(pc)):
                    row = " ".join(f"{cm[i, j].item():4d}" for j in range(len(pc)))
                    print(row); logging.info(row)

            if best_val_per_class is not None:
                log_per_class("BEST VAL", best_val_round,
                              best_val_per_class, best_val_correct,
                              best_val_total, best_val_cm)
            if best_test_per_class is not None:
                log_per_class("BEST TEST", best_test_round,
                              best_test_per_class, best_test_correct,
                              best_test_total, best_test_cm)

            context.term()

        main()

def test_client(device, server_model, socket, thread_no):
    server_model.eval()

    correct = 0
    total = 0

    test_iters = int(socket.recv().decode())
    socket.send(b"a")

    with torch.no_grad():
        for j in range(test_iters):
            #receive labels
            recv_labels = socket.recv()
            numpy_labels = convert.bytes_to_array(recv_labels)
            labels = torch.from_numpy(numpy_labels)
            labels = labels.to(device)

            ##dummy......
            socket.send(b"a")

            #get client activations
            recv_serv_inputs = socket.recv()
            numpy_server_inputs = convert.bytes_to_array(recv_serv_inputs)
            server_inputs = torch.from_numpy(numpy_server_inputs)
            server_inputs = server_inputs.to(device)

            #dummy
            socket.send(b"a")

            #forward pass
            server_inputs = Variable(server_inputs, requires_grad=True)
            outputs = server_model(server_inputs)

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = 100 * correct / total if total > 0 else 0

        socket.recv()
        socket.send(str(accuracy).encode())
        print(f" ***TH - {thread_no}*** Accuracy on test set: {accuracy}%")
        logging.info(f"***TH - {thread_no}*** Accuracy on test set: {accuracy}%")
        server_model.train()