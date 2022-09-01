
"""
arg1 --> SERVER_IP
arg2 --> SERVER_PORT
arg3 --> 'cpu' or 'gpu'
arg4 --> client number/thread no
arg5 --> cut_layer
arg6 --> epoch
arg7 --> split_parts, if on split type 'n' or '1'
arg8 --> split_no, starts from 0
arg9 --> batch_size
arg10 --> round
arg11 --> FED_SERVER_IP
arg12 --> FED_SERVER_PORT
"""

# eg command: python client_splitnn.py localhost 5555 cpu 0 3 10 1 0 128
# eg command: python client_splitnn.py localhost 5556 cpu 1 3 10

import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
from torch.autograd import Variable
import time
import zmq
import torch
from convert import array_to_bytes, bytes_to_array, ordered_dict_to_bytes, bytes_to_dict
import sys
from sys import getsizeof
import numpy as np

import logging
# from objsize import get_deep_size

# Create and configure logger
logging.basicConfig(filename="./client_thread_" + sys.argv[4] + "_" + sys.argv[2] + "_" + sys.argv[3] + "_" + sys.argv[5] + "_" + sys.argv[6] + "_" + sys.argv[7] + "_" + sys.argv[8] + "_" + sys.argv[9] + "_" + sys.argv[10] + "_" + sys.argv[12] + ".log",
                    format='%(asctime)s %(message)s',
                    filemode='a')

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.INFO)


if __name__ == '__main__':

    logging.info('Parameters (SF_CLIENT_LOG) ---------- [SERVER_PORT --> {}, DEVICE_TYPE --> {}, CLIENT_NO --> {}, CUT_LAYER --> {}, EPOCHS --> {}, SPLIT_PARTS --> {}, PART_NO --> {}, BATCH_SIZE --> {}, ROUNDS --> {}, FED_SERVER_PORT --> {}] ---------- '.format(
        sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9], sys.argv[10], sys.argv[12]))

    # ***

    if(sys.argv[3] == 'cpu'):
        device = 'cpu'
    else:
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # CIFAR10 is a dataset of natural images consisting of 50k training images and 10k test
    # Every image is labelled with one of the following class
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    batch_size = int(sys.argv[9])

    ## Dataloader Splitting....
    if (sys.argv[7] == 'n'):
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)
        datasetsize_used = len(trainset)

    else:
        dataset_size = len(trainset)                         # 50k images
        total_indices = list(range(dataset_size))
        list_of_indices = np.array_split(
            np.array(total_indices), int(sys.argv[7]))
        [l.tolist() for l in list_of_indices]

        use_indices = list_of_indices[int(sys.argv[8])]
        datasetsize_used = len(use_indices)
        # print(use_indices)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  num_workers=2, sampler=use_indices)  # shuffle=True (mutually exclusive with sampler)
    # print(len(trainloader))
    # exit()

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    # Explain nn.Module and explain the forward and backward pass

    class ResNet18Client(nn.Module):
        """docstring for ResNet"""

        # Explain initialize (listing the neural network architecture and other related parameters)
        def __init__(self, config):
            super(ResNet18Client, self).__init__()
            # Explain this line
            self.cut_layer = config["cut_layer"]

            # Explain this line
            self.model = models.resnet18(pretrained=True)

            self.model = nn.ModuleList(self.model.children())
            self.model = nn.Sequential(*self.model)

        # Explain forward (actually used during the execution of the neural network at runtime)
        def forward(self, x):
            for i, l in enumerate(self.model):
                if i > self.cut_layer:
                    break
                x = l(x)
            return x

    config = {"cut_layer": int(sys.argv[5]), "logits": 10}
    client_model = ResNet18Client(config).to(device)

    criterion = nn.CrossEntropyLoss()
    client_optimizer = optim.SGD(
        client_model.parameters(), lr=0.01, momentum=0.9)



    num_rounds = int(sys.argv[10])
    for r in range(num_rounds):

        if r > 0:
            client_model.load_state_dict(global_numpy_weights)
            print("GLOBAL_CLIENT_WEIGHTS_LOADED")
            del global_numpy_weights



        log_steps = 50
        num_epochs = int(sys.argv[6])

        training_start_time = time.time()
        for epoch in range(num_epochs):


            context = zmq.Context()

            #  Socket to talk to server
            print("Connecting to hello world server…")
            socket = context.socket(zmq.REQ)
            url = "tcp://"+sys.argv[1] + ":"+sys.argv[2]
            socket.connect(url)
            # socket.connect("tcp://35.237.244.119:5555")

            iterations = len(trainloader)
            print(iterations)
            send_iterations = str(iterations).encode()
            socket.send(send_iterations)

            names = socket.recv()
            recv_names = names.decode()
            # print(recv_names)

            print(datasetsize_used)
            send_dataset_size = str(datasetsize_used).encode()
            socket.send(send_dataset_size)

            names = socket.recv()
            recv_names = names.decode()





            epoch_start_time = time.time()
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                step_start_time = time.time()
                print(r, epoch, i)
                inputs, labels = data[0].to(device), data[1].to(device)

                client_optimizer.zero_grad()

                # print("LABELS", type(labels))
                bytes_labels = array_to_bytes(labels.cpu())
                socket.send(bytes_labels)
                print("labels_sent")

                ##dummy......
                names = socket.recv()
                recv_names = names.decode()
                print(recv_names)

                # Client part
                activations = client_model(inputs)
                server_inputs = activations.detach().clone()

                # print("inside for for...")
                bytes_server_inputs = array_to_bytes(server_inputs.cpu())
                server_work_time_start = time.time()
                socket.send(bytes_server_inputs)
                print("data_sent")

                ###################################################################################################

                # # Simulation of server part is happening in this portion
                # # Server part
                # server_inputs = Variable(server_inputs, requires_grad=True)
                # outputs = server_model(server_inputs)
                # loss = criterion(outputs, labels)
                # loss.backward()

                # # server optimization
                # server_optimizer.step()
                ################################################################################

                recv_loss = socket.recv()
                server_work_time_end = time.time()
                numpy_loss = bytes_to_array(recv_loss)
                loss = torch.from_numpy(numpy_loss)
                loss = loss.to(device)
                print("loss_recieved")

                # Simulation of Client Happening in this portion
                # Client optimization

                # activations.backward(server_inputs.grad)
                client_optimizer.step()

                running_loss += loss.item()

                if i % log_steps == log_steps-1:
                    print('[{}, {}] loss: {:.3f}'.format(
                        epoch + 1, i + 1, running_loss / log_steps))
                    logging.info('[{}, {}] loss:  {:.3f}'.format(
                        epoch + 1, i + 1, running_loss / log_steps))
                    running_loss = 0.0

                step_end_time = time.time()
                total_one_step_time = step_end_time - step_start_time
                server_work_time = server_work_time_end - server_work_time_start
                print("CLIENT_TOTAL_ONE_STEP_TIME = {:.3f}, SERVER_WORK_TIME = {:.3f}" .format(
                    total_one_step_time, server_work_time))
                logging.info("CLIENT_TOTAL_ONE_STEP_TIME = {:.3f}    , SERVER_WORK_TIME = {:.3f}" .format(
                    total_one_step_time, server_work_time))

            epoch_end_time = time.time()
            total_one_epoch_time = epoch_end_time - epoch_start_time
            print("CLIENT_TOTAL_ONE_EPOCH_TIME = ", total_one_epoch_time)
            logging.info('CLIENT_TOTAL_ONE_EPOCH_TIME = {:.3f}'.format(
                total_one_epoch_time))

        
            socket.close()
            context.term()

        ############################################################
        ########### Sending model to fedServer #####################
        ############################################################

        context1 = zmq.Context()

        #  Socket to talk to server
        print("Connecting to fed_avg server to give weights…")
        socket1 = context1.socket(zmq.REQ)
        url = "tcp://"+sys.argv[11] + ":"+sys.argv[12]
        socket1.connect(url)
        # socket.connect("tcp://35.237.244.119:5555")

        weights = client_model.state_dict()
        # print(type(weights))
        print("SIze of model weights (before) in bytes is:-", getsizeof(weights))
        bytes_weights = ordered_dict_to_bytes(weights)
        print("SIze of model weights (after) in bytes is:-",
              getsizeof(bytes_weights))
        # time.sleep(10)
        socket1.send(bytes_weights)

        ## dummy recv
        names = socket1.recv()
        recv_names = names.decode()

        ## send dataset size for weighted avg
        socket1.send(send_dataset_size)

        ## dummy recv
        names = socket1.recv()
        recv_names = names.decode()

        del weights
        del bytes_weights

        socket1.close()
        context1.term()

        #############################################################
        ######### Recieving global model from fedServer #############
        #############################################################

        context2 = zmq.Context()

        #  Socket to talk to server
        print("Connecting to fed_avg server to recv global weights…")
        socket2 = context2.socket(zmq.REQ)
        url = "tcp://"+sys.argv[11] + ":"+sys.argv[12]
        socket2.connect(url)
        # socket.connect("tcp://35.237.244.119:5555")

        msg = "send_global_weights"
        send_msg = msg.encode()
        socket2.send(send_msg)

        global_weights = socket2.recv()
        print("Global weights recieved from fedServer")
        print("SIze of global model weights (before) in bytes is:-", getsizeof(global_weights))
        global_numpy_weights = bytes_to_dict(global_weights)
        print("SIze of global model weights (after) in bytes is:-", getsizeof(global_numpy_weights))

        socket2.close()
        context2.term()



    training_end_time = time.time()
    training_time = training_end_time - training_start_time
    print("CLIENT_TOTAL_TRAINING_TIME = ", training_time)
    logging.info('CLIENT_TOTAL_TRAINING_TIME = {:.3f}'.format(training_time))

    # model_save_name = "./client_thread_model_" + sys.argv[4] + "_" + sys.argv[2] + "_" + sys.argv[3] + "_" + \
    #     sys.argv[5] + "_" + sys.argv[6] + "_" + sys.argv[7] + \
    #     "_" + sys.argv[8] + "_" + sys.argv[9] + ".pt"
    # torch.save(client_model.state_dict(), model_save_name)
    # print("MODEL_SAVED.")

#################################################################################################################################
