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
from convert import bytes_to_array, array_to_bytes


if __name__ == '__main__':


    ##################################################################################################################
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = 'cpu'
    print(device)

    class ResNet18Server(nn.Module):
        """docstring for ResNet"""

        def __init__(self, config):
            super(ResNet18Server, self).__init__()
            self.logits = config["logits"]
            self.cut_layer = config["cut_layer"]

            self.model = models.resnet18(pretrained=False)
            num_ftrs = self.model.fc.in_features
            # Explain this part
            self.model.fc = nn.Sequential(nn.Flatten(),
                                          nn.Linear(num_ftrs, self.logits))

            self.model = nn.ModuleList(self.model.children())
            self.model = nn.Sequential(*self.model)

        def forward(self, x):
            for i, l in enumerate(self.model):
                # Explain this part
                if i <= self.cut_layer:
                    continue
                x = l(x)
            return nn.functional.softmax(x, dim=1)

    config = {"cut_layer": 3, "logits": 10}
    # client_model = ResNet18Client(config).to(device)
    server_model = ResNet18Server(config).to(device)

    criterion = nn.CrossEntropyLoss()
    # client_optimizer = optim.SGD(client_model.parameters(), lr=0.01, momentum=0.9)
    server_optimizer = optim.SGD(
        server_model.parameters(), lr=0.01, momentum=0.9)


    total_client_num = 2                    # change depending on clients

    clients_connection_url = ["tcp://*:5555", "tcp://*:5556"]        # manually type the IPs of the client.
    client_weights = [None] * total_client_num
    active_client_num = 0

    num_epochs = 50                ## fixed manually, if changed inform all clients
    for epoch in range(num_epochs):
        print("EPOCH NO in server:- ", epoch)
        # Iterate over multiple clients in one epoch
        for client_num in range(total_client_num):
            print("client num:", client_num)

            print("Current active client is {}".format(client_num))

            context = zmq.Context()
            socket = context.socket(zmq.REP)
            socket.bind(clients_connection_url[client_num])

            iterations = socket.recv()
            recv_iterations = int(iterations.decode())
            print(recv_iterations)

            # names = "sagar"
            # send_names = names.encode()
            # socket.send(send_names)

            # Logic to transfer the weights from the previous client

            if total_client_num == 1:
                names = "sagar"
                send_names = names.encode()
                socket.send(send_names)

            else:
                if client_num == 0:
                    if epoch != 0:
                        prev_client = total_client_num - 1
                        print("Prev client:- ", prev_client)
                        # prev_client_weights = client_weights[prev_client]
                        socket.send(client_weights[prev_client])
                        # client.load_state_dict(prev_client_weights)
                        # print("Loaded client {}'s weight successfully".format(prev_client))

                    else:
                        names = "sagar"
                        send_names = names.encode()
                        socket.send(send_names)
                
                else:
                    prev_client = client_num - 1
                    # prev_client_weights = client_weights[prev_client]
                    socket.send(client_weights[prev_client])
                    # client.load_state_dict(prev_client_weights)
                    # print("Loaded client {}'s weight successfully".format(prev_client))

            print("MODEL SENT")







            for j in range(recv_iterations):
                ## Extra #####
                if j==5:
                    break
                ##############
                print(j)

                server_optimizer.zero_grad()

                recv_labels = socket.recv()
                numpy_labels = bytes_to_array(recv_labels)
                labels = torch.from_numpy(numpy_labels)
                labels = labels.to(device)
                # print("labels_recieved")

                ##dummy......
                names = "sagar"
                send_names = names.encode()
                socket.send(send_names)

                # print("inside for for")
                recv_serv_inputs = socket.recv()
                numpy_server_inputs = bytes_to_array(recv_serv_inputs)
                server_inputs = torch.from_numpy(numpy_server_inputs)
                server_inputs = server_inputs.to(device)
                # print("data_recieved")

                ###################################################################################################

                # Simulation of server part is happening in this portion
                # Server part
                server_inputs = Variable(server_inputs, requires_grad=True)
                outputs = server_model(server_inputs)
                loss = criterion(outputs, labels)
                loss.backward()

                # server optimization
                server_optimizer.step()

                transfer_loss = loss.detach().clone()
                bytes_loss = array_to_bytes(transfer_loss.cpu())
                socket.send(bytes_loss)
                # print("loss_sent")

                ################################################################################

            weights = socket.recv()
            print("Weights recieved from client")
            # numpy_weights = bytes_to_array(weights)
            client_weights[client_num] = weights

            del weights
            # del numpy_weights

            names = "weights_received"
            send_names = names.encode()
            socket.send(send_names)




            socket.close()
            context.term()


        


##################################################################################################################
