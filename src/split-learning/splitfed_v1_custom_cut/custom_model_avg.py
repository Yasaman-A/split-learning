import torch
import torch.nn as nn
from torchvision import models
import copy
from locale import atoi

# model_path = r"G:\MRU\Split Learning\zeroMQ\splitFed\avg\client_fedAvg_model_r1_5_4445_10.pt"
# model_path = r"G:\MRU\Split Learning\zeroMQ\experiments\exp1\client_model_5555_cpu_3_10.pt"


class ResNet18Client(nn.Module):
    """docstring for ResNet"""

    # Explain initialize (listing the neural network architecture and other related parameters)
    def __init__(self):
        super(ResNet18Client, self).__init__()
        # Explain this line
        # self.cut_layer = config["cut_layer"]

        # Explain this line
        self.model = models.resnet18(pretrained=False)

        self.logits = 10
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Flatten(),
                                            nn.Linear(num_ftrs, self.logits))

        self.model = nn.ModuleList(self.model.children())
        self.model = nn.Sequential(*self.model)

class ResNet18Server(nn.Module):
    """docstring for ResNet"""

    def __init__(self):
        super(ResNet18Server, self).__init__()

        self.model = models.resnet18(pretrained=False)
        num_ftrs = self.model.fc.in_features

        self.logits = 10

        # Explain this part
        self.model.fc = nn.Sequential(nn.Flatten(),
                                        nn.Linear(num_ftrs, self.logits))

        self.model = nn.ModuleList(self.model.children())
        self.model = nn.Sequential(*self.model)




class Cl_Custom_Avg(nn.Module):
    """docstring for ResNet"""

    # Explain initialize (listing the neural network architecture and other related parameters)
    def __init__(self):
        super(Cl_Custom_Avg, self).__init__()
        # Explain this line
        # self.cut_layer = config["cut_layer"]

        # Explain this line
        self.model = models.resnet18(pretrained=False)

        self.model = nn.ModuleList(self.model.children())
        self.model = nn.Sequential(*self.model)


    def client_custom_avg(self, model_list, datasize, cut_layer_list):

        ## Select the model with larger cut layer for finding average........
        # w_avg = copy.deepcopy(w[-1])
        # model with bigger cut layer..

        max_index = cut_layer_list.index(max(cut_layer_list))
        w_avg = copy.deepcopy(model_list[max_index].state_dict())


        ### for multiplying (weighted average)............
        for i, data in enumerate(datasize):
            tt = model_list[i].state_dict()
            for key in tt.keys():
                tt[key] *= data
            model_list[i].load_state_dict(tt)
            del tt


        # for l, n in enumerate(w[0].children()):
        for l, n in enumerate(self.model):
            # print("\nINDEX----", l)
            
            for name, wt in n.named_parameters():
                name = "model." + str(l) + "." + name
                # print("Index--> ", l, "NAME-->", name)

                weight_value = 0
                summation = 0

                flag = False

                for i in range(0, len(model_list)):
                    print("cut_layer_list1" + str(i+1) + ":" + str(cut_layer_list[i]))
                    if(cut_layer_list[i] >= l):
                        # print("INSIDE IF>>>")

                        flag = True
                        weight_value += model_list[i].state_dict()[name]
                        summation += datasize[i]
                
                if (flag == True):
                    w_avg[name] = torch.div(weight_value, float(summation))
                    # print("--------------------------------")

                del weight_value
                del summation

        return w_avg


class Serv_Custom_Avg(nn.Module):
    """docstring for ResNet"""

    # Explain initialize (listing the neural network architecture and other related parameters)
    def __init__(self):
        super(Serv_Custom_Avg, self).__init__()
        # Explain this line
        # self.cut_layer = config["cut_layer"]

        # Explain this line
        self.model = models.resnet18(pretrained=False)
        num_ftrs = self.model.fc.in_features

        self.logits = 10

        # Explain this part
        self.model.fc = nn.Sequential(nn.Flatten(),
                                      nn.Linear(num_ftrs, self.logits))

        self.model = nn.ModuleList(self.model.children())
        self.model = nn.Sequential(*self.model)


    def server_custom_avg(self, model_list, datasize, cut_layer_list):

        ## Select the model with smaller cut layer for finding average........
        # w_avg = copy.deepcopy(w[-1])
        # model with smaller cut layer..

        min_index = cut_layer_list.index(min(cut_layer_list))
        w_avg = copy.deepcopy(model_list[min_index].state_dict())

        ### for multiplying (weighted average)............
        for i, data in enumerate(datasize):
            tt = model_list[i].state_dict()
            for key in tt.keys():
                tt[key] *= data
            model_list[i].load_state_dict(tt)
            del tt

        # for l, n in enumerate(w[0].children()):
        for l, n in enumerate(self.model):
            # print("\nINDEX----", l)

            for name, wt in n.named_parameters():
                name = "model." + str(l) + "." + name
                # print("Index--> ", l, "NAME-->", name)

                weight_value = 0
                summation = 0

                flag = False

                for i in range(0, len(model_list)):
                    print("cut_layer_list" + str(i+1) + ":" + str(cut_layer_list[i]))
                    if(atoi(cut_layer_list[i]) < l):
                        # print("INSIDE IF>>>")

                        flag = True
                        weight_value += model_list[i].state_dict()[name]
                        summation += datasize[i]

                if (flag == True):
                    w_avg[name] = torch.div(weight_value, float(summation))
                    # print("--------------------------------")

                del weight_value
                del summation

        return w_avg





def client_custom_avg_model(weights, data_size, cut_layer):
    trained_models = []
    
    ## Putting weights to model architectures....
    for i in range(0, len(weights)):
        m = ResNet18Client()
        m.load_state_dict(weights[i])

        trained_models.append(m)
        del m

    # data_size = [5000, 10000, 15000]
    # cut_layer = [3, 4, 5]

    avg_model = Cl_Custom_Avg()
    return avg_model.client_custom_avg(trained_models, data_size, cut_layer)


def server_custom_avg_model(weights, data_size, cut_layer):
    trained_models = []

    ## Putting weights to model architectures....
    for i in range(0, len(weights)):
        m = ResNet18Server()
        m.load_state_dict(weights[i])

        trained_models.append(m)
        del m

    # data_size = [5000, 10000, 15000]
    # cut_layer = [3, 4, 5]

    avg_model = Serv_Custom_Avg()
    return avg_model.server_custom_avg(trained_models, data_size, cut_layer)
