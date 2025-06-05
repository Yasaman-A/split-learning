import torch.nn as nn
from torchvision import models
import copy
import operator

class ResNet18Client(nn.Module):
    """docstring for ResNet"""

    def __init__(self):
        super(ResNet18Client, self).__init__()

        self.model = models.resnet18(weights=None)

        self.logits = 10
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Flatten(),
                                                  nn.Linear(num_ftrs, self.logits))

        self.layers = list(self.model.children())

class ResNet18Server(nn.Module):
    """docstring for ResNet"""

    def __init__(self):
        super(ResNet18Server, self).__init__()

        self.model = models.resnet18(weights=None)
        num_ftrs = self.model.fc.in_features

        self.logits = 10

        self.model.fc = nn.Sequential(nn.Flatten(),
                                        nn.Linear(num_ftrs, self.logits))

        self.layers = list(self.model.children())




def average_models(server: bool, model_list, datasize, cut_layer_list):
    """
    Aggregates all input models into one model.

    Args:
        server (bool): Flag whether averaging as a server model (False = client averaging).
        model_list (list): List of all relevant models to aggregate.
        datasize (list): List of datasizes for each model (for weighted aggregation).
        cut_layer_list (list): List of cut layers for each model.

    Notes:
        model_list, datasize, and cut_layer_list indexes should be index-aligned by model.
    """
    if server: #Find earliest cut for server aggregation.
        split_idx = cut_layer_list.index(min(cut_layer_list))
        comparator = operator.gt
    else:      #Find latest cut for client aggregation.
        split_idx = cut_layer_list.index(max(cut_layer_list))
        comparator = operator.le

    weights_avg = copy.deepcopy(model_list[split_idx].state_dict())

    #multiply weights by allocated data for the weighted averaging
    for i, data in enumerate(datasize):
        state_dict = model_list[i].state_dict()
        for key in state_dict.keys():
            state_dict[key] *= data
        model_list[i].load_state_dict(state_dict)



    #for each layer, sum the weights from models that worked on those layers.
    for layer_idx, (layer_name, layer) in enumerate(model_list[split_idx].named_children()):
        for param_name, _ in layer.named_parameters():
            name = f"{layer_name}.{param_name}"

            weight_value = 0
            weight_size_sum = 0

            for model_idx, model in enumerate(model_list):
                    #comparator is:
                    #   server: >
                    #   client: <=
                    #if current_layer vs cut_layer_list
                if comparator(layer_idx, int(cut_layer_list[model_idx])):
                    weight_value += model.state_dict()[name]
                    weight_size_sum += datasize[model_idx]
            
            if weight_size_sum > 0:
                weights_avg[name] = weight_value / weight_size_sum

    return weights_avg




def custom_model_avg(server: bool, state_dicts, data_size, cut_layer_list):
    """
    Creates a custom average model given a list of state dicts, data sizes, and cut layers.
    Chooses between averaging a server or client model based on the input "server" flag.

    Args:
        server (bool): Flag to average as a server (True) or client (False) model.
        state_dicts (list): List of state dicts for each model to aggregate.
        data_size (list): List of data sizes each model trained on (used for weighted averaging).
        cut_layer_list (list): List of cut layers for each model.

    Notes:
        state_dicts, data_size, and cut_layer_list indexes should be index-aligned by model.
    """
    if server:
        model_container = ResNet18Server
    else:
        model_container = ResNet18Client
    
    trained_models = []

    for state_dict in state_dicts:
        defn = model_container()
        defn.load_state_dict(state_dict)
        trained_models.append(defn)

    return average_models(server, trained_models, data_size, cut_layer_list)

    

