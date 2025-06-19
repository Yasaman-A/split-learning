import torch.nn as nn
from torchvision import models
import copy
import operator

class ResNet18(nn.Module):
    """docstring for ResNet"""

    def __init__(self):
        super(ResNet18, self).__init__()

        self.model = models.resnet18(weights=None)

        self.logits = 10
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Flatten(),
                                                  nn.Linear(num_ftrs, self.logits))

        self.layers = list(self.model.children())


def average_models(server: bool, model_list, datasizes, cut_layer_list):
    """
    Aggregates all input models into one model.

    Args:
        server (bool): Flag whether averaging as a server model (False = client averaging).
        model_list (list): List of all relevant models to aggregate.
        datasizes (list): List of datasizes for each model (for weighted aggregation).
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
    for i, data in enumerate(datasizes):
        state_dict = model_list[i].state_dict()
        for key in state_dict.keys():
            state_dict[key] *= data
        model_list[i].load_state_dict(state_dict)


    weight_exposure = {}


    #for each layer, sum the weights from models that worked on those layers.
    for layer_idx, (layer_name, layer) in enumerate(model_list[split_idx].model.named_children()):
        for key in model_list[split_idx].state_dict().keys():
            if key.startswith(f"model.{layer_name}"):
                weight_value = 0
                weight_size_sum = 0

                for model_idx, model in enumerate(model_list):
                        #comparator is:
                        #   server: >
                        #   client: <=
                        #if current_layer vs cut_layer_list
                    if comparator(layer_idx, int(cut_layer_list[model_idx])):
                        weight_value += model.state_dict()[key]
                        weight_size_sum += datasizes[model_idx]
                
                if weight_size_sum > 0:
                    weights_avg[key] = weight_value / weight_size_sum
                    weight_exposure[key] = weight_exposure.get(key, 0) + weight_size_sum


    return weights_avg, weight_exposure


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
    #kept in case asymmetrical model definitions.
    #if used, change defn = ModelFoo() to defn = model_container
    # if server:
    #     model_container = ResNet18Server
    # else:
    #     model_container = ResNet18Client
    
    trained_models = []

    for state_dict in state_dicts:
        defn = ResNet18()
        defn.load_state_dict(state_dict)
        trained_models.append(defn)

    avg_model, avg_exposure = average_models(server, trained_models, data_size, cut_layer_list)

    return avg_model, avg_exposure




def combine_fed_avg_models(client_fedavg, client_exposure, server_fedavg, server_exposure):
    '''
    Merges the federated client model and federated server model into one combined
    fedearted model.

    Currently hardcoded to use symmetrical ResNet18.

    Args:
        client_fedavg (nn.Module): State dict from the federated client model
        client_exposure (dict): dict containing the magnitude of data key in the state dict was exposed to
        server_fedavg (nn.Module): State dict from the federated server model
        server_exposure (dict): dict containing the magnitude of data key in the state dict was exposed to
    
    '''
    combined = ResNet18()
    state_dict = combined.state_dict()


    for i, key in enumerate(client_fedavg.keys()):
        c_exposure = client_exposure.get(key, 0)
        s_exposure = server_exposure.get(key, 0)
        total = c_exposure + s_exposure

        state_dict[key] = (client_fedavg[key] * c_exposure + server_fedavg[key] * s_exposure) / total
    
    return state_dict

