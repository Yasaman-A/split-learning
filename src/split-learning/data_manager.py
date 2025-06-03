import random
import torch as t
import torchvision as tv
import pickle
import numpy as np
from torch.utils.data import Subset
from tqdm.auto import tqdm

'''
Dataset Getters
=================
Here are where the code for the Dataset Getters should be contained. 
Each getter should perform the needed transformations, then return a tuple containing:
        (training_data, testing data)
Note: Seeds will be set before the transformers are called. 

Afterwards, the function name and callable should be added to the DATASET_GETTERS 
global dictionary at the bottom of this file.
'''

def get_cifar10():
    training_data = tv.datasets.CIFAR10(root="./data", train=True, download=True)
    testing_data  = tv.datasets.CIFAR10(root="./data", train=False, download=True)

    return training_data, testing_data


def set_seed(seed):
    t.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    return




def shuffle_and_split_iid_data(training_data, testing_data, num_clients, output_name):
    def split_to_subsets(dataset, num_clients):
        size = len(dataset)
        indices = np.arange(size)
        np.random.shuffle(indices)
        
        split_sizes = [size // num_clients] * num_clients
        remainder = size % num_clients
        for i in range(remainder):
            split_sizes[i] +=1
        
        splits = []
        origin = 0
        for size in split_sizes:
            splits.append(indices[origin:origin+size])
            origin += size
        return splits

    def preprocess_subset(subset, index):
        data = []
        labels = []

        for x, y in tqdm(subset, desc=f"Processing data for client: {index}"):
            data.append(x)
            labels.append(y)

        return list(zip(data, labels))

    train_splits = split_to_subsets(training_data, num_clients)

    train_subsets = [Subset(training_data, indices) for indices in train_splits]

    preprocessed = []
    for index, subset in enumerate(train_subsets): #Note: This can be threaded if performance is lacking
        preprocessed.append(preprocess_subset(subset, index))

    testing_set = preprocess_subset(testing_data, -1)

    #TODO: Print out class distribution information
    print(f"Saving split data as: \'{output_name}.pkl\'")
    with open(output_name + ".pkl", 'wb') as f:
        pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saving test data as: \'{output_name}_test.pkl\'")
    with open(output_name + "_test" + ".pkl", 'wb') as f:
        pickle.dump(testing_set, f, protocol=pickle.HIGHEST_PROTOCOL)

    return

'''



'''
DATASET_GETTERS = {
    'cifar10': get_cifar10,
}

def create_iid_dataset(dataset_name, num_clients=1, output_name="output", seed=None):
    if seed is not None:
        set_seed(seed)
    if dataset_name not in DATASET_GETTERS:
        print(f"Dataset \"{dataset_name}\" not implemented. Please add it to the data manager.")
        return None
    
    train, test = DATASET_GETTERS[dataset_name]()
    shuffle_and_split_iid_data(train, test, num_clients, output_name)
    return 


#create_iid_dataset("cifar10", num_clients=6, seed=42, output_name="six_clients")