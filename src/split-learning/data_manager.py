import random
import torch as t
import torchvision as tv
import pickle
import numpy as np
from torch.utils.data import Subset
from tqdm.auto import tqdm
from collections import defaultdict, Counter

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






'''
===============================
Helpers
===============================
'''
def set_seed(seed):
    t.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    return


def preprocess_subset(subset, index):
    '''
    removes dataloader references to save file for pickling
    
    Args:
        subset (Subset): Subset object containing data
        index (int): used for tqdm display. use -1 to show testing set.
    '''
    data = []
    labels = []

    for x, y in tqdm(subset, desc=f"Processing data for client: {index}"):
        data.append(x)
        labels.append(y)

    return list(zip(data, labels))



def print_distribution_statistics(dataset, subsets):
    classes = sorted(set([label for _, label in dataset]))

    for i, subset in enumerate(subsets):
        labels = [dataset[idx][1] for idx in subset.indices]
        label_counts = Counter(labels)
        counts_per_class = [label_counts.get(c, 0) for c in classes]
        print(f"Client {i}: {counts_per_class}")

    return




'''
===============================
Distributors
===============================
'''

def shuffle_and_split_iid_data(training_data, testing_data, num_clients, output_name):
    '''
    Splits training_data evenly into num_clients IID subsets and saves them to a pickle file.
    Saves the unmodified testing_data separately.

    Args:
        training_data (Dataset): Dataset object for training
        testing_data (Dataset): Dataset object for testing
        num_clients (int): Number of clients to split the training data into.
        output_name (str): Base filename for saving the splits (training saved as output_name.pkl,
                           testing saved as output_name_test.pkl).
    '''
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

    train_splits = split_to_subsets(training_data, num_clients)

    train_subsets = [Subset(training_data, indices) for indices in train_splits]

    print_distribution_statistics(training_data, train_subsets)

    preprocessed = []
    for index, subset in enumerate(train_subsets): #Note: This can be threaded if performance is lacking
        preprocessed.append(preprocess_subset(subset, index))

    testing_set = preprocess_subset(testing_data, -1)

    print(f"Saving split data as: \'{output_name}.pkl\'")
    with open(output_name + ".pkl", 'wb') as f:
        pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saving test data as: \'{output_name}_test.pkl\'")
    with open(output_name + "_test" + ".pkl", 'wb') as f:
        pickle.dump(testing_set, f, protocol=pickle.HIGHEST_PROTOCOL)

    return





def split_non_iid_data(training_data, testing_data, num_clients, output_name, classes_per_client=2):
    '''
    Splits the dataset into non-IID subsets by assigning each client a fixed number of classes.

    If the number of clients is greater than the number of classes, then classes are reused and
    distributed among clients.

        training_data (Dataset): Dataset object for training
        testing_data (Dataset): Dataset object for testing
        num_clients (int): Number of clients to split the training data into.
        output_name (str): Base filename for saving the splits (training saved as output_name.pkl,
                           testing saved as output_name_test.pkl).
        classes_per_client (int): Number of classes each client should have data for.
    '''
    
    class_to_indices = defaultdict(list)

    for idx, (_, label) in enumerate(training_data):
        class_to_indices[label].append(idx)

    classes = list(class_to_indices.keys())
    num_classes = len(classes)
    random.shuffle(classes)


    client_classes = []
    for i in range(num_clients):
        assigned = []
        start = (i * classes_per_client) % num_classes
        for j in range(classes_per_client):
            assigned.append(classes[(start + j) % num_classes])
        client_classes.append(assigned)

    client_indices = []
    for assigned_classes in client_classes:
        indices = []
        for cls in assigned_classes:
            indices.extend(class_to_indices[cls])
        client_indices.append(indices)

    train_subsets = [Subset(training_data, indices) for indices in client_indices]

    print_distribution_statistics(training_data, train_subsets)
    
    preprocessed = []
    for index, subset in enumerate(train_subsets): #Note: This can be threaded if performance is lacking
        preprocessed.append(preprocess_subset(subset, index))

    testing_set = preprocess_subset(testing_data, -1)

    print(f"Saving split data as: \'{output_name}.pkl\'")
    with open(output_name + ".pkl", 'wb') as f:
        pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saving test data as: \'{output_name}_test.pkl\'")
    with open(output_name + "_test" + ".pkl", 'wb') as f:
        pickle.dump(testing_set, f, protocol=pickle.HIGHEST_PROTOCOL)

    return




'''
===============================
Wrappers
===============================
'''

DATASET_GETTERS = {
    'cifar10': get_cifar10,
    'cifar-10': get_cifar10,
}

def fetch_data(dataset_name):
    if dataset_name not in DATASET_GETTERS:
        print(f"Dataset \"{dataset_name}\" not implemented. Please add it to the data manager.")
        return None
    
    train, test = DATASET_GETTERS[dataset_name]()
    return train, test


def create_iid_dataset(dataset_name, num_clients=1, output_name="output", seed=None):
    if seed is not None:
        set_seed(seed)

    train, test = fetch_data(dataset_name)

    shuffle_and_split_iid_data(train, test, num_clients, output_name, seed=seed)
    return 


def create_non_iid_dataset(dataset_name, num_clients, output_name="output", classes_per_client=2, seed=None):
    if seed is not None:
        set_seed(seed)

    train, test = fetch_data(dataset_name)

    split_non_iid_data(train, test, num_clients, output_name, classes_per_client)
    return

#create_iid_dataset("cifar10", num_clients=6, seed=42, output_name="six_clients")
create_non_iid_dataset("cifar-10", 3, "non-iid", 4, seed=42)