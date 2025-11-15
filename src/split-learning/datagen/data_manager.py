import random
import torch as t
import torchvision as tv
import pickle
import numpy as np
from torch.utils.data import Subset
from tqdm.auto import tqdm
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split

from .feature_skew_dirichlet import run as feature_skew_dirichlet
from .feature_skew_gaussian import run as feature_skew_gaussian
from .label_skew_dirichlet import run as label_skew_dirichlet
from .label_skew_percentage import run as label_skew_percentage
from .quantity_skew_dirichlet import run as quantity_skew_dirichlet
from .quantity_skew_minsize_dirichlet import run as quantity_skew_minsize_dirichlet

'''
===============================
Dataset Getters
===============================
Here are where the code for the Dataset Getters should be contained. 
Each getter should perform the needed formatting code, then return a tuple containing:
        (training_data, testing data)
Note: Seeds will be set before the transformers are called. 

Afterwards, the function name and callable should be added to the DATASET_GETTERS 
global dictionary in the section below.
'''

def get_cifar10():
    training_data = tv.datasets.CIFAR10(root="./data", train=True, download=True)
    testing_data  = tv.datasets.CIFAR10(root="./data", train=False, download=True)

    return training_data, testing_data

def get_mnist():
    training_data = tv.datasets.MNIST(root="./data", train=True, download=True)
    testing_data  = tv.datasets.MNIST(root="./data", train=False, download=True)

    return training_data, testing_data


'''
===============================
Wrappers
===============================
'''

DATASET_GETTERS = {
    'cifar10': get_cifar10,
    'cifar-10': get_cifar10,
    'mnist': get_mnist
}

def fetch_data(dataset_name):
    if dataset_name not in DATASET_GETTERS:
        print(f"Dataset \"{dataset_name}\" not implemented. Please add it to the data manager.")
        return None
    
    train, test = DATASET_GETTERS[dataset_name]()
    return train, test

def split_train_validation(train_data, val_fraction=0.1, seed=None):
    train, val = train_test_split(
        train_data,
        test_size = val_fraction,
        shuffle=True,
        random_state=seed
    )

    return train, val



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
    if index == -1:   desc=f"Processing data for test set"
    elif index == -2: desc=f"Processing data for val set"
    else:             desc=f"Processing data for client: {index}"

    for x, y in tqdm(subset, desc=desc):
        data.append(x)
        labels.append(y)

    return list(zip(data, labels))



def print_distribution_statistics(dataset, subsets):
    classes = sorted(set([label for _, label in dataset]))

    total_counts = [0] * len(classes)

    for i, subset in enumerate(subsets):
        labels = [dataset[idx][1] for idx in subset.indices]
        label_counts = Counter(labels)
        counts_per_class = [label_counts.get(c, 0) for c in classes]

        for idx, count in enumerate(counts_per_class):
            total_counts[idx] += count

        print(f"Client {i:3d}: "
              f"[{', '.join(f'{count:5d}' for count in counts_per_class)}]"
              f" Train Length: {len(subset)}")

    return


def print_label_distribution(datasets):
    for i, dataset in enumerate(datasets):
        labels = [label for _, label in dataset]
        label_counts = Counter(labels)
        max_label = max(label_counts.keys())
        counts_list = [label_counts.get(label, 0) for label in range(max_label + 1)]
        print(f"Client {i}:")
        print(counts_list)

    return


def save_subsets(output_name, train_subsets, testing_set, val_set):

    print(f"Saving split data as: \'{output_name}.pkl\'")
    with open(output_name + ".pkl", 'wb') as f:
        pickle.dump(train_subsets, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saving test data as: \'{output_name}_test.pkl\'")
    with open(output_name + "_test" + ".pkl", 'wb') as f:
        pickle.dump(testing_set, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saving validation data as: \'{output_name}_val.pkl\'")
    with open(output_name + "_val" + ".pkl", 'wb') as f:
        pickle.dump(val_set, f, protocol=pickle.HIGHEST_PROTOCOL)

    return



'''
===============================
Distributors
===============================
'''

def shuffle_and_split_iid_data(training_data, num_clients):
    '''
    Splits training_data evenly into num_clients IID subsets and saves them to a pickle file.

    Args:
        training_data (Dataset): Dataset object for training
        num_clients (int): Number of clients to split the training data into.
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

    return train_subsets





def split_non_iid_data(training_data, num_clients, classes_per_client=2):
    '''
    Splits the dataset into non-IID subsets by assigning each client a fixed number of classes.

    If the number of clients is greater than the number of classes, then classes are reused and
    distributed among clients.

        training_data (Dataset): Dataset object for training
        num_clients (int): Number of clients to split the training data into.
        classes_per_client (int): Number of classes each client should have data for.
    '''
    
    labels = np.array([label for _, label in training_data])
    sorted_indices = np.argsort(labels)

    num_shards = num_clients * classes_per_client
    shard_size = len(training_data) // num_shards
    shards = [sorted_indices[i * shard_size:(i+1) * shard_size] for i in range(num_shards)]

    np.random.shuffle(shards)

    client_indices = []
    for i in range(num_clients):
        assigned_shards = shards[i * classes_per_client:(i + 1) * classes_per_client]
        client_indices.append(np.concatenate(assigned_shards))

    train_subsets = [Subset(training_data, indices) for indices in client_indices]

    print_distribution_statistics(training_data, train_subsets)
    

    return train_subsets






'''
===============================
Generators
===============================
'''

def create_iid_dataset(dataset_name, num_clients=1, output_name="iid", val_fraction=0.1, seed=None):
    if seed is not None:
        set_seed(seed)

    train, test = fetch_data(dataset_name)
    train, val = split_train_validation(train, val_fraction, seed)


    train_subsets = shuffle_and_split_iid_data(train, test, val, num_clients, output_name)

    preprocessed = []
    for index, subset in enumerate(train_subsets): #Note: This can be threaded if performance is lacking
        preprocessed.append(preprocess_subset(subset, index))

    testing_set = preprocess_subset(test, -1)
    val_set = preprocess_subset(val, -2)

    save_subsets(output_name, preprocessed, testing_set, val_set)
    return 



def create_non_iid_dataset(dataset_name, num_clients=1, output_name="shard", classes_per_client=2,  val_fraction=0.1, seed=None):
    if seed is not None:
        set_seed(seed)

    train, test = fetch_data(dataset_name)
    train, val = split_train_validation(train, val_fraction, seed)

    train_subsets = split_non_iid_data(train, test, val, num_clients, output_name, classes_per_client)

    preprocessed = []
    for index, subset in enumerate(train_subsets): #Note: This can be threaded if performance is lacking
        preprocessed.append(preprocess_subset(subset, index))

    testing_set = preprocess_subset(test, -1)
    val_set = preprocess_subset(val, -2)

    save_subsets(output_name, preprocessed, testing_set, val_set)
    return



def create_dirichlet_feature_skew(
    dataset_name, 
    num_clients=1, 
    output_name="dirichlet_feature", 
    alpha_feat_split=0.5, 
    val_fraction=0.1, 
    seed=None
):
    
    if seed is not None:
        set_seed(seed)

    train, test = fetch_data(dataset_name)
    train, val = split_train_validation(train, val_fraction, seed)

    train_subsets = feature_skew_dirichlet(
        train, 
        alpha_feat_split=alpha_feat_split, 
        num_clients=num_clients, 
        seed=seed
    )

    print_label_distribution(train_subsets)

    testing_set = preprocess_subset(test, -1)
    val_set = preprocess_subset(val, -2)

    save_subsets(output_name, train_subsets, testing_set, val_set)

    return



def create_gaussian_feature_skew(
    dataset_name, 
    num_clients=1, 
    output_name="gaussian_feature", 
    sigma_noise=2, 
    n_bins="n_samples", 
    feat_sample_rate=0.1,
    val_fraction=0.1, 
    seed=None
):

    if seed is not None:
        set_seed(seed)
    
    train, test = fetch_data(dataset_name)
    train, val = split_train_validation(train, val_fraction, seed)

    train_subsets = feature_skew_gaussian(
        train, 
        sigma_noise, 
        num_clients, 
        n_bins=n_bins, 
        feat_sample_rate=feat_sample_rate,
        seed=seed
    )

    print_label_distribution(train_subsets)

    testing_set = preprocess_subset(test, -1)
    val_set = preprocess_subset(val, -2)

    save_subsets(output_name, train_subsets, testing_set, val_set)
    
    return



def create_dirichlet_label_skew(
    dataset_name,
    num_clients=1,
    output_name="dirichlet_label",
    alpha_label_split=0.1,
    val_fraction=0.1,
    seed=None
):
    
    if seed is not None:
        set_seed(seed)

    train, test = fetch_data(dataset_name)
    train, val = split_train_validation(train, val_fraction, seed)

    train_subsets = label_skew_dirichlet(
        train,
        alpha_label_split=alpha_label_split,
        num_clients=num_clients,
        seed=seed
    )

    print_label_distribution(train_subsets)

    testing_set = preprocess_subset(test, -1)
    val_set = preprocess_subset(val, -2)

    save_subsets(output_name, train_subsets, testing_set, val_set)

    return



def create_percentage_label_skew(
    dataset_name,
    num_clients=1,
    output_name="percentage_label",
    percentage_skew=0.1,
    val_fraction=0.1,
    seed=None
):
    if seed is not None:
        set_seed(seed)

    train, test = fetch_data(dataset_name)
    train, val = split_train_validation(train, val_fraction, seed)

    train_subsets = label_skew_percentage(
        train,
        percentage_skew=percentage_skew,
        num_clients=num_clients
    )

    print_label_distribution(train_subsets)

    testing_set = preprocess_subset(test, -1)
    val_set = preprocess_subset(val, -2)

    save_subsets(output_name, train_subsets, testing_set, val_set)

    return



def create_quantity_skew_dirichlet(
    dataset_name,
    num_clients=1,
    output_name="quantity_skew_dirichlet",
    alpha_quant_split=0.1,
    val_fraction=0.1,
    seed=None
):

    if seed is not None:
        set_seed(seed)

    train, test = fetch_data(dataset_name)
    train, val = split_train_validation(train, val_fraction, seed)

    train_subsets = quantity_skew_dirichlet(
        train,
        alpha_quant_split,
        num_clients=num_clients,
        seed=seed
    )

    print_label_distribution(train_subsets)

    testing_set = preprocess_subset(test, -1)
    val_set = preprocess_subset(val, -2)

    save_subsets(output_name, train_subsets, testing_set, val_set)
    
    return



def create_quantity_skew_minsize_dirichlet(
    dataset_name,
    num_clients=1,
    output_name="quantity_skew_minsize_dirichlet",
    alpha_quant_split=0.1,
    val_fraction=0.1,
    seed=None
):

    if seed is not None:
        set_seed(seed)

    train, test = fetch_data(dataset_name)
    train, val = split_train_validation(train, val_fraction, seed)

    train_subsets = quantity_skew_minsize_dirichlet(
        train,
        alpha_quant_split,
        num_clients=num_clients,
        seed=seed
    )

    print_label_distribution(train_subsets)

    testing_set = preprocess_subset(test, -1)
    val_set = preprocess_subset(val, -2)

    save_subsets(output_name, train_subsets, testing_set, val_set)
    
    return