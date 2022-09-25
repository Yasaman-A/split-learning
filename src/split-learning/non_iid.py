import os
import random
from .lib import custom_image_dataset
from tqdm import tqdm
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import Compose
torch.backends.cudnn.benchmark = True
import sys
import pickle


##Not used 
#real_wd = True#sys.argv[4].lower() == 'true'  # False: non_iid dataset, True: Real-world dataset


"""
classes_pc: classes per client, it is used to divide the balanced dataset to non-IID dataset by creating an unbalanced representation of classes among the clients. For e.g., if the classes_pc=1, then all the clients will have images from one class only, thus creating an extensive imbalance among the clients. (Ref: Figure 2 )
num_clients: Total number of clients among which images are to be distributed.
batch_size: Loading of the data into the data loader by batches.

## Not used
real_wd: We are creating two types of datasets, one is the real-world dataset (Figure 1) and another is the extreme non-IID dataset (Figure 2). If real_wd is TRUE then dataset replicating real-life is created, i.e. real-world dataset (figure 1). If real_wd is FALSE (by default) then the extreme non-IID dataset is created.

"""



#### get cifar dataset in x and y form

def get_cifar10():
    '''Return CIFAR10 train/test data and labels as numpy arrays'''
    data_train = torchvision.datasets.CIFAR10('./data', train=True, download=True)
    data_test = torchvision.datasets.CIFAR10('./data', train=False, download=True) 

    x_train, y_train = data_train.data.transpose((0,3,1,2)), np.array(data_train.targets)
    x_test, y_test = data_test.data.transpose((0,3,1,2)), np.array(data_test.targets)

    return x_train, y_train, x_test, y_test

def print_image_data_stats(data_train, labels_train, data_test, labels_test):
    print("\nData: ")
    print(" - Train Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
    data_train.shape, labels_train.shape, np.min(data_train), np.max(data_train),
        np.min(labels_train), np.max(labels_train)))
    print(" - Test Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
    data_test.shape, labels_test.shape, np.min(data_train), np.max(data_train),
        np.min(labels_test), np.max(labels_test)))
  
  
def clients_rand(train_len, nclients):
    '''
    train_len: size of the train data
    nclients: number of clients

    Returns: to_ret

    This function creates a random distribution 
    for the clients, i.e. number of images each client 
    possess.
    '''
    client_tmp=[]
    sum_=0
    #### creating random values for each client ####
    for i in range(nclients-1):
        tmp=random.randint(10,100)
        sum_+=tmp
        client_tmp.append(tmp)

    client_tmp= np.array(client_tmp)
    #### using those random values as weights ####
    clients_dist= ((client_tmp/sum_)*train_len).astype(int)
    num  = train_len - clients_dist.sum()
    to_ret = list(clients_dist)
    to_ret.append(num)
    return to_ret


def split_image_data_realwd(data, labels, n_clients=100, verbose=True):
    '''
    Splits (data, labels) among 'n_clients s.t. every client can holds any number of classes which is trying to simulate real world dataset
    Input:
    data : [n_data x shape]
    labels : [n_data (x 1)] from 0 to n_labels(10)
    n_clients : number of clients
    verbose : True/False => True for printing some info, False otherwise
    Output:
    clients_split : splitted client data into desired format
    '''
    def break_into(n, m):
        ''' 
        return m random integers with sum equal to n 
        '''
        to_ret = [1 for i in range(m)]
        for i in range(n-m):
            ind = random.randint(0, m-1)
            to_ret[ind] += 1
        return to_ret

    #### constants ####
    n_classes = len(set(labels))
    classes = list(range(n_classes))
    np.random.shuffle(classes)
    label_indcs = [list(np.where(labels == class_)[0]) for class_ in classes]

    #### classes for each client ####
    tmp = [np.random.randint(1, 10) for i in range(n_clients)]
    total_partition = sum(tmp)

    #### create partition among classes to fulfill criteria for clients ####
    class_partition = break_into(total_partition, len(classes))

    #### applying greedy approach first come and first serve ####
    class_partition = sorted(class_partition, reverse=True)
    class_partition_split = {}

    #### based on class partition, partitioning the label indexes ###
    for ind, class_ in enumerate(classes):
        class_partition_split[class_] = [
            list(i) for i in np.array_split(label_indcs[ind], class_partition[ind])]

    #   print([len(class_partition_split[key]) for key in  class_partition_split.keys()])

    clients_split = []
    count = 0
    for i in range(n_clients):
        n = tmp[i]
        j = 0
        indcs = []

        while n > 0:
            class_ = classes[j]
            if len(class_partition_split[class_]) > 0:
                indcs.extend(class_partition_split[class_][-1])
                count += len(class_partition_split[class_][-1])
                class_partition_split[class_].pop()
                n -= 1
            j += 1

    ##### sorting classes based on the number of examples it has #####
    classes = sorted(classes, key=lambda x: len(
        class_partition_split[x]), reverse=True)
    if n > 0:
        raise ValueError(" Unable to fulfill the criteria ")
    clients_split.append([data[indcs], labels[indcs]])
    #   print(class_partition_split)
    #   print("total example ",count)

    def print_split(clients_split):
        print("Data split:")
        for i, client in enumerate(clients_split):
            split = np.sum(client[1].reshape(1, -1) ==
                            np.arange(n_labels).reshape(-1, 1), axis=1)
            print(" - Client {}: {}".format(i, split))
        print()

    if verbose:
      print_split(clients_split)

    clients_split = np.array(clients_split)

    return clients_split




def split_image_data(data, labels, n_clients=100, classes_per_client=10, shuffle=True, verbose=True):
    '''
    Splits (data, labels) among 'n_clients s.t. every client can holds 'classes_per_client' number of classes
    Input:
        data : [n_data x shape]
        labels : [n_data (x 1)] from 0 to n_labels
        n_clients : number of clients
        classes_per_client : number of classes per client
        shuffle : True/False => True for shuffling the dataset, False otherwise
        verbose : True/False => True for printing some info, False otherwise
    Output:
        clients_split : client data into desired format
    '''
    #### constants #### 
    n_data = data.shape[0]
    n_labels = np.max(labels) + 1


    ### client distribution ####
    data_per_client = clients_rand(len(data), n_clients)
    data_per_client_per_class = [np.maximum(1,nd // classes_per_client) for nd in data_per_client]
    
    # sort for labels
    data_idcs = [[] for i in range(n_labels)]
    for j, label in enumerate(labels):
        data_idcs[label] += [j]
    if shuffle:
        for idcs in data_idcs:
            np.random.shuffle(idcs)
    
    # split data among clients
    clients_split = []
    c = 0
    for i in range(n_clients):
            client_idcs = []
                
            budget = data_per_client[i]
            c = np.random.randint(n_labels)
            while budget > 0:
                take = min(data_per_client_per_class[i], len(data_idcs[c]), budget)
                
                client_idcs += data_idcs[c][:take]
                data_idcs[c] = data_idcs[c][take:]
                
                budget -= take
                c = (c + 1) % n_labels
            
            clients_split += [(data[client_idcs], labels[client_idcs])]

    def print_split(clients_split): 
        print("Data split:")
        for i, client in enumerate(clients_split):
            split = np.sum(client[1].reshape(1,-1)==np.arange(n_labels).reshape(-1,1), axis=1)
            print(" - Client {}: {}".format(i,split))
        print()
      
    if verbose:
      print_split(clients_split)
  
    clients_split = np.array(clients_split)
    
    return clients_split


def shuffle_list(data):
    '''
    This function returns the shuffled data
    '''
    for i in range(len(data)):
        tmp_len= len(data[i][0])
        index = [i for i in range(tmp_len)]
        random.shuffle(index)
        data[i][0],data[i][1] = shuffle_list_data(data[i][0],data[i][1])
    return data

def shuffle_list_data(x, y):
    '''
    This function is a helper function, shuffles an
    array while maintaining the mapping between x and y
    '''
    inds = list(range(len(x)))
    random.shuffle(inds)
    return x[inds],y[inds] 

def get_default_data_transforms(train=True, verbose=True):
    transforms_train = {
    'cifar10' : transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),#(0.24703223, 0.24348513, 0.26158784)
    }
    transforms_eval = {    
    'cifar10' : transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    }
    if verbose:
        print("\nData preprocessing: ")
        for transformation in transforms_train['cifar10'].transforms:
            print(' -', transformation)
        print()

    return (transforms_train['cifar10'], transforms_eval['cifar10'])

def get_data_loaders(nclients,batch_size,classes_pc=10 ,verbose=True ):
    
    x_train, y_train, x_test, y_test = get_cifar10()

    if verbose:
        print_image_data_stats(x_train, y_train, x_test, y_test)

    transforms_train, transforms_eval = get_default_data_transforms(verbose=False)
    
    split = split_image_data(x_train, y_train, n_clients=nclients, 
            classes_per_client=classes_pc, verbose=verbose)
    
    split_tmp = shuffle_list(split)
    
    client_loaders = [torch.utils.data.DataLoader(custom_image_dataset.CustomImageDataset(x, y, transforms_train), 
                                                                    batch_size=batch_size, shuffle=True) for x, y in split_tmp]

    test_loader  = torch.utils.data.DataLoader(custom_image_dataset.CustomImageDataset(x_test, y_test, transforms_eval), batch_size=batch_size, shuffle=False) 

    return client_loaders, test_loader


#classes_pc = 2# int(sys.argv[1])
#num_clients = 6# int(sys.argv[2])
#batch_size = 128 #int(sys.argv[3])

def run(classes_pc, num_clients, batch_size):
    ###### Loading the data using the above function ######
    train_loader, test_loader = get_data_loaders(classes_pc=classes_pc, nclients=num_clients,
                                                          batch_size=batch_size, verbose=True)

    with open('output.pickle', 'wb') as handle:
        pickle.dump(train_loader, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #print(len(train_loader[0]))
    #print(len(train_loader[1]))
    #print(len(train_loader[2]))
    #print(len(train_loader[3]))
    #print("tra loader length:" + len(train_loader[4]))

    for i in range(num_clients):
        print("train loader length" + str(i) +": " +  str(len(train_loader[i])))
    print("test loader length: " +  str(len(test_loader)))


