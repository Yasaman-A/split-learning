import os
import random
from ..lib import custom_image_dataset
from tqdm import tqdm
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import Compose
import sys
import pickle
import warnings

warnings.filterwarnings("ignore")

# Import FedArtML library for quantity skew
try:
    from fedartml import SplitAsFederatedData

    FEDARTML_AVAILABLE = True
except ImportError:
    FEDARTML_AVAILABLE = False
    print("Warning: fedartml not available. Install with: pip install fedartml")

torch.backends.cudnn.benchmark = True

"""
Quantity Skew MinSize-Dirichlet Implementation for Split-Learning Framework

This module creates quantity-skewed data distributions from CIFAR-10 dataset using FedArtML's
MinSize-Dirichlet method. This method ensures a minimum size for each client while creating
quantity skew using the Dirichlet distribution.

Key characteristics:
- Different clients have different amounts of data
- All clients have the same class distribution (balanced)
- Uses MinSize-Dirichlet method for more controlled quantity skew
- Creates realistic federated learning scenarios with data size imbalance

Parameters:
- alpha_quant_split: Alpha parameter for quantity skew (smaller = more non-IID)
- num_clients: Total number of clients among which images are to be distributed
- batch_size: Loading of the data into the data loader by batches
"""


def get_cifar10():
    """Return CIFAR10 train/test data and labels as numpy arrays"""
    data_train = torchvision.datasets.CIFAR10("./data", train=True, download=True)
    data_test = torchvision.datasets.CIFAR10("./data", train=False, download=True)

    x_train, y_train = data_train.data.transpose((0, 3, 1, 2)), np.array(
        data_train.targets
    )
    x_test, y_test = data_test.data.transpose((0, 3, 1, 2)), np.array(data_test.targets)

    return x_train, y_train, x_test, y_test


def print_image_data_stats(data_train, labels_train, data_test, labels_test):
    print("\nData: ")
    print(
        " - Train Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
            data_train.shape,
            labels_train.shape,
            np.min(data_train),
            np.max(data_train),
            np.min(labels_train),
            np.max(labels_train),
        )
    )
    print(
        " - Test Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
            data_test.shape,
            labels_test.shape,
            np.min(data_train),
            np.max(data_train),
            np.min(labels_test),
            np.max(labels_test),
        )
    )


def from_FedArtML_to_Flower_format(clients_dict):
    """
    Convert from SplitAsFederatedData function output (FedArtML) to list format.
    """
    # initialize list that contains clients (features and labels)
    list_x_train = []
    list_y_train = []

    # Get the name of the clients from the dictionary
    client_names = list(clients_dict.keys())

    # Iterate over each client
    for client in client_names:
        # Get data from each client
        each_client_train = np.array(clients_dict[client], dtype=object)

        # Extract features for each client
        feat = []
        x_tra = np.array(each_client_train[:, 0])
        for row in x_tra:
            feat.append(row)
        feat = np.array(feat)

        # Extract labels from each client
        y_tra = np.array(each_client_train[:, 1], dtype=np.int64)

        # Append in list features and labels
        list_x_train.append(feat)
        list_y_train.append(y_tra)

    return list_x_train, list_y_train


def create_quantity_skew_minsize_dirichlet_with_fedartml(
    data, labels, n_clients, alpha_quant_split=1.0, verbose=True
):
    """
    Create quantity skew using FedArtML library with minsize-dirichlet method.

    Parameters:
    - data: Input images
    - labels: Corresponding labels
    - n_clients: Number of clients
    - alpha_quant_split: Alpha parameter for quantity skew (smaller = more non-IID)
    - verbose: Print distribution information
    """
    if not FEDARTML_AVAILABLE:
        print("Error: fedartml library not available. Using fallback method.")
        list_x_train, list_y_train = create_fallback_quantity_skew(
            data, labels, n_clients, verbose
        )
        # Return dummy distances for fallback
        distances = None
        return list_x_train, list_y_train, distances

    print(
        f"Creating quantity skew using FedArtML with minsize-dirichlet method, alpha_quant_split: {alpha_quant_split}"
    )

    # Instantiate SplitAsFederatedData object
    my_federater = SplitAsFederatedData(random_state=0)

    # Create federated dataset with quantity skew using minsize-dirichlet method
    clients_glob_dic, list_ids_sampled_dic, miss_class_per_node, distances = (
        my_federater.create_clients(
            image_list=data,
            label_list=labels,
            num_clients=n_clients,
            prefix_cli="Local_node",
            method="no-label-skew",  # No label skew
            quant_skew_method="minsize-dirichlet",  # Use minsize-dirichlet for quantity skew
            alpha_quant_split=alpha_quant_split,
        )
    )

    # Use without class completion
    clients_glob = clients_glob_dic["without_class_completion"]

    # Convert to our format
    list_x_train, list_y_train = from_FedArtML_to_Flower_format(
        clients_dict=clients_glob
    )

    # Print quantity skew distances
    if verbose:
        print("\nQuantity Skew Distances (MinSize-Dirichlet):")
        JSD_glob_quant = distances["without_class_completion_quant"]["jensen-shannon"]
        print(f"Jensen-Shannon distance: {JSD_glob_quant}")
        HD_glob_quant = distances["without_class_completion_quant"]["hellinger"]
        print(f"Hellinger distance: {HD_glob_quant}")
        EMD_glob_quant = distances["without_class_completion_quant"]["earth-movers"]
        print(f"Earth Mover's distance: {EMD_glob_quant}")

        # Print feature skew distances (if available)
        if "without_class_completion_feat" in distances:
            print("\nFeature Skew Distances:")
            JSD_glob_feat = distances["without_class_completion_feat"]["jensen-shannon"]
            print(f"JSD_glob_feat: {JSD_glob_feat}")
            HD_glob_feat = distances["without_class_completion_feat"]["hellinger"]
            print(f"HD_glob_feat: {HD_glob_feat}")
            EMD_glob_feat = distances["without_class_completion_feat"]["earth-movers"]
            print(f"EMD_glob_feat: {EMD_glob_feat}")

        print("\nQuantity Distribution (MinSize-Dirichlet method):")
        for i, (x, y) in enumerate(zip(list_x_train, list_y_train)):
            if len(y) > 0:
                unique_classes, counts = np.unique(y, return_counts=True)
                class_distribution = dict(zip(unique_classes, counts))
                print(f" - Client {i}: {len(y)} samples, Classes: {class_distribution}")
            else:
                print(f" - Client {i}: 0 samples")
        print()

    return list_x_train, list_y_train, distances


def create_fallback_quantity_skew(data, labels, n_clients, verbose=True):
    """
    Fallback method for quantity skew when FedArtML is not available.
    Creates a simple quantity-skewed split.
    """
    print("Using fallback quantity skew method (simple size-based split)")

    n_samples = len(data)

    # Create quantity-skewed splits using different sizes
    # Use exponential distribution to create different sizes
    sizes = np.random.exponential(scale=n_samples / (n_clients * 2), size=n_clients)
    sizes = sizes / sizes.sum() * n_samples  # Normalize to total samples
    sizes = sizes.astype(int)

    # Adjust to ensure we use all samples
    diff = n_samples - sizes.sum()
    sizes[-1] += diff  # Add remaining samples to last client

    list_x_train = []
    list_y_train = []

    start_idx = 0
    for i, size in enumerate(sizes):
        end_idx = start_idx + size

        if end_idx > n_samples:
            end_idx = n_samples

        if start_idx < n_samples:
            client_x = data[start_idx:end_idx]
            client_y = labels[start_idx:end_idx]
        else:
            client_x = np.empty((0,) + data.shape[1:], dtype=data.dtype)
            client_y = np.empty(0, dtype=labels.dtype)

        list_x_train.append(client_x)
        list_y_train.append(client_y)

        start_idx = end_idx

    if verbose:
        print("\nFallback Quantity Skew Distribution:")
        for i, (x, y) in enumerate(zip(list_x_train, list_y_train)):
            if len(y) > 0:
                unique_classes, counts = np.unique(y, return_counts=True)
                class_distribution = dict(zip(unique_classes, counts))
                print(f" - Client {i}: {len(y)} samples, Classes: {class_distribution}")
            else:
                print(f" - Client {i}: 0 samples")
        print()

    return list_x_train, list_y_train


def get_default_data_transforms(train=True, verbose=True):
    """Get default data transformations for CIFAR-10"""
    transforms_train = {
        "cifar10": transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
    }
    transforms_eval = {
        "cifar10": transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
    }

    if verbose:
        print("\nData preprocessing: ")
        for transformation in transforms_train["cifar10"].transforms:
            print(" -", transformation)
        print()

    return (transforms_train["cifar10"], transforms_eval["cifar10"])


def get_data_loaders_quantity_skew_minsize_dirichlet(
    nclients, batch_size, alpha_quant_split=1.0, verbose=True
):
    """
    Create data loaders with quantity skew using FedArtML's minsize-dirichlet method.
    Returns list of client DataLoaders, test DataLoader, and distances.
    """
    x_train, y_train, x_test, y_test = get_cifar10()

    if verbose:
        print_image_data_stats(x_train, y_train, x_test, y_test)

    transforms_train, transforms_eval = get_default_data_transforms(verbose=False)

    # Create quantity-skewed split using FedArtML minsize-dirichlet method
    list_x_train, list_y_train, distances = (
        create_quantity_skew_minsize_dirichlet_with_fedartml(
            x_train, y_train, nclients, alpha_quant_split, verbose
        )
    )

    # Create DataLoaders for each client
    client_loaders = []
    for x, y in zip(list_x_train, list_y_train):
        if len(x) > 0:  # Only create loader if client has data
            dataset = custom_image_dataset.CustomImageDataset(x, y, transforms_train)
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True
            )
            client_loaders.append(loader)
        else:
            # Create empty loader
            empty_dataset = custom_image_dataset.CustomImageDataset(
                np.empty((0, 3, 32, 32)), np.empty(0), transforms_train
            )
            loader = torch.utils.data.DataLoader(
                empty_dataset, batch_size=batch_size, shuffle=True
            )
            client_loaders.append(loader)

    # Create test loader
    test_loader = torch.utils.data.DataLoader(
        custom_image_dataset.CustomImageDataset(x_test, y_test, transforms_eval),
        batch_size=batch_size,
        shuffle=False,
    )

    return client_loaders, test_loader, distances


def run(alpha_quant_split, num_clients, batch_size):
    """
    Main function to create quantity-skewed federated data using FedArtML's minsize-dirichlet method.
    This function follows the same interface as other modules for compatibility.

    Parameters:
    - alpha_quant_split: Alpha parameter for quantity skew (smaller = more non-IID)
    - num_clients: Number of clients
    - batch_size: Batch size for DataLoaders
    """
    print(
        f"Creating quantity-skewed data for {num_clients} clients using FedArtML minsize-dirichlet method..."
    )
    print(f"Alpha for quantity split: {alpha_quant_split}")

    # Create data loaders with quantity skew using FedArtML minsize-dirichlet method
    train_loader, test_loader, distances = (
        get_data_loaders_quantity_skew_minsize_dirichlet(
            nclients=num_clients,
            batch_size=batch_size,
            alpha_quant_split=alpha_quant_split,
            verbose=False,  # Set to False to avoid duplicate printing
        )
    )

    # Save to pickle file (same format as other modules)
    with open("output.pickle", "wb") as handle:
        pickle.dump(train_loader, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Print statistics
    for i in range(num_clients):
        print(f"Client {i} loader length: {len(train_loader[i])}")
    print(f"Test loader length: {len(test_loader)}")

    # Print quantity skew distances after data generation
    if distances and "without_class_completion_quant" in distances:
        JSD_glob_quant = distances["without_class_completion_quant"]["jensen-shannon"]
        HD_glob_quant = distances["without_class_completion_quant"]["hellinger"]
        EMD_glob_quant = distances["without_class_completion_quant"]["earth-movers"]

        print("\n" + "=" * 50)
        print("QUANTITY SKEW DISTANCES AFTER DATA GENERATION:")
        print("=" * 50)
        print(f"JSD_glob_quant: {JSD_glob_quant}")
        print(f"HD_glob_quant: {HD_glob_quant}")
        print(f"EMD_glob_quant: {EMD_glob_quant}")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("QUANTITY SKEW DISTANCES AFTER DATA GENERATION:")
        print("=" * 50)
        print("JSD_glob_quant: N/A (distances not available)")
        print("HD_glob_quant: N/A (distances not available)")
        print("EMD_glob_quant: N/A (distances not available)")
        print("=" * 50)

    return train_loader, test_loader


# Example usage (commented out for compatibility)
# if __name__ == "__main__":
#     import argparse
#
#     parser = argparse.ArgumentParser(
#         description="Create quantity-skewed federated data using FedArtML minsize-dirichlet method"
#     )
#     parser.add_argument(
#         "--alpha_quant_split", type=float, default=1.0,
#         help="Alpha parameter for quantity skew (default: 1.0, smaller = more non-IID)"
#     )
#     parser.add_argument(
#         "--num_clients", type=int, default=3,
#         help="Number of clients (default: 3)"
#     )
#     parser.add_argument(
#         "--batch_size", type=int, default=128,
#         help="Batch size for DataLoaders (default: 128)"
#     )
#
#     args = parser.parse_args()
#
#     run(
#         alpha_quant_split=args.alpha_quant_split,
#         num_clients=args.num_clients,
#         batch_size=args.batch_size
#     )
