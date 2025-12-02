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
from PIL import Image

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
    data, labels, n_clients, alpha_quant_split=1.0, verbose=True, seed=None
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
    my_federater = SplitAsFederatedData(random_state=seed)

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


def run(data, alpha_quant_split, num_clients, seed=None):
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

    images, labels = zip(*data)

    # Convert labels to list of integers (handle numpy arrays and other types)
    # fedartml requires hashable types (integers), not numpy arrays
    labels = [
        int(label.item()) if isinstance(label, np.ndarray) else int(label)
        for label in labels
    ]

    list_x_train, list_y_train, distances = (
        create_quantity_skew_minsize_dirichlet_with_fedartml(
            images,
            labels,
            n_clients=num_clients,
            alpha_quant_split=alpha_quant_split,
            seed=seed,
        )
    )

    list_x_train_pil = [
        [Image.fromarray(img.astype("uint8")) for img in client_imgs]
        for client_imgs in list_x_train
    ]

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

    return [list(zip(x, y)) for x, y in zip(list_x_train_pil, list_y_train)]


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
