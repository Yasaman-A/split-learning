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

# Import FedArtML library for label skew
try:
    from fedartml import SplitAsFederatedData

    FEDARTML_AVAILABLE = True
except ImportError:
    FEDARTML_AVAILABLE = False
    print("Warning: fedartml not available. Install with: pip install fedartml")

torch.backends.cudnn.benchmark = True

"""
Label Skew Implementation for Split-Learning Framework

This module creates label-skewed data distributions from CIFAR-10 dataset using FedArtML.
Label skew occurs when clients have different class distributions, creating heterogeneity
in the labels available to each client.

Key characteristics:
- Different clients have different class distributions
- Some clients may have only certain classes
- Creates realistic federated learning scenarios with class imbalance

Parameters:
- alpha_label_split: Alpha parameter for label skew (smaller = more non-IID)
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


def create_label_skew_with_fedartml(
    data, labels, n_clients, alpha_label_split=1.0, verbose=True, seed=None
):
    """
    Create label skew using FedArtML library with dirichlet method.

    Parameters:
    - data: Input images
    - labels: Corresponding labels
    - n_clients: Number of clients
    - alpha_label_split: Alpha parameter for label skew (smaller = more non-IID)
    - verbose: Print distribution information
    """
    if not FEDARTML_AVAILABLE:
        print("Error: fedartml library not available. Using fallback method.")
        list_x_train, list_y_train = create_fallback_label_skew(
            data, labels, n_clients, verbose
        )
        # Return dummy distances for fallback
        distances = None
        return list_x_train, list_y_train, distances

    print(
        f"Creating label skew using FedArtML with alpha_label_split: {alpha_label_split}"
    )

    # Instantiate SplitAsFederatedData object
    my_federater = SplitAsFederatedData(random_state=seed)

    # Create federated dataset with label skew using dirichlet method
    clients_glob_dic, list_ids_sampled_dic, miss_class_per_node, distances = (
        my_federater.create_clients(
            image_list=data,
            label_list=labels,
            num_clients=n_clients,
            prefix_cli="Local_node",
            method="dirichlet",  # Use dirichlet for label skew
            alpha=alpha_label_split,  # Use alpha parameter for dirichlet
        )
    )

    # Use without class completion
    clients_glob = clients_glob_dic["without_class_completion"]

    # Con
    # ert to our format
    list_x_train, list_y_train = from_FedArtML_to_Flower_format(
        clients_dict=clients_glob
    )

    # Print label skew distances
    if verbose:
        if distances and "without_class_completion_label" in distances:
            print("\nLabel Skew Distances:")
            JSD_glob_label = distances["without_class_completion_label"][
                "jensen-shannon"
            ]
            print(f"Jensen-Shannon distance: {JSD_glob_label}")
            HD_glob_label = distances["without_class_completion_label"]["hellinger"]
            print(f"Hellinger distance: {HD_glob_label}")
            EMD_glob_label = distances["without_class_completion_label"]["earth-movers"]
            print(f"Earth Mover's distance: {EMD_glob_label}")

        # Print feature skew distances (if available)
        if "without_class_completion_feat" in distances:
            print("\nFeature Skew Distances:")
            JSD_glob_feat = distances["without_class_completion_feat"]["jensen-shannon"]
            print(f"JSD_glob_feat: {JSD_glob_feat}")
            HD_glob_feat = distances["without_class_completion_feat"]["hellinger"]
            print(f"HD_glob_feat: {HD_glob_feat}")
            EMD_glob_feat = distances["without_class_completion_feat"]["earth-movers"]
            print(f"EMD_glob_feat: {EMD_glob_feat}")

        print("\nLabel Distribution (should be skewed for label skew):")
        for i, (x, y) in enumerate(zip(list_x_train, list_y_train)):
            if len(y) > 0:
                unique_classes, counts = np.unique(y, return_counts=True)
                class_distribution = dict(zip(unique_classes, counts))
                print(f" - Client {i}: {len(y)} samples, Classes: {class_distribution}")
            else:
                print(f" - Client {i}: 0 samples")
        print()

    return list_x_train, list_y_train, distances


def create_fallback_label_skew(data, labels, n_clients, verbose=True):
    """
    Fallback method for label skew when FedArtML is not available.
    Creates a simple label-skewed split.
    """
    print("Using fallback label skew method (simple class-based split)")

    n_samples = len(data)
    n_classes = len(np.unique(labels))

    # Create label-skewed splits
    list_x_train = []
    list_y_train = []

    # Assign different classes to different clients
    classes_per_client = max(1, n_classes // n_clients)

    for i in range(n_clients):
        # Select classes for this client
        start_class = i * classes_per_client
        if i == n_clients - 1:  # Last client gets remaining classes
            end_class = n_classes
        else:
            end_class = min((i + 1) * classes_per_client, n_classes)

        client_classes = list(range(start_class, end_class))

        # Get samples for these classes
        client_indices = []
        for class_label in client_classes:
            class_indices = np.where(labels == class_label)[0]
            client_indices.extend(class_indices)

        if client_indices:
            client_indices = np.array(client_indices)
            client_x = data[client_indices]
            client_y = labels[client_indices]
        else:
            client_x = np.empty((0,) + data.shape[1:], dtype=data.dtype)
            client_y = np.empty(0, dtype=labels.dtype)

        list_x_train.append(client_x)
        list_y_train.append(client_y)

    if verbose:
        print("\nFallback Label Skew Distribution:")
        for i, (x, y) in enumerate(zip(list_x_train, list_y_train)):
            if len(y) > 0:
                unique_classes, counts = np.unique(y, return_counts=True)
                class_distribution = dict(zip(unique_classes, counts))
                print(f" - Client {i}: {len(y)} samples, Classes: {class_distribution}")
            else:
                print(f" - Client {i}: 0 samples")
        print()

    return list_x_train, list_y_train


def run(data, alpha_label_split, num_clients, seed=None):
    """
    Main function to create label-skewed federated data using FedArtML.
    This function follows the same interface as non_iid.py for compatibility.

    Parameters:
    - alpha_label_split: Alpha parameter for label skew (smaller = more non-IID)
    - num_clients: Number of clients
    - batch_size: Batch size for DataLoaders
    """
    print(f"Creating label-skewed data for {num_clients} clients using FedArtML...")
    print(f"Alpha for label split: {alpha_label_split}")

    images, labels = zip(*data)

    list_x_train, list_y_train, distances = create_label_skew_with_fedartml(
        images, labels, num_clients, alpha_label_split=alpha_label_split, seed=seed
    )

    list_x_train_pil = [
        [Image.fromarray(img.astype("uint8")) for img in client_imgs]
        for client_imgs in list_x_train
    ]

    # Print label skew distances after data generation
    if distances and "without_class_completion" in distances:
        JSD_glob_label = distances["without_class_completion"]["jensen-shannon"]
        HD_glob_label = distances["without_class_completion"]["hellinger"]
        EMD_glob_label = distances["without_class_completion"]["earth-movers"]

        print("\n" + "=" * 50)
        print("LABEL SKEW DISTANCES AFTER DATA GENERATION:")
        print("=" * 50)
        print(f"JSD_glob_label: {JSD_glob_label}")
        print(f"HD_glob_label: {HD_glob_label}")
        print(f"EMD_glob_label: {EMD_glob_label}")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("LABEL SKEW DISTANCES AFTER DATA GENERATION:")
        print("=" * 50)
        print("JSD_glob_label: N/A (distances not available)")
        print("HD_glob_label: N/A (distances not available)")
        print("EMD_glob_label: N/A (distances not available)")
        print("=" * 50)

    return [list(zip(x, y)) for x, y in zip(list_x_train_pil, list_y_train)]


# Example usage (commented out for compatibility)
# if __name__ == "__main__":
#     import argparse
#
#     parser = argparse.ArgumentParser(
#         description="Create label-skewed federated data using FedArtML"
#     )
#     parser.add_argument(
#         "--alpha_label_split", type=float, default=1.0,
#         help="Alpha parameter for label skew (default: 1.0, smaller = more non-IID)"
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
#         alpha_label_split=args.alpha_label_split,
#         num_clients=args.num_clients,
#         batch_size=args.batch_size
#     )
