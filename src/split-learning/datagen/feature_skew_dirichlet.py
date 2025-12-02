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

# Import FedArtML library for feature skew
try:
    from fedartml import SplitAsFederatedData

    FEDARTML_AVAILABLE = True
except ImportError:
    FEDARTML_AVAILABLE = False
    print("Warning: fedartml not available. Install with: pip install fedartml")

torch.backends.cudnn.benchmark = True

"""
Feature Skew Implementation for Split-Learning Framework

This module creates feature-skewed data distributions from CIFAR-10 dataset.
Feature skew occurs when clients have the same classes but with different 
feature distributions (e.g., different image styles, lighting, or characteristics).

Key differences from label skew:
- All clients have access to all classes
- Feature distributions vary across clients
- Creates heterogeneity in input characteristics while maintaining label balance

Parameters:
- alpha_feat_split: Alpha parameter for feature skew (smaller = more non-IID)
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


def create_feature_skew_with_fedartml(
    data, labels, n_clients, alpha_feat_split=1.0, verbose=True, seed=None
):
    """
    Create feature skew using FedArtML library with hist-dirichlet method.

    Parameters:
    - data: Input images
    - labels: Corresponding labels
    - n_clients: Number of clients
    - alpha_feat_split: Alpha parameter for feature skew (smaller = more non-IID)
    - verbose: Print distribution information
    """
    if not FEDARTML_AVAILABLE:
        print("Error: fedartml library not available. Using fallback method.")
        list_x_train, list_y_train = create_fallback_feature_skew(
            data, labels, n_clients, verbose
        )
        # Return dummy distances for fallback
        distances = None
        return list_x_train, list_y_train, distances

    print(
        f"Creating feature skew using FedArtML with alpha_feat_split: {alpha_feat_split}"
    )

    # Instantiate SplitAsFederatedData object
    my_federater = SplitAsFederatedData(random_state=seed)

    # Create federated dataset with feature skew using hist-dirichlet method
    clients_glob_dic, list_ids_sampled_dic, miss_class_per_node, distances = (
        my_federater.create_clients(
            image_list=data,
            label_list=labels,
            num_clients=n_clients,
            prefix_cli="Local_node",
            method="no-label-skew",  # No label skew
            feat_skew_method="hist-dirichlet",  # Use hist-dirichlet for feature skew
            alpha_feat_split=alpha_feat_split,
        )
    )

    # Use without class completion
    clients_glob = clients_glob_dic["without_class_completion"]

    # Convert to our format
    list_x_train, list_y_train = from_FedArtML_to_Flower_format(
        clients_dict=clients_glob
    )

    # Print feature skew distances
    if verbose:
        print("\nFeature Skew Distances:")
        JSD_glob_feat = distances["without_class_completion_feat"]["jensen-shannon"]
        print(f"JSD_glob_feat: {JSD_glob_feat}")
        HD_glob_feat = distances["without_class_completion_feat"]["hellinger"]
        print(f"HD_glob_feat: {HD_glob_feat}")
        EMD_glob_feat = distances["without_class_completion_feat"]["earth-movers"]
        print(f"EMD_glob_feat: {EMD_glob_feat}")

        print("\nLabel Distribution (should be balanced for feature skew):")
        for i, (x, y) in enumerate(zip(list_x_train, list_y_train)):
            if len(y) > 0:
                unique_classes, counts = np.unique(y, return_counts=True)
                class_distribution = dict(zip(unique_classes, counts))
                print(f" - Client {i}: {len(y)} samples, Classes: {class_distribution}")
            else:
                print(f" - Client {i}: 0 samples")
        print()

    return list_x_train, list_y_train, distances


def create_fallback_feature_skew(data, labels, n_clients, verbose=True):
    """
    Fallback method for feature skew when FedArtML is not available.
    Creates a simple balanced split with some randomization.
    """
    print("Using fallback feature skew method (balanced split with randomization)")

    n_samples = len(data)
    samples_per_client = n_samples // n_clients

    # Create balanced splits
    list_x_train = []
    list_y_train = []

    for i in range(n_clients):
        start_idx = i * samples_per_client
        if i == n_clients - 1:  # Last client gets remaining samples
            end_idx = n_samples
        else:
            end_idx = (i + 1) * samples_per_client

        client_x = data[start_idx:end_idx]
        client_y = labels[start_idx:end_idx]

        # Add some randomization to simulate feature skew
        if len(client_x) > 0:
            # Apply random transformations to simulate feature skew
            indices = np.random.permutation(len(client_x))
            client_x = client_x[indices]
            client_y = client_y[indices]

        list_x_train.append(client_x)
        list_y_train.append(client_y)

    if verbose:
        print("\nFallback Feature Skew Distribution:")
        for i, (x, y) in enumerate(zip(list_x_train, list_y_train)):
            if len(y) > 0:
                unique_classes, counts = np.unique(y, return_counts=True)
                class_distribution = dict(zip(unique_classes, counts))
                print(f" - Client {i}: {len(y)} samples, Classes: {class_distribution}")
            else:
                print(f" - Client {i}: 0 samples")
        print()

    return list_x_train, list_y_train


def run(data, alpha_feat_split, num_clients, seed):
    """
    Main function to create feature-skewed federated data using FedArtML.
    This function follows the same interface as non_iid.py for compatibility.

    Parameters:
    - alpha_feat_split: Alpha parameter for feature skew (smaller = more non-IID)
    - num_clients: Number of clients
    - batch_size: Batch size for DataLoaders
    """
    print(f"Creating feature-skewed data for {num_clients} clients using FedArtML...")
    print(f"Alpha for feature split: {alpha_feat_split}")

    images, labels = zip(*data)

    # Convert labels to list of integers (handle numpy arrays and other types)
    # fedartml requires hashable types (integers), not numpy arrays
    labels = [
        int(label.item()) if isinstance(label, np.ndarray) else int(label)
        for label in labels
    ]

    list_x_train, list_y_train, distances = create_feature_skew_with_fedartml(
        images, labels, num_clients, alpha_feat_split=alpha_feat_split, seed=seed
    )

    list_x_train_pil = [
        [Image.fromarray(img.astype("uint8")) for img in client_imgs]
        for client_imgs in list_x_train
    ]

    # Print feature skew distances after data generation
    if distances and "without_class_completion_feat" in distances:
        JSD_glob_feat = distances["without_class_completion_feat"]["jensen-shannon"]
        HD_glob_feat = distances["without_class_completion_feat"]["hellinger"]
        EMD_glob_feat = distances["without_class_completion_feat"]["earth-movers"]

        print("\n" + "=" * 50)
        print("FEATURE SKEW DISTANCES AFTER DATA GENERATION:")
        print("=" * 50)
        print(f"JSD_glob_feat: {JSD_glob_feat}")
        print(f"HD_glob_feat: {HD_glob_feat}")
        print(f"EMD_glob_feat: {EMD_glob_feat}")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("FEATURE SKEW DISTANCES AFTER DATA GENERATION:")
        print("=" * 50)
        print("JSD_glob_feat: N/A (distances not available)")
        print("HD_glob_feat: N/A (distances not available)")
        print("EMD_glob_feat: N/A (distances not available)")
        print("=" * 50)

    return [list(zip(x, y)) for x, y in zip(list_x_train_pil, list_y_train)]


# Example usage (commented out for compatibility)
# if __name__ == "__main__":
#     import argparse
#
#     parser = argparse.ArgumentParser(
#         description="Create feature-skewed federated data using FedArtML"
#     )
#     parser.add_argument(
#         "--alpha_feat_split", type=float, default=1.0,
#         help="Alpha parameter for feature skew (default: 1.0, smaller = more non-IID)"
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
#     run_feature_skew(
#         alpha_feat_split=args.alpha_feat_split,
#         num_clients=args.num_clients,
#         batch_size=args.batch_size
#     )
