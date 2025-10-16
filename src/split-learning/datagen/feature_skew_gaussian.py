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
Feature Skew Gaussian Noise Implementation for Split-Learning Framework

This module creates feature-skewed data distributions from CIFAR-10 dataset using FedArtML's
Gaussian Noise method. This method adds Gaussian noise to features to create feature skew
while maintaining the same class distribution across clients.

Key characteristics:
- Different clients have different feature distributions due to added Gaussian noise
- All clients have the same class distribution (balanced)
- Uses Gaussian noise method for controlled feature skew
- Creates realistic federated learning scenarios with feature heterogeneity

Parameters:
- sigma_noise: Standard deviation of Gaussian noise (higher = more non-IID)
- num_clients: Total number of clients among which images are to be distributed
- n_bins: Number of bins for histogram calculation (default: 'n_samples')
- feat_sample_rate: Proportion of features to sample when measuring feature skew
"""


def convert_data_to_numpy(data):
    
    x_list = []
    y_list = []
    for image, label in data:
        if isinstance(image, torch.Tensor):
            image_np = image.numpy()
        elif hasattr(image, 'convert'):
            image_np = np.array(image)
        elif isinstance(image, np.ndarray):
            image_np = image

        x_list.append(image_np)
        y_list.append(label)
    
    x_train, y_train = np.stack(x_list), np.array(y_list)

    return zip(x_train, y_train)

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


def create_feature_skew_gaussian_with_fedartml(
    data,
    labels,
    n_clients,
    sigma_noise=2.0,
    n_bins="n_samples",
    feat_sample_rate=0.1,
    verbose=True,
    seed=None
):
    """
    Create feature skew using FedArtML library with Gaussian noise method.

    Parameters:
    - data: Input images
    - labels: Corresponding labels
    - n_clients: Number of clients
    - sigma_noise: Standard deviation of Gaussian noise (higher = more non-IID)
    - n_bins: Number of bins for histogram calculation
    - feat_sample_rate: Proportion of features to sample when measuring feature skew
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
        f"Creating feature skew using FedArtML with Gaussian noise method, sigma_noise: {sigma_noise}"
    )

    # Instantiate SplitAsFederatedData object
    my_federater = SplitAsFederatedData(random_state=seed)

    # Create federated dataset with feature skew using Gaussian noise method
    clients_glob_dic, list_ids_sampled_dic, miss_class_per_node, distances = (
        my_federater.create_clients(
            image_list=data,
            label_list=labels,
            num_clients=n_clients,
            prefix_cli="Local_node",
            feat_skew_method="gaussian-noise",  # Use Gaussian noise for feature skew
            sigma_noise=sigma_noise,
            bins=n_bins,
            feat_sample_rate=feat_sample_rate,
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
        print("\nFeature Skew Distances (Gaussian Noise):")
        JSD_glob_feat = distances["without_class_completion_feat"]["jensen-shannon"]
        print(f"JSD_glob_feat: {JSD_glob_feat}")
        HD_glob_feat = distances["without_class_completion_feat"]["hellinger"]
        print(f"HD_glob_feat: {HD_glob_feat}")
        EMD_glob_feat = distances["without_class_completion_feat"]["earth-movers"]
        print(f"EMD_glob_feat: {EMD_glob_feat}")

        print("\nFeature Skew Distribution (Gaussian Noise method):")
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
    Creates a simple feature-skewed split by adding random noise.
    """
    print("Using fallback feature skew method (simple noise-based split)")

    n_samples = len(data)
    samples_per_client = n_samples // n_clients

    list_x_train = []
    list_y_train = []

    for i in range(n_clients):
        start_idx = i * samples_per_client
        if i == n_clients - 1:
            end_idx = n_samples
        else:
            end_idx = (i + 1) * samples_per_client

        client_x = data[start_idx:end_idx].copy()
        client_y = labels[start_idx:end_idx].copy()

        # Add different amounts of noise to each client for feature skew
        noise_std = 0.1 * (i + 1)  # Different noise levels for each client
        noise = np.random.normal(0, noise_std, client_x.shape)
        client_x = client_x + noise
        client_x = np.clip(client_x, 0, 1)  # Clip to valid range

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




def run(data, sigma_noise, num_clients, n_bins="n_samples", feat_sample_rate=0.1, seed=None):
    """
    Main function to create feature-skewed federated data using FedArtML's Gaussian noise method.
    This function follows the same interface as other modules for compatibility.

    Parameters:
    - sigma_noise: Standard deviation of Gaussian noise (higher = more non-IID)
    - num_clients: Number of clients
    - batch_size: Batch size for DataLoaders
    - n_bins: Number of bins for histogram calculation
    - feat_sample_rate: Proportion of features to sample when measuring feature skew
    """
    print(
        f"Creating feature-skewed data for {num_clients} clients using FedArtML Gaussian noise method..."
    )
    print(f"Sigma noise: {sigma_noise}")

    data = convert_data_to_numpy(data)

    images, labels = zip(*data)

    list_x_train, list_y_train, distances = create_feature_skew_gaussian_with_fedartml(
        images, 
        labels, 
        num_clients,
        sigma_noise=sigma_noise,
        n_bins=n_bins,
        feat_sample_rate=feat_sample_rate,
        seed=seed
    )

    list_x_train_pil = [[Image.fromarray(img.astype('uint8')) 
                        for img in client_imgs] for client_imgs in list_x_train]


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
#         description="Create feature-skewed federated data using FedArtML Gaussian noise method"
#     )
#     parser.add_argument(
#         "--sigma_noise", type=float, default=2.0,
#         help="Standard deviation of Gaussian noise (default: 2.0, higher = more non-IID)"
#     )
#     parser.add_argument(
#         "--num_clients", type=int, default=3,
#         help="Number of clients (default: 3)"
#     )
#     parser.add_argument(
#         "--batch_size", type=int, default=128,
#         help="Batch size for DataLoaders (default: 128)"
#     )
#     parser.add_argument(
#         "--n_bins", type=str, default='n_samples',
#         help="Number of bins for histogram calculation (default: 'n_samples')"
#     )
#     parser.add_argument(
#         "--feat_sample_rate", type=float, default=0.1,
#         help="Proportion of features to sample when measuring feature skew (default: 0.1)"
#     )
#
#     args = parser.parse_args()
#
#     run(
#         sigma_noise=args.sigma_noise,
#         num_clients=args.num_clients,
#         batch_size=args.batch_size,
#         n_bins=args.n_bins,
#         feat_sample_rate=args.feat_sample_rate
#     )
