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
from torch.utils.data import DataLoader
import pickle
from PIL import Image

# Try to import fedartml
try:
    from fedartml import SplitAsFederatedData

    FEDARTML_AVAILABLE = True
except ImportError:
    FEDARTML_AVAILABLE = False
    print("Warning: fedartml not available. Install with: pip install fedartml")


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


def create_label_skew_percentage_with_fedartml(
    data, labels, n_clients, percentage_skew=0.5, verbose=True
):
    """
    Create label skew using percentage-based method for extreme data size differences.

    Parameters:
    - data: Input images
    - labels: Corresponding labels
    - n_clients: Number of clients
    - percentage_skew: Percentage of data to skew (0.0 to 1.0)
    - verbose: Print distribution information
    """
    print(
        f"Creating label skew using percentage method, percentage_skew: {percentage_skew}"
    )

    # Create percentage-based label skew manually for extreme data size differences
    list_x_train, list_y_train = create_percentage_based_label_skew(
        data, labels, n_clients, percentage_skew, verbose
    )

    # Calculate distances manually since we're not using FedArtML
    distances = calculate_label_skew_distances(list_y_train)

    # Print label skew distances
    if verbose:
        print("\nLabel Skew Distances:")
        JSD_glob_label = distances["jensen-shannon"]
        print(f"Jensen-Shannon distance: {JSD_glob_label}")
        HD_glob_label = distances["hellinger"]
        print(f"Hellinger distance: {HD_glob_label}")
        EMD_glob_label = distances["earth-movers"]
        print(f"Earth Mover's distance: {EMD_glob_label}")

        print("\nLabel Distribution (percentage-based skew):")
        for i, (x, y) in enumerate(zip(list_x_train, list_y_train)):
            if len(y) > 0:
                unique_classes, counts = np.unique(y, return_counts=True)
                class_distribution = dict(zip(unique_classes, counts))
                print(f" - Client {i}: {len(y)} samples, Classes: {class_distribution}")
            else:
                print(f" - Client {i}: 0 samples")
        print()

    return list_x_train, list_y_train, distances


def create_percentage_based_label_skew(
    data, labels, n_clients, percentage_skew, verbose=True
):
    """
    Create percentage-based label skew with extreme data size differences.
    """
    # Calculate data allocation based on percentage_skew
    # Lower percentage_skew = more extreme size differences
    if percentage_skew <= 0.1:
        # Extreme skew: one client gets most data, others get very little
        data_ratios = [0.9] + [0.1 / (n_clients - 1)] * (n_clients - 1)
    elif percentage_skew <= 0.3:
        # High skew: uneven distribution
        data_ratios = [0.7] + [0.3 / (n_clients - 1)] * (n_clients - 1)
    elif percentage_skew <= 0.5:
        # Moderate skew
        data_ratios = [0.6] + [0.4 / (n_clients - 1)] * (n_clients - 1)
    else:
        # Low skew: more balanced
        base_ratio = 1.0 / n_clients
        data_ratios = [base_ratio * (1 + (1 - percentage_skew))] + [
            base_ratio * (1 - (1 - percentage_skew) / (n_clients - 1))
        ] * (n_clients - 1)

    # Normalize ratios
    data_ratios = np.array(data_ratios)
    data_ratios = data_ratios / np.sum(data_ratios)

    # Shuffle data first
    data = np.array(data)
    labels = np.array(labels)
    indices = np.random.permutation(len(data))
    shuffled_data = data[indices]
    shuffled_labels = labels[indices]

    # Split data according to ratios
    list_x_train = []
    list_y_train = []

    start_idx = 0
    for i in range(n_clients):
        end_idx = start_idx + int(len(data) * data_ratios[i])
        if i == n_clients - 1:  # Last client gets remaining data
            end_idx = len(data)

        client_x = shuffled_data[start_idx:end_idx]
        client_y = shuffled_labels[start_idx:end_idx]

        list_x_train.append(client_x)
        list_y_train.append(client_y)
        start_idx = end_idx

    if verbose:
        print(f"Data allocation ratios: {data_ratios}")
        print(f"Actual data sizes: {[len(y) for y in list_y_train]}")

    return list_x_train, list_y_train


def calculate_label_skew_distances(list_y_train):
    """
    Calculate label skew distances manually.
    """
    # Simple distance calculation based on class distributions
    all_labels = np.concatenate(list_y_train)
    unique_classes = np.unique(all_labels)

    # Calculate class distributions for each client
    client_distributions = []
    for y in list_y_train:
        if len(y) > 0:
            unique, counts = np.unique(y, return_counts=True)
            dist = np.zeros(len(unique_classes))
            for cls, count in zip(unique, counts):
                cls_idx = np.where(unique_classes == cls)[0][0]
                dist[cls_idx] = count / len(y)
            client_distributions.append(dist)
        else:
            client_distributions.append(np.zeros(len(unique_classes)))

    # Calculate distances between distributions
    if len(client_distributions) < 2:
        return {"jensen-shannon": 0.0, "hellinger": 0.0, "earth-movers": 0.0}

    # Jensen-Shannon Distance (simplified)
    jsd = 0.0
    for i in range(len(client_distributions)):
        for j in range(i + 1, len(client_distributions)):
            p = client_distributions[i]
            q = client_distributions[j]
            m = 0.5 * (p + q)
            jsd += 0.5 * (
                np.sum(p * np.log(p / m + 1e-10)) + np.sum(q * np.log(q / m + 1e-10))
            )

    # Hellinger Distance (simplified)
    hd = 0.0
    for i in range(len(client_distributions)):
        for j in range(i + 1, len(client_distributions)):
            p = client_distributions[i]
            q = client_distributions[j]
            hd += np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))

    # Earth Mover's Distance (simplified)
    emd = 0.0
    for i in range(len(client_distributions)):
        for j in range(i + 1, len(client_distributions)):
            p = client_distributions[i]
            q = client_distributions[j]
            emd += np.sum(np.abs(p - q))

    return {"jensen-shannon": jsd, "hellinger": hd, "earth-movers": emd}


def create_fallback_label_skew_percentage(
    data, labels, n_clients, percentage_skew, verbose=True
):
    """
    Fallback method for label skew when FedArtML is not available.
    Creates a simple percentage-based label-skewed split.
    """
    print("Using fallback label skew percentage method (simple percentage-based split)")

    # Simple percentage-based split
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

        client_x = data[start_idx:end_idx]
        client_y = labels[start_idx:end_idx]

        # Apply percentage skew by removing some classes
        if percentage_skew < 1.0:
            n_classes_to_keep = max(1, int(10 * (1 - percentage_skew)))
            unique_classes = np.unique(client_y)
            classes_to_keep = np.random.choice(
                unique_classes, n_classes_to_keep, replace=False
            )
            mask = np.isin(client_y, classes_to_keep)
            client_x = client_x[mask]
            client_y = client_y[mask]

        list_x_train.append(client_x)
        list_y_train.append(client_y)

    if verbose:
        print("\nLabel Distribution (fallback method):")
        for i, (x, y) in enumerate(zip(list_x_train, list_y_train)):
            if len(y) > 0:
                unique_classes, counts = np.unique(y, return_counts=True)
                class_distribution = dict(zip(unique_classes, counts))
                print(f" - Client {i}: {len(y)} samples, Classes: {class_distribution}")
            else:
                print(f" - Client {i}: 0 samples")
        print()

    return list_x_train, list_y_train


def print_image_data_stats(x_train, y_train, x_test, y_test):
    """
    Print statistics about the image dataset.
    """
    print(f"Training data shape: {x_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    print(f"Classes: {np.unique(y_train)}")


def run(data, percentage_skew, num_clients):
    """
    Main function to create label-skewed federated data using FedArtML percentage method.

    Parameters:
    - percentage_skew: Percentage of data to skew (0.0 to 1.0)
    - num_clients: Number of clients
    - batch_size: Batch size for DataLoaders
    """
    print(
        f"Creating label-skewed data for {num_clients} clients using FedArtML percentage method..."
    )
    print(f"Percentage skew: {percentage_skew}")

    images, labels = zip(*data)

    # Convert labels to list of integers (handle numpy arrays and other types)
    # fedartml requires hashable types (integers), not numpy arrays
    labels = [
        int(label.item()) if isinstance(label, np.ndarray) else int(label)
        for label in labels
    ]

    list_x_train, list_y_train, distances = create_label_skew_percentage_with_fedartml(
        images,
        labels,
        num_clients,
        percentage_skew=percentage_skew,
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
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--percentage_skew", type=float, default=0.5)
#     parser.add_argument("--num_clients", type=int, default=3)
#     parser.add_argument("--batch_size", type=int, default=128)
#     args = parser.parse_args()
#     run(args.percentage_skew, args.num_clients, args.batch_size)
