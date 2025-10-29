#!/usr/bin/env python3
"""
Visualization script for split-learning data distribution.
Reads pickle files and visualizes how data is distributed among clients.
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from pathlib import Path


def load_pickle_data(file_path):
    """Load data from pickle file."""
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return None


def visualize_client_distribution(data, output_name="output", selected_clients=None):
    """Visualize data distribution across clients."""

    # Set up the plotting style
    plt.style.use("default")

    # Extract client data - data is a list of client datasets
    num_clients = len(data)
    num_classes = 10  # CIFAR-10 has 10 classes
    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    # Filter clients if specified
    if selected_clients is not None:
        print(f"Visualizing only clients: {selected_clients}")
        client_indices = selected_clients
    else:
        client_indices = list(range(num_clients))

    # Prepare data for visualization
    client_data = []
    for i, client_id in enumerate(client_indices):
        if client_id < len(data):
            client_samples = data[client_id]
            # Count samples per class
            class_counts = [0] * num_classes
            for sample in client_samples:
                if len(sample) >= 2:  # Assuming (features, label) format
                    label = sample[1]
                    if 0 <= label < num_classes:
                        class_counts[label] += 1
            client_data.append(class_counts)
            print(
                f"Client {client_id}: {sum(class_counts)} total samples, class distribution: {class_counts}"
            )
        else:
            print(f"Client {client_id} data not found")
            client_data.append([0] * num_classes)

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Data Distribution Analysis: {output_name}", fontsize=16, fontweight="bold"
    )

    # 1. Bar chart showing total samples per client
    ax1 = axes[0, 0]
    total_samples = [sum(client) for client in client_data]
    bars1 = ax1.bar(
        range(len(client_data)),
        total_samples,
        color="skyblue",
        edgecolor="navy",
        alpha=0.7,
    )
    ax1.set_xlabel("Client ID")
    ax1.set_ylabel("Total Samples")
    ax1.set_title("Total Samples per Client")
    ax1.set_xticks(range(len(client_data)))
    ax1.set_xticklabels([f"Client {client_id}" for client_id in client_indices])

    # Add value labels on bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 50,
            f"{int(height)}",
            ha="center",
            va="bottom",
        )

    # 2. Heatmap showing class distribution across clients
    ax2 = axes[0, 1]
    data_matrix = np.array(client_data)
    im = ax2.imshow(data_matrix, cmap="YlOrRd", aspect="auto")
    ax2.set_xlabel("Class")
    ax2.set_ylabel("Client ID")
    ax2.set_title("Class Distribution Heatmap")
    ax2.set_xticks(range(num_classes))
    ax2.set_xticklabels([f"C{i}" for i in range(num_classes)])
    ax2.set_yticks(range(len(client_data)))
    ax2.set_yticklabels([f"Client {client_id}" for client_id in client_indices])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label("Number of Samples")

    # 3. Stacked bar chart showing class distribution
    ax3 = axes[1, 0]
    bottom = np.zeros(len(client_data))
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    for class_id in range(num_classes):
        class_counts = [
            client_data[client_id][class_id] for client_id in range(len(client_data))
        ]
        ax3.bar(
            range(len(client_data)),
            class_counts,
            bottom=bottom,
            label=f"Class {class_id}",
            color=colors[class_id],
            alpha=0.8,
        )
        bottom += class_counts

    ax3.set_xlabel("Client ID")
    ax3.set_ylabel("Number of Samples")
    ax3.set_title("Class Distribution per Client (Stacked)")
    ax3.set_xticks(range(len(client_data)))
    ax3.set_xticklabels([f"Client {client_id}" for client_id in client_indices])
    ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # 4. Class balance analysis
    ax4 = axes[1, 1]
    class_totals = np.sum(data_matrix, axis=0)
    bars4 = ax4.bar(
        range(num_classes),
        class_totals,
        color="lightcoral",
        edgecolor="darkred",
        alpha=0.7,
    )
    ax4.set_xlabel("Class")
    ax4.set_ylabel("Total Samples")
    ax4.set_title("Total Samples per Class (Global)")
    ax4.set_xticks(range(num_classes))
    ax4.set_xticklabels([f"C{i}" for i in range(num_classes)])

    # Add value labels
    for i, bar in enumerate(bars4):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 50,
            f"{int(height)}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    # Save the plot
    output_file = f"{output_name}_distribution_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Visualization saved as: {output_file}")

    # Close the plot to free memory
    plt.close()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("DATA DISTRIBUTION SUMMARY")
    print("=" * 60)

    for i, client_id in enumerate(client_indices):
        total = sum(client_data[i])
        print(f"Client {client_id}: {total} total samples")
        print(f"  Class distribution: {client_data[i]}")
        print(
            f"  Most common class: {np.argmax(client_data[i])} ({max(client_data[i])} samples)"
        )
        print(
            f"  Least common class: {np.argmin(client_data[i])} ({min(client_data[i])} samples)"
        )
        print()

    # Calculate and print imbalance metrics
    print("IMBALANCE METRICS:")
    print("-" * 30)

    # Calculate coefficient of variation for each class
    for class_id in range(num_classes):
        class_distribution = [
            client_data[client_id][class_id] for client_id in range(len(client_data))
        ]
        mean_samples = np.mean(class_distribution)
        std_samples = np.std(class_distribution)
        cv = std_samples / mean_samples if mean_samples > 0 else 0
        print(
            f"Class {class_id}: Mean={mean_samples:.1f}, Std={std_samples:.1f}, CV={cv:.3f}"
        )

    return client_data


def create_detailed_visualization(client_data, output_name="output"):
    """Create a more detailed visualization focusing on class imbalance."""

    num_clients = len(client_data)
    num_classes = 10

    # Create a large figure with multiple subplots
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    fig.suptitle(
        f"Detailed Data Distribution Analysis: {output_name}",
        fontsize=18,
        fontweight="bold",
    )

    # 1. Individual client class distributions
    ax1 = axes[0, 0]
    for client_id in range(num_clients):
        ax1.plot(
            range(num_classes),
            client_data[client_id],
            marker="o",
            label=f"Client {client_id}",
            linewidth=2,
        )
    ax1.set_xlabel("Class")
    ax1.set_ylabel("Number of Samples")
    ax1.set_title("Class Distribution per Client")
    ax1.set_xticks(range(num_classes))
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Box plot showing distribution of each class across clients
    ax2 = axes[0, 1]
    data_for_box = []
    labels_for_box = []
    for class_id in range(num_classes):
        class_distribution = [
            client_data[client_id][class_id] for client_id in range(num_clients)
        ]
        data_for_box.append(class_distribution)
        labels_for_box.append(f"C{class_id}")

    ax2.boxplot(data_for_box, labels=labels_for_box)
    ax2.set_xlabel("Class")
    ax2.set_ylabel("Number of Samples")
    ax2.set_title("Class Distribution Variability Across Clients")
    ax2.grid(True, alpha=0.3)

    # 3. Client size distribution
    ax3 = axes[1, 0]
    client_sizes = [sum(client_data[client_id]) for client_id in range(num_clients)]
    bars3 = ax3.bar(
        range(num_clients), client_sizes, color="lightblue", edgecolor="navy"
    )
    ax3.set_xlabel("Client ID")
    ax3.set_ylabel("Total Samples")
    ax3.set_title("Client Size Distribution")
    ax3.set_xticks(range(num_clients))

    # Add percentage labels
    total_samples = sum(client_sizes)
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        percentage = (height / total_samples) * 100
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 50,
            f"{percentage:.1f}%",
            ha="center",
            va="bottom",
        )

    # 4. Class imbalance heatmap
    ax4 = axes[1, 1]
    data_matrix = np.array(client_data)
    # Normalize by client size to show relative distribution
    normalized_matrix = data_matrix / np.sum(data_matrix, axis=1, keepdims=True)
    im = ax4.imshow(normalized_matrix, cmap="RdYlBu_r", aspect="auto")
    ax4.set_xlabel("Class")
    ax4.set_ylabel("Client ID")
    ax4.set_title("Normalized Class Distribution (Proportion)")
    ax4.set_xticks(range(num_classes))
    ax4.set_xticklabels([f"C{i}" for i in range(num_classes)])
    ax4.set_yticks(range(num_clients))
    ax4.set_yticklabels([f"Client {i}" for i in range(num_clients)])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label("Proportion of Client Data")

    # 5. Imbalance metrics
    ax5 = axes[2, 0]
    cv_values = []
    for class_id in range(num_classes):
        class_distribution = [
            client_data[client_id][class_id] for client_id in range(num_clients)
        ]
        mean_samples = np.mean(class_distribution)
        std_samples = np.std(class_distribution)
        cv = std_samples / mean_samples if mean_samples > 0 else 0
        cv_values.append(cv)

    bars5 = ax5.bar(
        range(num_classes), cv_values, color="orange", edgecolor="darkorange"
    )
    ax5.set_xlabel("Class")
    ax5.set_ylabel("Coefficient of Variation")
    ax5.set_title("Class Imbalance Across Clients (CV)")
    ax5.set_xticks(range(num_classes))
    ax5.set_xticklabels([f"C{i}" for i in range(num_classes)])
    ax5.grid(True, alpha=0.3)

    # 6. Cumulative distribution
    ax6 = axes[2, 1]
    for client_id in range(num_clients):
        cumulative = np.cumsum(client_data[client_id])
        ax6.plot(
            range(num_classes), cumulative, marker="o", label=f"Client {client_id}"
        )
    ax6.set_xlabel("Class")
    ax6.set_ylabel("Cumulative Samples")
    ax6.set_title("Cumulative Class Distribution per Client")
    ax6.set_xticks(range(num_classes))
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the detailed plot
    detailed_output = f"{output_name}_detailed_analysis.png"
    plt.savefig(detailed_output, dpi=300, bbox_inches="tight")
    print(f"Detailed visualization saved as: {detailed_output}")

    # Close the plot to free memory
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize split-learning data distribution"
    )
    parser.add_argument(
        "--pickle_file",
        type=str,
        default="output.pkl",
        help="Path to the pickle file to visualize",
    )
    parser.add_argument(
        "--output_name", type=str, default="output", help="Name for output files"
    )
    parser.add_argument(
        "--detailed", action="store_true", help="Create detailed visualizations"
    )
    parser.add_argument(
        "--clients",
        type=str,
        default=None,
        help="Comma-separated list of client IDs to visualize (e.g., '0,1,2' or '1,3,5')",
    )

    args = parser.parse_args()

    # Parse clients argument
    selected_clients = None
    if args.clients:
        try:
            selected_clients = [int(x.strip()) for x in args.clients.split(",")]
            print(f"Selected clients: {selected_clients}")
        except ValueError:
            print(
                "Error: Invalid client IDs. Use comma-separated integers (e.g., '0,1,2')"
            )
            return

    # Check if pickle file exists
    if not os.path.exists(args.pickle_file):
        print(f"Error: Pickle file '{args.pickle_file}' not found!")
        print("Available pickle files:")
        for pkl_file in Path(".").glob("*.pkl"):
            print(f"  - {pkl_file}")
        return

    print(f"Loading data from: {args.pickle_file}")
    data = load_pickle_data(args.pickle_file)

    if data is None:
        print("Failed to load data from pickle file.")
        return

    print(f"Data loaded successfully. Found {len(data)} clients.")

    # Extract pickle file name (without extension) for output naming
    pickle_name = os.path.splitext(os.path.basename(args.pickle_file))[0]
    if args.output_name == "output":  # Use pickle name if default output name
        output_name = pickle_name
    else:
        output_name = args.output_name

    # Create basic visualization
    print("Creating basic visualization...")
    client_data = visualize_client_distribution(data, output_name, selected_clients)

    # Create detailed visualization if requested
    if args.detailed:
        print("Creating detailed visualization...")
        create_detailed_visualization(client_data, output_name)

    print("Visualization complete!")


if __name__ == "__main__":
    main()
