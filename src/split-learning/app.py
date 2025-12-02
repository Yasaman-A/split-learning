from .base_model import client as base_client
from .base_model import server as base_server
from .rr_multiclient import client as rr_multiclient_client
from .rr_multiclient import server as rr_multiclient_server
from .th_multiclient import client as th_multiclient_client
from .th_multiclient import server as th_multiclient_server
from .splitfed_v1 import client as splitfed_v1_client
from .splitfed_v1 import server as splitfed_v1_server
from .splitfed_v1 import fed_server as splitfed_v1_fed_server
from .splitfed_v1_custom_cut import client as splitfed_v1_custom_cut_client
from .splitfed_v1_custom_cut import server as splitfed_v1_custom_cut_server
from .splitfed_v1_custom_cut import fed_server as splitfed_v1_custom_cut_fed_server
from .splitfed_v2 import client as splitfed_v2_client
from .splitfed_v2 import server as splitfed_v2_server
from .splitfed_v2 import fed_server as splitfed_v2_fed_server
from .splitfed_v2_custom_cut import client as splitfed_v2_custom_cut_client
from .splitfed_v2_custom_cut import server as splitfed_v2_custom_cut_server
from .splitfed_v2_custom_cut import fed_server as splitfed_v2_custom_cut_fed_server

# from .non_iid import run
from .datagen.data_manager import (
    create_iid_dataset,
    create_non_iid_dataset,
    create_dirichlet_feature_skew,
    create_gaussian_feature_skew,
    create_dirichlet_label_skew,
    create_percentage_label_skew,
    create_quantity_skew_dirichlet,
    create_quantity_skew_minsize_dirichlet,
)
from .datagen.viz import visualize_client_distribution
import argparse


def get_short_output_name(
    data_type,
    alpha_feat_split=None,
    alpha_label_split=None,
    alpha_quant_split=None,
    sigma_noise=None,
    percentage_skew=None,
):
    """Generate short output names based on data type and parameters."""
    if data_type == "iid":
        return "iid"
    elif data_type == "non_iid":
        return "shard"
    elif data_type == "label_skew_dirichlet":
        return f"ld_{alpha_label_split}"
    elif data_type == "label_skew_percentage":
        return f"lp_{percentage_skew}"
    elif data_type == "feature_skew_dirichlet":
        return f"fd_{alpha_feat_split}"
    elif data_type == "feature_skew_gaussian":
        return f"fg_{sigma_noise}"
    elif data_type == "quantity_skew_dirichlet":
        return f"qm_{alpha_quant_split}"
    elif data_type == "quantity_skew_minsize_dirichlet":
        return f"qmd_{alpha_quant_split}"
    else:
        return "output"


class Main:

    @staticmethod
    def run():
        parser = argparse.ArgumentParser(
            description="Run client and server part of different modes."
        )
        parser.add_argument("--mode", type=str, help="The mode to run.")
        parser.add_argument("--config", type=str, help="Config file path.")
        parser.add_argument("--server", action="store_true", help="Run the server.")
        parser.add_argument("--fed", action="store_true", help="Run the fed server.")
        parser.add_argument("--client", type=int, help="Run client with id.")
        parser.add_argument("--extra", type=str, help="Run client with id.")

        # Arguments for creating the data pickle files. If this exists, it will have priority over the above
        parser.add_argument("--generate", action="store_true", help="For non-iid.")
        parser.add_argument(
            "--data_type",
            type=str,
            default="non_iid",
            choices=[
                "iid",
                "non_iid",
                "feature_skew_dirichlet",
                "label_skew_dirichlet",
                "label_skew_percentage",
                "quantity_skew_dirichlet",
                "quantity_skew_minsize_dirichlet",
                "feature_skew_gaussian",
            ],
            help="Type of data generation: non_iid, feature_skew_dirichlet, label_skew, quantity_skew_dirichlet, quantity_skew_minsize_dirichlet, or feature_skew_gaussian.",
        )

        parser.add_argument(
            "--dataset_name",
            type=str,
            default=None,
            help="For selecting which dataset to use",
        )
        parser.add_argument(
            "--output_name", type=str, default=None, help="data output file name"
        )
        parser.add_argument("--classes_pc", type=int, default=2, help="For non-iid.")
        parser.add_argument(
            "--num_clients",
            type=int,
            default=6,
            help="Number of clients to split data for",
        )
        parser.add_argument(
            "--seed", type=int, default=None, help="Seed for consistent data generation"
        )
        parser.add_argument(
            "--viz",
            action="store_true",
            help="Generate visualization after data creation",
        )
        parser.add_argument(
            "--alpha_feat_split",
            type=float,
            default=1.0,
            help="Alpha for feature skew (for feature_skew).",
        )
        parser.add_argument(
            "--alpha_label_split",
            type=float,
            default=1.0,
            help="Alpha for label skew (for label_skew).",
        )
        parser.add_argument(
            "--alpha_quant_split",
            type=float,
            default=1.0,
            help="Alpha for quantity skew (for quantity_dirichlet).",
        )
        parser.add_argument(
            "--sigma_noise",
            type=float,
            default=2.0,
            help="Sigma for Gaussian noise (for feature_skew_gaussian).",
        )
        parser.add_argument(
            "--percentage_skew",
            type=float,
            default=0.5,
            help="Percentage skew (for label_skew_percentage).",
        )
        parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")

        # old iid/non-iid data manager calls
        # args = parser.parse_args()
        # if args.generate:
        #     if args.iid:
        #         create_iid_dataset(args.dataset_name, num_clients=args.num_clients,
        #                            output_name=args.output_name, seed=args.seed)
        #     else:
        #         create_non_iid_dataset(args.dataset_name, args.num_clients,
        #                                output_name=args.output_name, classes_per_client=args.classes_pc,
        #                                seed=args.seed)

        # Arguments for creating the data pickle file. If this exists, it will have priority over the above

        args = parser.parse_args()
        if args.generate:
            # Generate short output name if not provided
            if not args.output_name:
                args.output_name = get_short_output_name(
                    args.data_type,
                    args.alpha_feat_split,
                    args.alpha_label_split,
                    args.alpha_quant_split,
                    args.sigma_noise,
                    args.percentage_skew,
                )

            match args.data_type:
                case "iid":
                    print(f"Generating IID dataset with {args.num_clients} clients")
                    create_iid_dataset(
                        args.dataset_name,
                        num_clients=args.num_clients,
                        output_name=args.output_name,
                        seed=args.seed,
                    )

                case "non_iid":
                    print(f"Generating non-IID data with classes_pc={args.classes_pc}")
                    create_non_iid_dataset(
                        args.dataset_name,
                        num_clients=args.num_clients,
                        output_name=args.output_name,
                        classes_per_client=args.classes_pc,
                        seed=args.seed,
                    )

                case "feature_skew_dirichlet":
                    print(
                        f"Generating feature skew data with alpha_feat_split={args.alpha_feat_split}"
                    )
                    create_dirichlet_feature_skew(
                        args.dataset_name,
                        num_clients=args.num_clients,
                        output_name=args.output_name,
                        alpha_feat_split=args.alpha_feat_split,
                        seed=args.seed,
                    )

                case "label_skew_dirichlet":
                    print(
                        f"Generating label skew data with alpha_label_split={args.alpha_label_split}"
                    )
                    create_dirichlet_label_skew(
                        args.dataset_name,
                        args.num_clients,
                        output_name=args.output_name,
                        alpha_label_split=args.alpha_label_split,
                        seed=args.seed,
                    )

                case "label_skew_percentage":
                    print(
                        f"Generating label skew data with percentage_skew={args.percentage_skew}"
                    )
                    create_percentage_label_skew(
                        args.dataset_name,
                        args.num_clients,
                        output_name=args.output_name,
                        percentage_skew=args.percentage_skew,
                        seed=args.seed,
                    )

                case "quantity_skew_dirichlet":
                    print(
                        f"Generating quantity skew data with alpha_quant_split={args.alpha_quant_split}"
                    )
                    create_quantity_skew_dirichlet(
                        args.dataset_name,
                        args.num_clients,
                        output_name=args.output_name,
                        alpha_quant_split=args.alpha_quant_split,
                        seed=args.seed,
                    )

                case "quantity_skew_minsize_dirichlet":
                    print(
                        f"Generating quantity skew data with minsize-dirichlet method, alpha_quant_split={args.alpha_quant_split}"
                    )
                    create_quantity_skew_minsize_dirichlet(
                        args.dataset_name,
                        args.num_clients,
                        output_name=args.output_name,
                        alpha_quant_split=args.alpha_quant_split,
                        seed=args.seed,
                    )

                case "feature_skew_gaussian":
                    print(
                        f"Generating feature skew data with Gaussian noise method, sigma_noise={args.sigma_noise}"
                    )
                    create_gaussian_feature_skew(
                        args.dataset_name,
                        num_clients=args.num_clients,
                        output_name=args.output_name,
                        sigma_noise=args.sigma_noise,
                        # n_bins=args.n_bins,
                        # feat_sample_rate=args.feat_sample_rate,
                        seed=args.seed,
                    )
                    # feature_skew_gaussian_run( args.sigma_noise, args.num_clients, args.batch_size)

                case _:
                    print("invalid generator scheme provided.")

            # Generate visualization if requested
            if args.viz:
                print("\nGenerating visualization...")
                try:
                    import pickle
                    import os

                    # Use the same output name that was used for data generation
                    output_name = args.output_name

                    pickle_file = f"{output_name}.pkl"

                    # Check if pickle file exists
                    if os.path.exists(pickle_file):
                        # Load the data
                        with open(pickle_file, "rb") as f:
                            data = pickle.load(f)

                        # Generate visualization for all clients with detailed analysis
                        client_data = visualize_client_distribution(data, output_name)
                        from .datagen.viz import create_detailed_visualization

                        create_detailed_visualization(client_data, output_name)
                        print("Visualization completed successfully!")
                    else:
                        print(
                            f"Warning: Pickle file {pickle_file} not found. Cannot generate visualization."
                        )

                except Exception as e:
                    print(f"Error generating visualization: {e}")

            return

        if args.server == None and args.client == None and args.fed == None:
            print("Must select client, server, or fed mode")
            return
        if (
            args.server == True and args.client != None
        ):  # TODO: smart way of checking no two modes are given at the same time
            print("Cannot provide both client and server at the same time")
            return
        if not args.mode:
            print("Mode cannot be empty")
            return

        match args.mode:
            case "basic_model":
                path = "./src/split-learning/base_model"
                if args.client != None:
                    module = base_client
                else:
                    module = base_server
            case "rr_multiclient":
                path = "./src/split-learning/rr_multiclient"
                if args.client != None:
                    module = rr_multiclient_client
                else:
                    module = rr_multiclient_server
            case "th_multiclient":
                path = "./src/split-learning/th_multiclient"
                if args.client != None:
                    module = th_multiclient_client
                else:
                    module = th_multiclient_server
            case "splitfed_v1":
                path = "./src/split-learning/splitfed_v1"
                if args.client != None:
                    module = splitfed_v1_client
                elif args.server:
                    module = splitfed_v1_server
                else:
                    module = splitfed_v1_fed_server
            case "splitfed_v1_custom_cut":
                path = "./src/split-learning/splitfed_v1_custom_cut"
                if args.client != None:
                    module = splitfed_v1_custom_cut_client
                elif args.server:
                    module = splitfed_v1_custom_cut_server
                else:
                    module = splitfed_v1_custom_cut_fed_server
            case "splitfed_v2":
                path = "./src/split-learning/splitfed_v2"
                if args.client != None:
                    module = splitfed_v2_client
                elif args.server:
                    module = splitfed_v2_server
                else:
                    module = splitfed_v2_fed_server
            case "splitfed_v2_custom_cut":
                path = "./src/split-learning/splitfed_v2_custom_cut/"
                if args.client != None:
                    module = splitfed_v2_custom_cut_client
                elif args.server:
                    module = splitfed_v2_custom_cut_server
                else:
                    module = splitfed_v2_custom_cut_fed_server
            case _:
                print("Unrecognized mode", args.mode)
                return

        if not args.config:
            args.config = path + "/config.yaml"

        runner = module.Runner(args.config)
        runner.client_id = args.client
        if args.extra:
            runner.set_extra_options(args.extra)
        runner.run()
