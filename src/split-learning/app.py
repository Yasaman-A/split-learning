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
from .datagen.non_iid import run as non_iid_run
from .datagen.feature_skew_dirichlet import run as feature_skew_dirichlet_run
from .datagen.label_skew_dirichlet import run as label_skew_dirichlet_run
from .datagen.quantity_skew_dirichlet import run as quantity_skew_dirichlet_run
from .datagen.quantity_skew_minsize_dirichlet import (
    run as quantity_skew_minsize_dirichlet_run,
)
from .datagen.feature_skew_gaussian import run as feature_skew_gaussian_run
from .datagen.label_skew_percentage import run as label_skew_percentage_run
import argparse


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

        # Arguments for creating the data pickle file. If this exists, it will have priority over the above
        parser.add_argument(
            "--generate",
            action="store_true",
            help="Generate data (non-iid or feature-skew).",
        )
        parser.add_argument(
            "--data_type",
            type=str,
            default="non_iid",
            choices=[
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
            "--classes_pc",
            type=int,
            default=2,
            help="Classes per client (for non_iid).",
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
        parser.add_argument(
            "--num_clients", type=int, default=6, help="Number of clients."
        )
        parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")

        args = parser.parse_args()
        if args.generate:
            if args.data_type == "non_iid":
                print(f"Generating non-IID data with classes_pc={args.classes_pc}")
                non_iid_run(args.classes_pc, args.num_clients, args.batch_size)
            elif args.data_type == "feature_skew_dirichlet":
                print(
                    f"Generating feature skew data with alpha_feat_split={args.alpha_feat_split}"
                )
                feature_skew_dirichlet_run(
                    args.alpha_feat_split, args.num_clients, args.batch_size
                )
            elif args.data_type == "label_skew_dirichlet":
                print(
                    f"Generating label skew data with alpha_label_split={args.alpha_label_split}"
                )
                label_skew_dirichlet_run(
                    args.alpha_label_split, args.num_clients, args.batch_size
                )
            elif args.data_type == "label_skew_percentage":
                print(
                    f"Generating label skew data with percentage_skew={args.percentage_skew}"
                )
                label_skew_percentage_run(
                    args.percentage_skew, args.num_clients, args.batch_size
                )
            elif args.data_type == "quantity_skew_dirichlet":
                print(
                    f"Generating quantity skew data with alpha_quant_split={args.alpha_quant_split}"
                )
                quantity_skew_dirichlet_run(
                    args.alpha_quant_split, args.num_clients, args.batch_size
                )
            elif args.data_type == "quantity_skew_minsize_dirichlet":
                print(
                    f"Generating quantity skew data with minsize-dirichlet method, alpha_quant_split={args.alpha_quant_split}"
                )
                quantity_skew_minsize_dirichlet_run(
                    args.alpha_quant_split, args.num_clients, args.batch_size
                )
            elif args.data_type == "feature_skew_gaussian":
                print(
                    f"Generating feature skew data with Gaussian noise method, sigma_noise={args.sigma_noise}"
                )
                feature_skew_gaussian_run(
                    args.sigma_noise, args.num_clients, args.batch_size
                )
            return

        if args.server == None and args.client == None and args.fed == None:
            print("Must select clien, server, or fed mode")
            return
        if (
            args.server == True and args.client != None
        ):  # TODO: smart way of checking no two modes are given at the same time
            print("Cannot provide both client and server at the same time")
            return
        if not args.mode:
            print("Mode cannot be empty")
            return

        if args.mode == "basic_model":
            path = "./src/split-learning/base_model"
            if args.client != None:
                module = base_client
            else:
                module = base_server
        elif args.mode == "rr_multiclient":
            path = "./src/split-learning/rr_multiclient"
            if args.client != None:
                module = rr_multiclient_client
            else:
                module = rr_multiclient_server
        elif args.mode == "th_multiclient":
            path = "./src/split-learning/th_multiclient"
            if args.client != None:
                module = th_multiclient_client
            else:
                module = th_multiclient_server
        elif args.mode == "splitfed_v1":
            path = "./src/split-learning/splitfed_v1"
            if args.client != None:
                module = splitfed_v1_client
            elif args.server:
                module = splitfed_v1_server
            else:
                module = splitfed_v1_fed_server
        elif args.mode == "splitfed_v1_custom_cut":
            path = "./src/split-learning/splitfed_v1_custom_cut"
            if args.client != None:
                module = splitfed_v1_custom_cut_client
            elif args.server:
                module = splitfed_v1_custom_cut_server
            else:
                module = splitfed_v1_custom_cut_fed_server
        elif args.mode == "splitfed_v2":
            path = "./src/split-learning/splitfed_v2"
            if args.client != None:
                module = splitfed_v2_client
            elif args.server:
                module = splitfed_v2_server
            else:
                module = splitfed_v2_fed_server
        else:
            print("Unrecognized mode", args.mode)
            return
        if not args.config:
            args.config = path + "/config.yaml"

        runner = module.Runner(args.config)
        runner.client_id = args.client
        if args.extra:
            runner.set_extra_options(args.extra)
        runner.run()
