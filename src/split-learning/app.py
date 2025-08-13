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


from .non_iid import run
from .data_manager import create_iid_dataset, create_non_iid_dataset

import argparse

class Main:

    @staticmethod
    def run():
        parser = argparse.ArgumentParser(description='Run client and server part of different modes.')
        parser.add_argument('--mode',  type=str,            help='The mode to run.')
        parser.add_argument('--config',type=str,            help='Config file path.')
        parser.add_argument('--server',action='store_true', help='Run the server.')
        parser.add_argument('--fed',   action='store_true', help='Run the fed server.')
        parser.add_argument('--client',type=int,            help='Run client with id.')
        parser.add_argument('--extra', type=str,            help='Run client with id.')

        # Arguments for creating the data pickle files. If this exists, it will have priority over the above
        parser.add_argument('--generate',    action='store_true',              help='For non-iid.')
        parser.add_argument('--iid',         dest='iid', action='store_true',  help='For generating iid data')
        parser.add_argument('--non-iid',     dest='iid', action='store_false', help='For generating non-iid data')
        parser.add_argument('--dataset_name',type=str,                         help='For selecting which dataset to use')
        parser.add_argument('--output_name', type=str,                         help='data output file name')
        parser.add_argument('--classes_pc',  type=int, default=2,              help='For non-iid.')
        parser.add_argument('--num_clients', type=int, default=6,              help='Number of clients to split data for')
        parser.add_argument('--seed',        type=int, default=None,           help='Seed for consistent data generation')

        args = parser.parse_args()
        if args.generate:
            if args.iid:
                create_iid_dataset(args.dataset_name, num_clients=args.num_clients,
                                   output_name=args.output_name, seed=args.seed)
            else:
                create_non_iid_dataset(args.dataset_name, args.num_clients, 
                                       output_name=args.output_name, classes_per_client=args.classes_pc,
                                       seed=args.seed)
            return

        if args.server == None and args.client == None and args.fed == None:
            print("Must select clien, server, or fed mode")
            return
        if args.server == True and args.client != None: # TODO: smart way of checking no two modes are given at the same time
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
