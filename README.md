# Split Learning Models

In this repository, we have implemented split learning with different architectures. The implemented architectures and the commands to run them are described below.  


When starting in a new platform, first install the requirements: `pip3 install -r requirements.txt`
To start, first enable the python environment: `source venv/bin/activate`
To stop your python environment, deactivate it: `deactivate`


## Architectures
1. **basic architecture (basic_model)**:
This is the implementation of the vanilla split learning with 1 client algorithm proposed in [[1]](#1). The basic model works with only one client and client id does not play any role, you can pass any integer value.
- `python -m src.split-learning --mode basic_model --server`
- `python -m src.split-learning --mode basic_model --client 1`



2. **round-robin architecture (rr_multiclient)**:
This is a 2-client implementation of basic split learning, where the two clients complete their tasks in a round-robin fashion. In this scenario, each client sends the whole model to the next server (and the next client) after doing its computation. This architecture is suitable for cases where data is *vertically distributed* among clients.

- `python -m src.split-learning --mode rr_multiclient --server`
- `python -m src.split-learning --mode rr_multiclient --client 1`


*TODO*: The current code does not devide the data among clients. 


3. **multi-threaded architecture (th_multiclient)**:
This is a 2-client implementation of basic split learning, where the two clients complete their tasks concurrently using threads. 

- `python -m src.split-learning --mode th_multiclient --server`
- `python -m src.split-learning --mode th_multiclient --client 1`


*TODO*: The current code does not devide the data among clients. 


4. **split-fed v1 architecture (splitfed_v1)**:
This is the implementation of the version 1 of the algorithm proposed in [[2]](#2). This architecture is suitable for cases where data is *vertically distributed* among clients. The current implementation needs a data server, please see the Section related to *Splitting Data* and start the data server.

- `python -m src.split-learning --mode splitfed_v1 --server`
- `python -m src.split-learning --mode splitfed_v1 --fed`
- `python -m src.split-learning --mode splitfed_v1 --client 1`

5. **split-fed v2 architecture (splitfed_v2)**:
This is the implementation of the version 2 of the algorithm proposed in [[2]](#2). This architecture is suitable for cases where data is *vertically distributed* among clients. The current implementation needs a data server, please see the Section related to *Splitting Data* and start the data server.

- `python -m src.split-learning --mode splitfed_v2 --server`
- `python -m src.split-learning --mode splitfed_v2 --fed`
- `python -m src.split-learning --mode splitfed_v2 --client 1`

6. **split-fed v1 custom architecture (splitfed_v1_custom_cut)**:
Clients in this architecture can have different cut layers. The current implementation needs a data server, please see the Section related to *Splitting Data* and start the data server.

- `python -m src.split-learning --mode splitfed_v1_custom_cut --server --extra 2,5`
- `python -m src.split-learning --mode splitfed_v1_custom_cut --fed`
- `python -m src.split-learning --mode splitfed_v1_custom_cut --client 1 --extra 2`

7. **split-fed v2 custom architecture (splitfed_v2_custom_cut)**:
Clients in this architecture can have differing cut layers. The current implementation needs a data server, please see the Section related to *Splitting Data* and start the data server. Note that unlike the v1 custom architecture, the split server does not require any extra input.

- `python -m src.split-learning --mode splitfed_v2_custom_cut --server`
- `python -m src.split-learning --mode splitfed_v2_custom_cut --fed`
- `python -m src.split-learning --mode splitfed_v2_custom_cut --client 1 --extra 2`


## Splitting Data
Multiple data splitting strategies are implemented based on fedArtML library [[3]](#3). These scripts can create Label, Feature, and Quantity skews for the non-iid data.

Each splitting strategy also accepts the following additional parameters:
`--dataset_name`: Name of the dataset to use for generation (e.g. 'cifar10') **REQUIRED**
`--output_name`: Name of the output files. Will create {output_name}.pkl, {output_name}_test.pkl, and {output_name}_val.pkl  
`--seed`: Seed to fix random generation to. Leave unassigned for randomized generation.

In addition, each 

The following are the implemented data splitting strategies:
- `iid`
    - num_clients
- `non-iid`: Data sharding
    - num_clients
    - classes_pc
- `label_skew_dirichlet`
    - num_clients
    - alpha_label_split (smaller = more non-IID)
- `label_skew_percentage`
    - num_clients
    - percentage_skew (0.0 to 1.0)
- `feature_skew_dirichlet`
    - num_clients
    - alpha_feat_split (smaller = more non-IID)
- `feature_skew_gaussian`
    - num_clients
    - sigma_noise (larger = more non-IID)
- `quantity_skew_dirichlet`
    - num_clients
    - alpha_quant_split (smaller = more non-IID)
- `quantity_skew_minsize_dirichlet`
    - num_clients
    - alpha_quant_split (smaller = more non-IID)

Below are example commands to run in project `root` to use the various data splitting strategies.

`python -m src.split-learning --generate --dataset_name cifar10 --data_type iid --num_clients 6`

`python -m src.split-learning --generate --dataset_name cifar10 --data_type non-iid --num_clients 6 --classes_pc 2`

`python -m src.split-learning --generate --dataset_name cifar10 --data_type label_skew_dirichlet --alpha_label_split 0.1 --num_clients 3`

`python -m src.split-learning --generate --dataset_name cifar10 --data_type label_skew_percentage --percentage_skew 0.5 --num_clients 2`

`python -m src.split-learning --generate --dataset_name cifar10 --data_type feature_skew_dirichlet --alpha_feat_split 0.1 --num_clients 3`

`python -m src.split-learning --generate --dataset_name cifar10 --data_type feature_skew_gaussian --sigma_noise 1 --num_clients 3`

`python -m src.split-learning --generate --dataset_name cifar10 --data_type quantity_skew_dirichlet --alpha_quant_split 0.1 --num_clients 3`

`python -m src.split-learning --generate --dataset_name cifar10 --data_type quantity_skew_minsize_dirichlet --alpha_quant_split 0.1 --num_clients 3`


The generated pickle files should be placed on the data server directory. Data server can be started using: `python -m http.server port_number`, e.g. `python -m http.server 8000`


Example calls with additional parameters are as follows:

`python -m src.split-learning --generate --dataset_name cifar10 --output_name output --num_clients 6 --seed 42`  
`python -m src.split-learning --generate --dataset_name cifar10 --data_type quantity_skew_minsize_dirichlet --alpha_quant_split 0.1 --num_clients 3 --output_name minsize_fixed_42 --seed 42`
`python -m src.split-learning --generate --dataset_name cifar10 --data_type quantity_skew_minsize_dirichlet --alpha_quant_split 0.0005 --num_clients 5 --seed 42 --viz`  

More datasets may be added to the datamanager by adding a simple getter function to the dictionary of datasets.

### Data Visualization
The generated data can be visualzied by passing --viz input to the data generator commands.

The data visualization script can be calleded directly by passing an already generated data file passed by --pickle_file, and can do the visualization only for selected clients passed by --clinets input

`python src/split-learning/datagen/viz.py --pickle_file ld_0.01.pkl --detailed --clients "0,1,2"`

## Common Files
- **Convert**: This is a utility file that contains some conversion utility methods.
- **Config**: Both client and server files read the setup configuration from config.yaml file. The config file for different architectures are sligthly different (depending on what parameters were required for each implementation).
- **CustomImageDataset**: This file is used for reading the datasert transfered over socket. (TODO)
- **app.py** The src/split-learning/app.py file is the file that runs the other codes. 


## Scripts
- **run.sh**: A bash script that automates the running of multiple experiments based on an input file. Allows for running multiple passes of either SplitFedV1 or SplitFedV1_Custom_Cut. Automatically enables the python environment and data server.

For automatic running, the input file should be formatted as follows:
cut_layer epochs rounds split_type.

e.g.
`5, 5, 10, "s"`

If you are using custom cut, separate all the client split layers by commas.
e.g.
`3,5 5 10 "s"`

Each new line dictates a new experiment.

Based on running mode, output is saved in a timestamped directory to either `/$HOME/manual_experiments/` or `/$HOME/auto_experiments/.` 


Assumptions:
- The repo is found in /$HOME/.
- Data server is hosted on the split server.
- Read/Write permissions.

Dependencies:
- dialog
- yq
- collectl
- sysstat


## Docker Setup

1. Dockerfile  
The Dockerfile is found in the /src/ folder, and contains all model code when built.
Line 5 of the file installs torch and torchvision with a cuda wheel. This line may need to be
changed depending on the hardware available to the docker engine.

2. docker-compose
two docker-compose files can be found in the repository root. One corresponds to v1 and the other to v2. This file starts up the Split Server, the Fed server, and as many clients as are listed.

Of note within this docker-compose file:
- Expects there to be a config.yaml file in the repo root folder. This is the config that will be used by all the containers. Within /src/ there is a dummy config.yaml that ensures a folder called "config.yaml" isn't created when docker attempts to create a link to the config on the host.
- Expects the project to have a /logs/ folder where logs will be output to.
- Additional clients can be added by copying and pasting the client template. Command lines should be modified as usual.

3. Data server
The data server is expected to be hosted locally on the host. See the section on Splitting Data.




## References
<a id="1">[1]</a> 
Vepakomma P, Gupta O, Swedish T, Raskar R. Split learning for health: Distributed deep learning without sharing raw patient data. arXiv preprint arXiv:1812.00564. 2018 Dec 3.

<a id="2">[2]</a> 
Thapa C, Arachchige PC, Camtepe S, Sun L. Splitfed: When federated learning meets split learning. In Proceedings of the AAAI Conference on Artificial Intelligence 2022 Jun 28 (Vol. 36, No. 8, pp. 8485-8493).

<a id="3">[3]</a> 
Jimenez GGM, Anagnostopoulos A, Chatzigiannakis I, and Vitaletti A. Fedartml: A tool to facilitate the
generation of non-iid datasets in a controlled way to support federated learning research. IEEE Access, 2024.