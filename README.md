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


## Splitting Data
Running non-iid.py will resolve in the number of non-idd data splits. Using `--generate` you can run the non-iid.py script from the `root` folder which generates data splits:
`python -m src.split-learning --generate` this code can accept three parameters: `--classes_pc`, `--num_clients`, and `--batch_size`. If any of these parameters are not passed, then the default values of 2, 6, 128 will be used. To generate data with non-default values:
`python -m src.split-learning --generate --classes_pc 4 --num_clients 6 --batch_size 128`


Note: if you need to devide code among *x* clients, pass *x+1* as the *num_clients*. This is due to the implementation of the code that assigns very few data points to the last client which makes the last split to be a useless split.

Running non-iid.py results in *output.pickle* file. This pickle file should be placed on the data server directory. Data server can be started using: `python -m http.server port_number`, e.g. `python -m http.server 8000`



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


## References
<a id="1">[1]</a> 
Vepakomma P, Gupta O, Swedish T, Raskar R. Split learning for health: Distributed deep learning without sharing raw patient data. arXiv preprint arXiv:1812.00564. 2018 Dec 3.

<a id="2">[2]</a> 
Thapa C, Arachchige PC, Camtepe S, Sun L. Splitfed: When federated learning meets split learning. In Proceedings of the AAAI Conference on Artificial Intelligence 2022 Jun 28 (Vol. 36, No. 8, pp. 8485-8493).
