# Split Learning Models

In this repository, we have implemented split learning with different architectures. These implementations and the commands to run them are described below. All of these implemntations use a set of common files described in the common section.

## Architectures
1. basic architecture
This is the implementation of the vanilla split learning with 1 client algorithm proposed in [[1]](#1).

- `python3 client.py config.yaml client_id`
- `python3 server.py config.yaml`


2. round-robin architecture
This is a 2-client implementation of basic split learning, where the two clients complete their tasks in a round-robin fashion. In this scenario, each client sends the whole model to the next server (and the next client) after doing its computation. This architecture is suitable for cases where data is *vertically distributed* among clients.

- `python3 client.py config.yaml client_id`
- `python3 server.py config.yaml`

TODO: The current code does not devide the data among clients. 


3. multi-threaded architecture
This is a 2-client implementation of basic split learning, where the two clients complete their tasks concurrently using threads. 

- `python3 client.py config.yaml client_id`
- `python3 server.py config.yaml`

TODO: The current code does not devide the data among clients. 


4. split-fed v1 architecture
This is the implementation of the version 1 of the algorithm proposed in [[2]](#2). This architecture is suitable for cases where data is *vertically distributed* among clients.

- `python3 client.py config.yaml client_id`
- `python3 server.py config.yaml`
- `python3 fedServer.py config.yaml`

5. split-fed v2 architecture
This is the implementation of the version 2 of the algorithm proposed in [[3]](#3). This architecture is suitable for cases where data is *vertically distributed* among clients.

- `python3 client.py config.yaml client_id`
- `python3 server.py config.yaml`
- `python3 fedServer.py config.yaml`

6. split-fed v1 custom architecture
Clients in this architecture can have different cut layers.

- `python3 client.py config.yaml client_id cut_layer`
- `python3 server.py config.yaml list_of_cut_layers`, e.g., `python3 server.py config.yaml 3 5`
- `python3 fedServer.py config.yaml`

## Splitting Data
Running non-iid.py will resolve in the number of non-idd data splits. To run this code, use: `python3 non-iid.py classes_pc num_clients batch_size`, e.g., `python3 non-iid.py 2 6 128`. 

If you need to device code among x clients, pass *x+1* as the *num_clients*. This is due to the implementation of the code that assigns very few data points to the last client which makes the last split to be a useless split.

Running non-iid.py results in *output.pickle* file. This pickle file should be placed on the data server directory. Data server can be started using: `python3 -m http.server port_number`

## Common Files
- Convert: This is a utility file that contains some conversion utility methods.
- Config file: Both client and server files read the setup configuration from config.yaml file. The config file for different architectures are sligthly different (depending on what parameters were required for each implementation).
- CustomImageDataset: This file is used for reading the datasert transfered over socket.


## References
<a id="1">[1]</a> 
Vepakomma P, Gupta O, Swedish T, Raskar R. Split learning for health: Distributed deep learning without sharing raw patient data. arXiv preprint arXiv:1812.00564. 2018 Dec 3.

<a id="2">[2]</a> 
Thapa C, Arachchige PC, Camtepe S, Sun L. Splitfed: When federated learning meets split learning. InProceedings of the AAAI Conference on Artificial Intelligence 2022 Jun 28 (Vol. 36, No. 8, pp. 8485-8493).

<a id="3">[3]</a> 
Thapa C, Arachchige PC, Camtepe S, Sun L. Splitfed: When federated learning meets split learning. InProceedings of the AAAI Conference on Artificial Intelligence 2022 Jun 28 (Vol. 36, No. 8, pp. 8485-8493).
