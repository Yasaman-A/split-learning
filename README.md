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


## Model Architecture System

The repository uses a flexible model architecture system that allows easy addition of new model architectures. All model architectures implemented in the proper location are automatically discovered and registered at runtime.

### Available Model Architectures

The following model architectures are currently available:

1. **ResNet18_CIFAR10**: ResNet18 adapted for CIFAR-10 (32x32 RGB, 10 classes)
2. **ResNet18_MNIST**: ResNet18 adapted for MNIST (28x28 grayscale, 10 classes)
3. **SimpleCNN_MNIST**: Custom CNN for MNIST (28x28 grayscale, 10 classes)
4. **ResNet18_DermaMNIST**: ResNet18 adapted for DermaMNIST (28x28 RGB, 7 classes)
5. **EfficientNet_DermaMNIST**: EfficientNet-B0 adapted for DermaMNIST (28x28 RGB, 7 classes)

### Configuration

Model architectures are specified in the `config.yaml` file for each split learning mode:

```yaml
model_architecture: "ResNet18_CIFAR10"  # Name of the model architecture
logits: 10  # Number of output classes
```

**Note**: Architecture names are case-insensitive (e.g., `"resnet18_cifar10"` and `"ResNet18_CIFAR10"` are equivalent).

### Creating New Model Architectures

To create a new model architecture, create a new file in `src/split-learning/architectures/models/` following this structure:

```python
import torch.nn as nn
from torchvision import transforms
from ..model_registry import register
from ..architecture_bundle import ArchitectureBundle

# 1. Base Model (complete model, no splitting)
class YourModel(nn.Module):
    def __init__(self, config):
        super(YourModel, self).__init__()
        # ... model definition ...
        self.logits = config["logits"]
        self.layers = list(self.model.children())  # For split learning

# 2. Client Model (forward pass stops at cut_layer)
class YourModelClient(nn.Module):
    def __init__(self, config):
        super(YourModelClient, self).__init__()
        self.cut_layer = config["cut_layer"]
        # ... model definition ...
        self.layers = list(self.model.children())
    
    def forward(self, x):
        for i, l in enumerate(self.layers):
            if i > self.cut_layer:
                break
            x = l(x)
        return x

# 3. Server Model (forward pass starts after cut_layer)
class YourModelServer(nn.Module):
    def __init__(self, config):
        super(YourModelServer, self).__init__()
        self.cut_layer = config["cut_layer"]
        # ... model definition ...
        self.layers = list(self.model.children())
    
    def forward(self, x):
        for i, l in enumerate(self.layers):
            if i <= self.cut_layer:
                continue
            x = l(x)
        return x
    
    def classify(self, x):
        return nn.functional.softmax(self.forward(x))

# 4. Training Transformer (can include augmentations)
transformer = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# 5. Evaluation Transformer (no augmentations)
eval_transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# 6. Register the architecture
@register("YourModel_Name")
def grab():
    return ArchitectureBundle(
        base=YourModel,
        client=YourModelClient,
        server=YourModelServer,
        training_transformer=transformer,
        eval_transformer=eval_transformer,
    )
```

**Important Notes:**
- The model file must be placed in `src/split-learning/architectures/models/`
- The file will be automatically imported and registered at runtime
- All three model classes (base, client, server) must be implemented
- Both transformers (training and eval) must be provided
- The architecture must be registered using the `@register()` decorator
- For compatibility with model averaging, ensure layers are accessible via `self.model` (see `resnet18_cifar10.py` or `simplecnn_mnist.py` for examples)

For complete examples, see:
- `src/split-learning/architectures/models/resnet18_cifar10.py` (torchvision model)
- `src/split-learning/architectures/models/simplecnn_mnist.py` (custom model)

## Splitting Data

Multiple data splitting strategies are implemented based on fedArtML library [[3]](#3). These scripts can create Label, Feature, and Quantity skews for the non-iid data.

### Supported Datasets

The following datasets are currently supported:
- **CIFAR-10**: 32x32 RGB images, 10 classes
- **MNIST**: 28x28 grayscale images, 10 classes
- **DermaMNIST**: 28x28 RGB images, 7 classes (from MedMNIST)
- **OrganAMNIST (Axial)**: 28x28 RGB images, 11 classes (from MedMNIST)
- **OrganCMNIST (Coronal)**: 28x28 RGB images, 11 classes (from MedMNIST)
- **OrganSMNIST (Sagittal)**: 28x28 RGB images, 11 classes (from MedMNIST)

More datasets can be added by implementing a getter function in `src/split-learning/datagen/data_manager.py`.

### Data Generation Parameters

Each splitting strategy accepts the following parameters:
- `--dataset_name`: Name of the dataset to use for generation (e.g., 'cifar10', 'mnist', 'dermamnist') **REQUIRED**
- `--data_type`: Type of data splitting strategy (see below)
- `--num_clients`: Number of clients to split data among **REQUIRED**
- `--output_name`: Name of the output files. Will create `{output_name}.pkl`, `{output_name}_test.pkl`, and `{output_name}_val.pkl`
- `--seed`: Seed to fix random generation to. Leave unassigned for randomized generation.
- `--viz`: Optional flag to visualize the data distribution after generation

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

### Example Commands

Below are example commands to run in project `root` to use the various data splitting strategies:

**CIFAR-10 Examples:**
```bash
python -m src.split-learning --generate --dataset_name cifar10 --data_type iid --num_clients 6
python -m src.split-learning --generate --dataset_name cifar10 --data_type non-iid --num_clients 6 --classes_pc 2
python -m src.split-learning --generate --dataset_name cifar10 --data_type label_skew_dirichlet --alpha_label_split 0.1 --num_clients 3
python -m src.split-learning --generate --dataset_name cifar10 --data_type label_skew_percentage --percentage_skew 0.5 --num_clients 2
python -m src.split-learning --generate --dataset_name cifar10 --data_type feature_skew_dirichlet --alpha_feat_split 0.1 --num_clients 3
python -m src.split-learning --generate --dataset_name cifar10 --data_type feature_skew_gaussian --sigma_noise 1 --num_clients 3
python -m src.split-learning --generate --dataset_name cifar10 --data_type quantity_skew_dirichlet --alpha_quant_split 0.1 --num_clients 3
python -m src.split-learning --generate --dataset_name cifar10 --data_type quantity_skew_minsize_dirichlet --alpha_quant_split 0.1 --num_clients 3
```

**MNIST Examples:**
```bash
python -m src.split-learning --generate --dataset_name mnist --data_type label_skew_dirichlet --alpha_label_split 0.1 --num_clients 5 --output_name output_mnist --seed 42
```

**DermaMNIST Examples:**
```bash
python -m src.split-learning --generate --dataset_name dermamnist --data_type label_skew_dirichlet --alpha_label_split 100 --num_clients 5 --output_name output_dermamnist --seed 42 --viz
```

**Example calls with additional parameters:**
```bash
python -m src.split-learning --generate --dataset_name cifar10 --output_name output --num_clients 6 --seed 42
python -m src.split-learning --generate --dataset_name cifar10 --data_type quantity_skew_minsize_dirichlet --alpha_quant_split 0.1 --num_clients 3 --output_name minsize_fixed_42 --seed 42
python -m src.split-learning --generate --dataset_name cifar10 --data_type quantity_skew_minsize_dirichlet --alpha_quant_split 0.0005 --num_clients 5 --seed 42 --viz
```

### Data Server Setup

The generated pickle files should be placed in the data server directory. The data server can be started using:

```bash
python -m http.server <port_number>
```

For example:
```bash
python -m http.server 8000
```

Make sure the `output_file` in your `config.yaml` matches the generated pickle file name (without the `.pkl` extension). For example, if you generated `output_cifar10.pkl`, set `output_file: output_cifar10.pkl` in the config.

### Data Visualization
The generated data can be visualzied by passing --viz input to the data generator commands.

The data visualization script can be calleded directly by passing an already generated data file passed by --pickle_file, and can do the visualization only for selected clients passed by --clinets input

`python src/split-learning/datagen/viz.py --pickle_file ld_0.01.pkl --detailed --clients "0,1,2"`

## Configuration

Both client and server files read the setup configuration from `config.yaml` file. The config file location depends on the split learning mode (e.g., `splitfed_v1_custom_cut/config.yaml`).

### Key Configuration Parameters

- `model_architecture`: Name of the model architecture to use (e.g., `"ResNet18_CIFAR10"`, `"ResNet18_DermaMNIST"`)
- `logits`: Number of output classes (e.g., `10` for CIFAR-10/MNIST, `7` for DermaMNIST)
- `client_total`: Total number of clients
- `epoch`: Number of training epochs per round
- `round`: Number of federated learning rounds
- `batch_size`: Batch size for training
- `test_cut_layer`: Cut layer index for evaluation
- `data_server.output_file`: Name of the pickle file to download from the data server

### Example Configuration

```yaml
model_architecture: "ResNet18_CIFAR10"
logits: 10
client_total: 2
epoch: 5
round: 2
batch_size: 32
test_cut_layer: 4
data_server:
  server_address: http://127.0.0.1:8000
  output_file: output_cifar10.pkl
```

## Common Files

- **Convert**: Utility file containing conversion utility methods for data serialization.
- **Config**: Configuration files (`config.yaml`) that specify parameters for each split learning mode.
- **CustomImageDataset**: Used for reading datasets transferred over socket.
- **app.py**: Main entry point (`src/split-learning/app.py`) that orchestrates the split learning execution. 


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