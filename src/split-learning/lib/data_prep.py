"""Downloads and prepares the data for SplitFed V1 or V2"""
import os
import pickle
import urllib.request
from torch.utils.data import DataLoader
from transformed_dataset import TransformedDataset

class DataPrep:
    """ Downloads data from the Data Server and applies transformers.
        For downloading the training data, the first client is assumed by default
        for the ease of testing."""
    def __init__(self, config: dict, arch: ArchitectureBundle, client_id=1):
        self.training_transformer = arch.training_transformer
        self.eval_transformer = arch.eval_transformer

        self.batch_size = config['batch_size']
        self.data_server = config['data_server']
        self.split_type = config.get['split_type']

        self.client_id = client_id


    def __download_data_from_server(self, data_split):

        payload = self.data_server['output_file']

        match data_split:
            case "training":
                payload_temp = f"tmp_{self.client_id}_{payload}"
            case "validation":
                payload = payload.replace(".pkl", "_val.pkl")
                payload_temp = f"tmp_validation_{payload}"
            case "testing":
                payload = payload.replace(".pkl", "_test.pkl")
                payload_temp = f"tmp_testing_{payload}"
            case _ as invalid_data_split:
                error_str = f"""Error: invalid data split specified. \n
                               The valid options are: \n
                               \t 'training' \n
                               \t 'validation' \n
                               \t 'testing \n
                               Specified data split: {invalid_data_split}
                            """
                raise ValueError(error_str)

        print(f"Getting {payload} from {self.data_server['server_address']}/")
        urllib.request.urlretrieve(
            f"{self.data_server['server_address']}/{payload}", payload_temp
        )

        with open(payload_temp, "rb") as payload_data:
            dataset = pickle.load(payload_data)

        if data_split == "training":
            dataset = dataset[self.client_id-1]

        os.remove(payload_temp)

        return dataset


    def __get_transformed_training_set(self):
        match self.split_type:
            case 's' | None:
                dataset = self.__download_data_from_server("training")
                transformed_set = TransformedDataset(dataset, self.training_transformer)
                sampler = None
                shuffle = True
            case _ as invalid_split_type:
                error_str = f"""Error: Legacy or Invalid Split type specified in Config. \n
                         for V1 or V2, a split type of 's' (or none defined) is required.
                         A value of {invalid_split_type} was given."""
                raise ValueError(error_str)

        return transformed_set, sampler, shuffle

    def __get_transformed_eval_set(self, data_split="validation"):
        dataset = self.__download_data_from_server(data_split)
        transformed_set = TransformedDataset(dataset, self.eval_transformer)
        return transformed_set

    def get_training_loader(self):
        """ Downloads a training dataset from the data server and applies the training
        transformer to it, and wraps it in a DataLoader"""
        trainset, sampler, shuffle = self.__get_transformed_training_set()

        trainloader = DataLoader(
            trainset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler = sampler,
            num_workers=2,
            persistent_workers=True)

        return trainloader

    def get_eval_loader(self, eval_type="validation"):
        """ Downloads an evaluation dataset from the data server and applies the training
        transformer to it, and wraps it in a DataLoader"""
        dataset = self.__get_transformed_eval_set(eval_type)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            persistent_workers=False,
        )
        return dataloader
