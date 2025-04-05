from utils import load_config
import pprint
from utils import set_seed, setup_gpus, check_directories
from dataloader import get_dataloader, check_cache, prepare_features, process_data, prepare_inputs
from load import load_data, load_tokenizer
from model import ScenarioModel, SupConModel, CustomModel, LoRAModel
from torch import nn
from trainer import baseline_train,custom_train,supcon_train, run_eval
import torch
import logging
import argparse
from pprint import pformat
from loss import SupConLoss

class Trainer():
    def __init__(self,experiment_config_path, model_config_path):
        config = load_config(experiment_config_path, model_config_path)
        self.set_logger()
        self.logger.info("Using config:")
        self.logger.info(pformat(config))
        args = self.setup_args(config)
        args = setup_gpus(args)
        self.device = torch.device("cuda" if torch.cuda.is_available else "cpu")
        args = check_directories(args)
        set_seed(args)
        if args.distributed:
            self.__set_up_distributed()
        self.args = args
        self.load_dataset()

        
    def setup_args(self,config):
        return argparse.Namespace(**config)

    def __set_up_distributed(self):
        raise NotImplementedError
    
    def set_logger(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler('trainer.log')
        file_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO) 
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)


    def load_dataset(self):
        cache_results, already_exist = check_cache(self.args)
        self.tokenizer = load_tokenizer(self.args)
        if already_exist:
            features = cache_results
        else:
            data = load_data()
            features = prepare_features(self.args, data, self.tokenizer, cache_results)
        self.datasets = process_data(self.args, features, self.tokenizer)
        for k,v in self.datasets.items():
            print(k, len(v))


    def train(self,resume_id = False):
        """
        each time the train provides a unique id, see in train.log or wandb
        you can use that train_id to resume the interrupt training from checkpoint.
        """
        if self.args.task == 'baseline':
            self.logger.info("Running Baseline task")
            criterion = nn.CrossEntropyLoss()
            model = ScenarioModel(self.args, self.tokenizer, target_size=18).to(self.device)
            run_eval(self.args, model, self.datasets, criterion, self.tokenizer, self.logger, split='validation')
            run_eval(self.args, model, self.datasets, criterion, self.tokenizer, self.logger, split='test')
            baseline_train(self.args, model, self.datasets, self.tokenizer, self.logger, resume_id)
        elif self.args.task == 'custom': # you can have multiple custom task for different techniques
            self.logger.info("Running Custom task")
            criterion = nn.CrossEntropyLoss()
            model = CustomModel(self.args, self.tokenizer, target_size=18).to(self.device)
            run_eval(self.args, model, self.datasets, criterion, self.tokenizer, self.logger, split='validation')
            run_eval(self.args, model, self.datasets, criterion, self.tokenizer, self.logger, split='test')
            custom_train(self.args, model, self.datasets, self.tokenizer, self.logger, resume_id)
        elif self.args.task == 'LoRA':
            criterion = nn.CrossEntropyLoss()
            model = LoRAModel(self.args, self.tokenizer, target_size=18, 
                r=self.args.r, lora_alpha=self.args.lora_alpha, lora_dropout=self.args.lora_dropout
            ).to(self.device)
            run_eval(self.args, model, self.datasets, criterion, self.tokenizer, self.logger, split='validation')
            run_eval(self.args, model, self.datasets, criterion, self.tokenizer, self.logger, split='test')
            baseline_train(self.args, model, self.datasets, criterion, self.tokenizer, self.logger, resume_id)
        else:
            raise Exception(f"Found unexpected task: {self.args.task}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments with default baseline configs.")

    parser.add_argument(
        "--experiment_config",
        type=str,
        default="experimental_configs/baseline.yaml", # Default = baseline
        help="Path to the experiment config file."
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="model_configs/baseline.yaml", # Default = baseline
        help="Path to the model config file."
    )

    args = parser.parse_args()

    trainer = Trainer(
        experiment_config_path=args.experiment_config,
        model_config_path=args.model_config
    )
    trainer.train()