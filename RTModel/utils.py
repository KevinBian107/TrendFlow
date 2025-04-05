import os
import numpy as np
import torch
import random
import re
import yaml


def check_directories(args):
    data_path = os.path.join("assets")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
        print(f"Created {data_path} directory")
    
    folder = args.task

    task_path = os.path.join(args.output_dir)
    if not os.path.exists(task_path):
        os.mkdir(task_path)
        print(f"Created {task_path} directory")
    
    folder = args.task
    
    save_path = os.path.join(task_path, folder)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print(f"Created {save_path} directory")
    args.save_dir = save_path

    cache_path = os.path.join(args.input_dir, 'cache')
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
        print(f"Created {cache_path} directory")

    checkpoint_path = os.path.join('checkpoints')
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
        print(f"Created {checkpoint_path} directory")

    if args.debug:
        args.log_interval /= 10

    return args

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def setup_gpus(args):
    n_gpu = 0  # set the default to 0
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
    args.n_gpu = n_gpu
    if n_gpu > 0:   # this is not an 'else' statement and cannot be combined
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return args

def load_config(experiment_config_path:str, model_config_path:str) -> str:

    with open(experiment_config_path, 'r') as file:
        experiment_config = yaml.safe_load(file)
    with open(model_config_path,'r') as file:
        model_config = yaml.safe_load(file)
    

    return {**experiment_config, **model_config}

