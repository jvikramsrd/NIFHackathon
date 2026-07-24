import logging
import random
import os
import torch
import numpy as np

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def setup(seed: int = 42) -> torch.device:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def atomic_torch_save(obj, filepath: str):
    tmp_path = filepath + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, filepath)

def get_yolo_device() -> str:
    return "0" if torch.cuda.is_available() else "cpu"
