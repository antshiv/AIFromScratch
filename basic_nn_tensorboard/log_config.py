import json
import torch
from torch.utils.tensorboard import SummaryWriter

def read_config(file_path):
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config

def log_config(writer, config):
    for key, value in config.items():
        if isinstance(value, int) or isinstance(value, float):
            writer.add_scalar(f'Config/{key}', value)

def main():
    config_file_path = "transformer_config.json"
    config = read_config(config_file_path)
    
    writer = SummaryWriter(log_dir='./tensorboard_logs')
    log_config(writer, config)
    
    writer.close()
    print("TensorBoard logs written to ./tensorboard_logs")

if __name__ == "__main__":
    main()
