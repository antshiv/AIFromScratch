import json
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class TransformerModel(nn.Module):
    def __init__(self, config):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(config['vocab_size'], config['embed_dim'])
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=config['embed_dim'], nhead=config['num_heads'], dim_feedforward=config['feedforward_dim'])
            for _ in range(config['num_layers'])
        ])
        self.classifier = nn.Linear(config['embed_dim'], config['num_classes'])

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.encoder_layers:
            x = layer(x)
        x = x.mean(dim=1)  # Pooling
        return self.classifier(x)

def read_config(file_path):
    with open(file_path, 'r') as f:
        config_data = json.load(f)
    return config_data['config']

def log_parallel_computation(writer, config):
    # Dummy input for visualization
    dummy_input = torch.randint(0, config['vocab_size'], (config['batch_size'], config['max_length']))

    # Define model
    model = TransformerModel(config)

    # Parallel computation for mini-batch
    batch_splits = torch.split(dummy_input, config['batch_size'] // 4, dim=0)
    outputs = []
    for split in batch_splits:
        outputs.append(model(split))
    output = torch.cat(outputs, dim=0)

    # Log computational graph
    writer.add_graph(model, dummy_input)

def main():
    config_file_path = "transformer_config.json"
    config_data = read_config(config_file_path)

    writer = SummaryWriter(log_dir='./tensorboard_logs')
    log_parallel_computation(writer, config_data)

    writer.close()
    print("TensorBoard logs written to ./tensorboard_logs")

if __name__ == "__main__":
    main()
