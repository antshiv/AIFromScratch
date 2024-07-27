import json
import torch
from torch.utils.tensorboard import SummaryWriter

def read_config(file_path):
    with open(file_path, 'r') as f:
        config_data = json.load(f)
    return config_data

def log_config(writer, config_data):
    config = config_data['config']

    # Log configuration
    for key, value in config.items():
        writer.add_text('Config/' + key, str(value))
        writer.add_scalar(f'Config/{key}', value if isinstance(value, (int, float)) else 0)

    # Transformer model based on config
    class TransformerModel(torch.nn.Module):
        def __init__(self, config):
            super(TransformerModel, self).__init__()
            self.embedding = torch.nn.Embedding(config['vocab_size'], config['embed_dim'])
            self.encoder_layers = torch.nn.ModuleList([
                torch.nn.TransformerEncoderLayer(d_model=config['embed_dim'], nhead=config['num_heads'], dim_feedforward=config['feedforward_dim'])
                for _ in range(config['num_layers'])
            ])
            self.classifier = torch.nn.Linear(config['embed_dim'], config['num_classes'])

        def forward(self, x):
            x = self.embedding(x)
            for layer in self.encoder_layers:
                x = layer(x)
            x = self.classifier(x.mean(dim=1))  # Assume pooling is done by taking the mean
            return x

    model = TransformerModel(config)
    dummy_input = torch.randint(0, config['vocab_size'], (config['batch_size'], config['max_length']))
    writer.add_graph(model, dummy_input)

def main():
    config_file_path = "transformer_config.json"
    config_data = read_config(config_file_path)

    writer = SummaryWriter(log_dir='./tensorboard_logs')
    log_config(writer, config_data)

    writer.close()
    print("TensorBoard logs written to ./tensorboard_logs")

if __name__ == "__main__":
    main()
