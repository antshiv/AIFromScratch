import json
import csv
import torch
from torch.utils.tensorboard import SummaryWriter

def read_computational_graph(file_path):
    with open(file_path, 'r') as f:
        graph_data = json.load(f)
    return graph_data

def read_loss_values(file_path):
    steps = []
    losses = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            step, loss = int(row[0]), float(row[1])
            steps.append(step)
            losses.append(loss)
    return steps, losses

def log_computational_graph(writer, graph_data):
    # Create a dummy model to log the graph
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.input = torch.nn.Parameter(torch.randn(1, 3))
            self.weights = torch.nn.Parameter(torch.randn(3, 1))
            self.bias = torch.nn.Parameter(torch.randn(1))
        
        def forward(self, x):
            x = torch.matmul(x, self.weights)
            return x + self.bias

    model = DummyModel()
    dummy_input = torch.randn(1, 3)
    writer.add_graph(model, dummy_input)

def main():
    graph_file_path = "computational_graph.json"
    loss_file_path = "loss_values.csv"
    
    graph_data = read_computational_graph(graph_file_path)
    steps, losses = read_loss_values(loss_file_path)
    
    writer = SummaryWriter(log_dir='./tensorboard_logs')
    
    for step, loss in zip(steps, losses):
        writer.add_scalar('Loss', loss, step)
    
    log_computational_graph(writer, graph_data)
    
    writer.close()
    print("TensorBoard logs written to ./tensorboard_logs")

if __name__ == "__main__":
    main()
