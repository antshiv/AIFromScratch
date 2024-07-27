import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNN()
writer = SummaryWriter('runs/simple_nn')

# Dummy input for visualization
dummy_input = torch.randn(1, 784)
writer.add_graph(model, dummy_input)
writer.close()

