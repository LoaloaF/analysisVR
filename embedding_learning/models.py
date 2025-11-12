import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_size, hidden_size),    
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )

class LinearAutoEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearAutoEncoder, self).__init__()
        self.encoder = nn.Linear(input_size,  output_size)
        self.decoder = nn.Linear(output_size, input_size)