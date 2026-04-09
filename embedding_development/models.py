import torch

class LinearModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearModel, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        return None, self.fc(x)  # same interface as MLP

class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers, output_size):
        assert num_hidden_layers >= 1
        super(MLP, self).__init__()
        self.hidden_size = hidden_size
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            *([torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU()] * (num_hidden_layers - 1)),
            torch.nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return None, self.fc(x)

class NeuronLSTM(torch.nn.Module):
    """
    Causal LSTM: given a window of T past behavioural frames,
    predict the firing rate at the final timestep.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,       # (batch, seq, features)
        )
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, window, input_size)
        out, _ = self.lstm(x)       # out: (batch, window, hidden)
        last    = out[:, -1, :]     # take the last timestep
        return self.fc(last).squeeze(-1)  # (batch,)
