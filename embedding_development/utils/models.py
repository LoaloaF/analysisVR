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

    def embed(self, x):
        """Activations of the last hidden layer (before the output projection)."""
        return self.fc[:-1](x)

    def forward(self, x):
        h = self.embed(x)
        return h, self.fc[-1](h)

class NeuronLSTM(torch.nn.Module):
    """
    Causal LSTM: given a window of T past behavioural frames,
    predict the firing rate at the final timestep.

    The action embedding is the LSTM output at the final timestep
    (out[:, -1, :], shape hidden_size), extracted via embed().
    The FC layer maps this embedding to neuron predictions.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size=1, dropout=0.0):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,       # (batch, seq, features)
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def embed(self, x):
        """Return the LSTM output at the final timestep: (batch, hidden_size)."""
        out, _ = self.lstm(x)
        return out[:, -1, :]

    def forward(self, x):
        # x: (batch, window, input_size)
        last = self.embed(x)             # (batch, hidden_size)
        return self.fc(self.dropout(last)).squeeze(-1)  # (batch,) or (batch, output_size)
