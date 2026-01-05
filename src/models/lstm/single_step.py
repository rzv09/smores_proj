import torch
import torch.nn as nn

# Model parameters
# hidden_dim = 128
# num_layers = 3
output_dim = 1
# hidden layer is [3 x 128]

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=False)
        self.fc = nn.Linear(hidden_dim, output_dim)
        # self.do = nn.Dropout()
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, self.hidden_dim).to(x.device)  # Initial hidden state
        c0 = torch.zeros(self.num_layers, self.hidden_dim).to(x.device)  # Initial cell state
        out, _ = self.lstm(x, (h0, c0))
        # out = self.do(out)
        out = self.fc(out[-1, :])  # Take the last point
        return out


class MultiFeatureLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, num_features):
        super(MultiFeatureLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_features = num_features
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=False)
        # self.lstm2 = nn.LSTM(hidden_dim, )
        self.fc = nn.Linear(hidden_dim, output_dim)
        # self.do = nn.Dropout()
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, 1, self.hidden_dim).to(x.device)  # Initial hidden state
        c0 = torch.zeros(self.num_layers, 1, self.hidden_dim).to(x.device)  # Initial cell state
        x = x.permute(1, self.num_features, 0)
        out, _ = self.lstm(x, (h0, c0))
        # out = self.do(out)
        out = self.fc(out[-1, :])  # Take the last point
        return out

def create_lstm_single_step(input_dim: int = 1, num_features: int = 1, hidden_dim=16, num_layers=1):
    if num_features == 1:
        return LSTMModel(input_dim, hidden_dim, num_layers, output_dim)
    else:
        return MultiFeatureLSTMModel(input_dim, hidden_dim, num_layers, output_dim, num_features)