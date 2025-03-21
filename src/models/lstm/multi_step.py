import torch
import torch.nn as nn

# Model parameters
input_dim = 1
hidden_dim = 256
num_layers = 3
output_dim = 1

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 30)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)  # Initial hidden state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)  # Initial cell state
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc1(out[:,-1, :]) # take the last hidden state because we assume it encodes all the information about the sequence
        return out.unsqueeze(2)

def create_lstm_single_step():
    return LSTMModel(input_dim, hidden_dim, num_layers, output_dim)