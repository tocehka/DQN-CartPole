import torch.nn as nn
import torch.nn.functional as F

class Q_Model(nn.Module):
    def __init__(self, actions_dim, states_dim):
        super().__init__()
        self.fc1 = nn.Linear(states_dim, 80)
        self.fc2 = nn.Linear(80, actions_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
        