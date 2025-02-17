import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

class CollatzGNN(nn.Module):
    def __init__(self):
        super(CollatzGNN, self).__init__()
        self.conv1 = GCNConv(2, 16) 
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, 16)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index) 
        return x
    
class FinalPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


def collatz_steps(n):
    steps = 0
    while n != 1:
        if n % 2 == 0:
            n //= 2
        else:
            n = 3 * n + 1
        steps += 1
    return steps

data_list = list(range(1, 20))
correct_list = [collatz_steps(i) for i in data_list]

feature_vectors = torch.tensor([
    [i % 2, torch.log(torch.tensor(i, dtype=torch.float32) + 1.00)]
    for i in data_list
], dtype=torch.float)

# Load checkpoint properly
checkpoint_path = "/Users/gromeronaranjo/Desktop/mathematical-infinite-sequence/gnn-approach/model_checkpoint.pth"
checkpoint = torch.load(checkpoint_path)

gnn = CollatzGNN()
predictor = FinalPredictor()

gnn.load_state_dict(checkpoint["gnn_state_dict"])
predictor.load_state_dict(checkpoint["predictor_state_dict"])

gnn.eval()
predictor.eval()

edge_index = checkpoint.get("edge_index", torch.empty((2, 0), dtype=torch.long)) 

data = Data(x=feature_vectors, edge_index=edge_index)

with torch.no_grad():
    x = gnn(data)
    x = predictor(x)

predicted_steps = x.tolist()

indices = list(range(1, 20))
plt.plot(indices, correct_list, color="blue", label="True Steps")
plt.plot(indices, predicted_steps, color="red", linestyle='dashed', label="Predicted Steps")
plt.xlabel("Number")
plt.ylabel("Steps to 1")
plt.legend()
plt.show()
