import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import data

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

class Trainer(nn.Module):
    def __init__(self, gnn, predictor, criterion, lr):
        super().__init__()
        self.gnn = gnn
        self.predictor = predictor
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(list(gnn.parameters()) + list(predictor.parameters()), lr=lr)
        self.losses = []

    def train(self, data, epochs):
        for epoch in range(epochs):
            self.gnn.train()
            self.predictor.train()
            self.optimizer.zero_grad()
            
            embeddings = self.gnn(data)
            output = self.predictor(embeddings)
            loss = self.criterion(output, data.y.view(-1))
            loss.backward()
            self.optimizer.step()

            self.losses.append(loss.item()) 
            print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
        
        torch.save({
            'gnn_state_dict': self.gnn.state_dict(),
            'predictor_state_dict': self.predictor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.losses
        }, 'model_checkpoint.pth')
        print("Model parameters and loss history saved successfully!")

    def save_losses(self, filename='loss_history.pth'):
        torch.save(self.losses, filename)
        print(f"Loss history saved to {filename}")

    def load_losses(self, filename='loss_history.pth'):
        self.losses = torch.load(filename)
        print(f"Loss history loaded from {filename}")

data = data.create_collatz_graph(20)
gnn = CollatzGNN()
predictor = FinalPredictor()
criterion = nn.SmoothL1Loss()

trainer = Trainer(gnn, predictor, criterion, 0.01)
trainer.train(data, 500)
trainer.save_losses()