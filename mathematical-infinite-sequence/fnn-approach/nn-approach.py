import torch
from torch import nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import pickle

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class Trainer:
    def __init__(self, model, criterion, optimizer, batch_size):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size

    def train(self, x, y, epochs):
        loss_list = []
        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(epochs):
            for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
                self.optimizer.zero_grad()
                output = self.model(batch_x)
                loss = self.criterion(output, batch_y)
                loss.backward()
                self.optimizer.step()
                
                loss_value = loss.item()
                loss_list.append(int(loss_value))
                
                print(f"Epoch: {epoch+1}, Batch: {batch_idx+1}, Loss: {loss.item()}")

        return loss


class System:
    def __init__(self, X_train, y_train, batch_size=32):
        self.X_train = X_train
        self.y_train = y_train
        self.model = Model()
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.trainer = Trainer(self.model, self.criterion, self.optimizer, batch_size)
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.batch_size = batch_size
        self.scaler_fit()

    def scaler_fit(self):
        self.scaler_x.fit(self.X_train)
        self.scaler_y.fit(self.y_train)

    def scale_x(self, x, function):
        if function == 'transform':
            return torch.tensor(self.scaler_x.transform(x), dtype=torch.float32)
        elif function == 'inverse_transform':
            return torch.tensor(self.scaler_x.inverse_transform(x), dtype=torch.float32)
        else:
            raise ValueError("Function must be either 'transform' or 'inverse_transform'")
        
    def scale_y(self, y, function):
        if function == 'transform':
            return torch.tensor(self.scaler_y.transform(y), dtype=torch.float32)
        elif function == 'inverse_transform':
            return torch.tensor(self.scaler_y.inverse_transform(y), dtype=torch.float32)
        else:
            raise ValueError("Function must be either 'transform' or 'inverse_transform'")
        
    def train(self, epochs):
        X_train = self.X_train
        y_train = self.y_train

        #X_train = self.scale_x(self.X_train, "transform")
        #y_train = self.scale_y(self.y_train, "transform")
                      
        loss = self.trainer.train(X_train, y_train, epochs)
        self.save_model()
        return loss

    def predict(self, x):
        x = self.scale_x(x, "transform")
        y = self.model(x)
        return self.scale_y(y, "inverse_transform")

    def save_model(self):
        torch.save(self.model.state_dict(), "model_weights.pth")
        with open("scalers.pkl", "wb") as f:
            pickle.dump({"scaler_x": self.scaler_x, "scaler_y": self.scaler_y}, f)

X_train = pd.read_csv("/Users/gromeronaranjo/Desktop/mathematical-infinite-sequence/data/numbers.csv")
y_train = pd.read_csv("/Users/gromeronaranjo/Desktop/mathematical-infinite-sequence/data/steps.csv")

X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)

system = System(X_train, y_train, batch_size=64)
loss = system.train(epochs=1)
print(loss)

