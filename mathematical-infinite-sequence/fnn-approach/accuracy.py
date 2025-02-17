import pandas as pd
import torch
from torch import nn
import pickle
import matplotlib.pyplot as plt
import numpy as np

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

X_train = pd.read_csv("/Users/gromeronaranjo/Desktop/mathematical-infinite-sequence/data/numbers.csv")
y_train = pd.read_csv("/Users/gromeronaranjo/Desktop/mathematical-infinite-sequence/data/steps.csv")

X_train = X_train[:50000].values.reshape(-1, 1)
y_train = y_train[:50000].values.reshape(-1, 1)

model = Model()
weights_path = "/Users/gromeronaranjo/Desktop/mathematical-infinite-sequence/model_weights.pth"
model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
model.eval()

with open("/Users/gromeronaranjo/Desktop/mathematical-infinite-sequence/scalers.pkl", "rb") as f:
    scalers = pickle.load(f)

X_train = torch.tensor(X_train, dtype=torch.float32)

output = model(X_train).detach().numpy()

random_indices = np.arange(len(output))
plt.plot(random_indices, y_train, color="blue")
plt.plot(random_indices, output, color="red")
plt.xlabel("Index")
plt.ylabel("Predicted Output")
plt.title("Predictions Scatter Plot")
plt.show()

print("Mean of actual step counts:", np.mean(y_train))
print("Mean of predicted step counts:", np.mean(output))
print("Variance of actual step counts:", np.var(y_train))
print("Variance of predicted step counts:", np.var(output))
