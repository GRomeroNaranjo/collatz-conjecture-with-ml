import torch
import matplotlib.pyplot as plt
import numpy as np

weights_path = "/Users/gromeronaranjo/Desktop/mathematical-infinite-sequence/model_weights.pth"
weights = torch.load(weights_path, map_location=torch.device('cpu'))

print(weights.keys())

np.random.seed(42) # For reproducibility


layer_name = "fc1.weight"
weight_tensor = weights[layer_name]
print(weight_tensor)
weight_numpy = weight_tensor.cpu().detach().numpy().flatten()

x_positions = np.random.normal(loc=0, scale=0.1, size=len(weight_numpy))

plt.figure(figsize=(10, 6))
plt.scatter(x_positions, weight_numpy, alpha=0.6, color="red", s=10)
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
plt.title(f"Scatter Plot of Weights - {layer_name}")
plt.xlabel("Random Jitter (to separate points)")
plt.ylabel("Weight Value")
plt.show()

layer_name = "fc1.bias"
weight_tensor = weights[layer_name]
print(weight_tensor)
weight_numpy = weight_tensor.cpu().detach().numpy().flatten()

x_positions = np.random.normal(loc=0, scale=0.1, size=len(weight_numpy))

plt.figure(figsize=(10, 6))
plt.scatter(x_positions, weight_numpy, alpha=0.6, color="red", s=10)
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
plt.title(f"Scatter Plot of Weights - {layer_name}")
plt.xlabel("Random Jitter (to separate points)")
plt.ylabel("Weight Value")
plt.show()

layer_name = "fc2.weight"
weight_tensor = weights[layer_name]
print(weight_tensor)
weight_numpy = weight_tensor.cpu().detach().numpy().flatten()

x_positions = np.random.normal(loc=0, scale=0.1, size=len(weight_numpy))

plt.figure(figsize=(10, 6))
plt.scatter(x_positions, weight_numpy, alpha=0.6, color="red", s=10)
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
plt.title(f"Scatter Plot of Weights - {layer_name}")
plt.xlabel("Random Jitter (to separate points)")
plt.ylabel("Weight Value")
plt.show()

layer_name = "fc2.bias"
weight_tensor = weights[layer_name]
print(weight_tensor)
weight_numpy = weight_tensor.cpu().detach().numpy().flatten()

x_positions = np.random.normal(loc=0, scale=0.1, size=len(weight_numpy))

plt.figure(figsize=(10, 6))
plt.scatter(x_positions, weight_numpy, alpha=0.6, color="red", s=10)
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
plt.title(f"Scatter Plot of Weights - {layer_name}")
plt.xlabel("Random Jitter (to separate points)")
plt.ylabel("Weight Value")
plt.show()

layer_name = "fc3.weight"
weight_tensor = weights[layer_name]
print(weight_tensor)
weight_numpy = weight_tensor.cpu().detach().numpy().flatten()

x_positions = np.random.normal(loc=0, scale=0.1, size=len(weight_numpy))

plt.figure(figsize=(10, 6))
plt.scatter(x_positions, weight_numpy, alpha=0.6, color="red", s=10)
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
plt.title(f"Scatter Plot of Weights - {layer_name}")
plt.xlabel("Random Jitter (to separate points)")
plt.ylabel("Weight Value")
plt.show()

layer_name = "fc3.bias"
weight_tensor = weights[layer_name]
print(weight_tensor)
weight_numpy = weight_tensor.cpu().detach().numpy().flatten()

x_positions = np.random.normal(loc=0, scale=0.1, size=len(weight_numpy))

plt.figure(figsize=(10, 6))
plt.scatter(x_positions, weight_numpy, alpha=0.6, color="red", s=10)
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
plt.title(f"Scatter Plot of Weights - {layer_name}")
plt.xlabel("Random Jitter (to separate points)")
plt.ylabel("Weight Value")
plt.show()

layer_name = "fc4.weight"
weight_tensor = weights[layer_name]
print(weight_tensor)
weight_numpy = weight_tensor.cpu().detach().numpy().flatten()

x_positions = np.random.normal(loc=0, scale=0.1, size=len(weight_numpy))

plt.figure(figsize=(10, 6))
plt.scatter(x_positions, weight_numpy, alpha=0.6, color="red", s=10)
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
plt.title(f"Scatter Plot of Weights - {layer_name}")
plt.xlabel("Random Jitter (to separate points)")
plt.ylabel("Weight Value")
plt.show()

layer_name = "fc4.bias"
weight_tensor = weights[layer_name]
print(weight_tensor)
weight_numpy = weight_tensor.cpu().detach().numpy().flatten()

x_positions = np.random.normal(loc=0, scale=0.1, size=len(weight_numpy))

plt.figure(figsize=(10, 6))
plt.scatter(x_positions, weight_numpy, alpha=0.6, color="red", s=10)
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
plt.title(f"Scatter Plot of Weights - {layer_name}")
plt.xlabel("Random Jitter (to separate points)")
plt.ylabel("Weight Value")
plt.show()
