import torch
from torch_geometric.data import Data

def collatz_next_step(n):
    if n % 2 == 0:
        return n // 2
    else:
        return 3 * n + 1
    
def create_collatz_graph(max_steps):
    edge_list = []
    node_features = []
    labels = []
    node_map = {}

    def add_node(n):
        if n not in node_map:
            node_map[n] = len(node_map) 
            node_features.append([n % 2, torch.log(torch.tensor(n, dtype=torch.float32) + 1.00)])

    for n in range(1, max_steps):
        steps = 0
        current = n
        add_node(n)
        
        while current != 1:
            next_step = collatz_next_step(current)
            add_node(next_step)
            edge_list.append([node_map[current], node_map[next_step]])
            current = next_step
            steps += 1
        
        labels.append([steps])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.float)
    
    print("Graph generation complete.")
    print("Final edge connections:")
    for edge in edge_list:
        print(f"{edge[0]} -> {edge[1]}")
    
    return Data(x=x, edge_index=edge_index, y=y)


max_n = 100
data = create_collatz_graph(max_n)

print("Node features (x):", data.x)
print("Edge index:", data.edge_index)
print("Target labels (y):", data.y)
