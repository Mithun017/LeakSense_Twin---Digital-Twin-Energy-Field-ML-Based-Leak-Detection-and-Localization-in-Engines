import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torch_geometric.nn as gnn
except ImportError:
    gnn = None # Fallback if torch-geometric not installed correctly

class LeakSenseNet(nn.Module):
    def __init__(self, input_dim=67, hidden_dim=128):
        super().__init__()
        # input_dim: 31 (base) + 36 (energy field flattened) = 67
        self.W_cos = nn.Linear(input_dim, hidden_dim, bias=True)
        self.W_relu = nn.Linear(hidden_dim, 64, bias=True)
        self.sigmoid_out = nn.Linear(64, 1, bias=True)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        cos_features = F.normalize(self.W_cos(x), dim=1)
        relu_features = F.relu(self.W_relu(cos_features))
        relu_features = self.dropout(relu_features)
        logit = self.sigmoid_out(relu_features)
        return torch.sigmoid(logit).squeeze()

class LeakLocalizationNet(nn.Module):
    def __init__(self, input_dim=67, n_classes=6):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_classes)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return F.softmax(self.fc3(x), dim=1)

class SensorGNN(nn.Module):
    def __init__(self, node_features=3, hidden=32, n_classes=6):
        super().__init__()
        if gnn:
            self.conv1 = gnn.GCNConv(node_features, hidden)
            self.conv2 = gnn.GCNConv(hidden, hidden)
            self.classifier = nn.Linear(hidden * 10, n_classes) # Assuming 10 sensors/nodes
        else:
            self.fc = nn.Linear(node_features * 10, n_classes)
    
    def forward(self, x, edge_index):
        if gnn:
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = x.flatten()
            return F.softmax(self.classifier(x.unsqueeze(0)), dim=1)
        else:
            x = x.flatten()
            return F.softmax(self.fc(x.unsqueeze(0)), dim=1)
