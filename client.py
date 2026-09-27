import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl
from collections import OrderedDict

# Define the 1D-CNN Architecture for IoT Flow Features
class Net(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Net, self).__init__()
        # 1D Convolution for extracting spatial patterns from flow features
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Calculate flattened dimension after conv and pooling
        pooled_dim = input_dim // 2
        self.fc1 = nn.Linear(32 * pooled_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Add channel dimension for 1D CNN: (batch_size, 1, input_dim)
        x = x.unsqueeze(1) 
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(net, trainloader, epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    net.train()
    for _ in range(epochs):
        for data, target in trainloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = net(data.float())
            loss = criterion(output, target.long())
            loss.backward()
            optimizer.step()

def test(net, testloader, device):
    criterion = nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            outputs = net(data.float())
            loss += criterion(outputs, target.long()).item()
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == target).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy

# Define the Federated Learning Client
class IoTClient(fl.client.NumPyClient):
    def __init__(self, trainloader, testloader, num_features, num_classes):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = Net(num_features, num_classes).to(self.device)
        self.trainloader = trainloader
        self.testloader = testloader

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.net, self.trainloader, epochs=1, device=self.device)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader, self.device)
        # The accuracy metric is transmitted to the server for the reputation calculation
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}