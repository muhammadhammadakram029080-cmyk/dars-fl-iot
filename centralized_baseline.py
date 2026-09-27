import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from client import Net
import time

def run_centralized():
    print("Loading entire IoTID20 dataset for centralized baseline...")
    df = pd.read_csv("IoT Network Intrusion Dataset.csv")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    
    X = df.drop(columns=['Label'])
    y = df['Label']
    
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X[numeric_cols])
    
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    tensor_X = torch.tensor(X_scaled, dtype=torch.float32)
    tensor_y = torch.tensor(y_encoded, dtype=torch.long)
    
    dataset_size = len(tensor_X)
    train_size = int(0.8 * dataset_size)
    
    trainloader = DataLoader(TensorDataset(tensor_X[:train_size], tensor_y[:train_size]), batch_size=64, shuffle=True)
    testloader = DataLoader(TensorDataset(tensor_X[train_size:], tensor_y[train_size:]), batch_size=64, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net(X_scaled.shape[1], len(encoder.classes_)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    print("Starting Centralized Training (5 Epochs)...")
    start_time = time.time()
    
    for epoch in range(5):
        net.train()
        for data, target in trainloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(net(data.float()), target.long())
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} Completed")
    
    net.eval()
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            outputs = net(data.float())
            correct += (torch.argmax(outputs, dim=1) == target).sum().item()
            
    accuracy = correct / len(testloader.dataset)
    elapsed = time.time() - start_time
    print(f"\n[BASELINE RESULTS] Overall Accuracy: {accuracy:.4f} | Total Training Time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    run_centralized()