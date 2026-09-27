import flwr as fl
import torch
import numpy as np
from dataset import load_and_partition_data
from client import IoTClient
from torch.utils.data import DataLoader, TensorDataset

# Custom Federated Strategy with Dynamic Reputation Scoring
class ReputationStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_reputation = {}

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated_loss, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)
        print(f"\n--- Round {server_round} Dynamic Reputation Update ---")
        
        # Calculate the network average accuracy for this specific round
        accuracies = [eval_res.metrics.get("accuracy", 0.0) for _, eval_res in results]
        network_avg = sum(accuracies) / len(accuracies) if accuracies else 0.0
        
        for client_proxy, eval_res in results:
            client_id = client_proxy.cid
            accuracy = eval_res.metrics.get("accuracy", 0.0)
            
            if client_id not in self.client_reputation:
                self.client_reputation[client_id] = 1.0
                
            # DYNAMIC TRUST LOGIC: Penalize if a node falls 5% below the network average
            if accuracy < (network_avg - 0.05):
                self.client_reputation[client_id] -= 0.3 
                print(f"[WARNING] Client {client_id} fell below dynamic threshold (Avg: {network_avg:.4f}). Penalizing!")
            else:
                self.client_reputation[client_id] = min(1.0, self.client_reputation[client_id] + 0.05)
                
            print(f"Client {client_id}: Accuracy = {accuracy:.4f} | Trust Score = {self.client_reputation[client_id]:.2f}")
            
        return aggregated_loss, metrics_aggregated

def main():
    num_clients = 4
    client_datasets, classes = load_and_partition_data("IoT Network Intrusion Dataset.csv", num_clients=num_clients)
    
    def client_fn(cid: str) -> fl.client.Client:
        client_idx = int(cid)
        X_train, y_train = client_datasets[client_idx]
        
        # PHASE 4: Inject Malicious Behavior into Client 0
        if client_idx == 0:
            print(f"\n[!] Simulating Compromised Node for Client {client_idx} (Label Flipping Attack)")
            y_train = y_train.copy()
            np.random.shuffle(y_train)
        
        tensor_X = torch.tensor(X_train, dtype=torch.float32)
        tensor_y = torch.tensor(y_train, dtype=torch.long)
        
        dataset_size = len(tensor_X)
        train_size = int(0.8 * dataset_size)
        
        train_data = TensorDataset(tensor_X[:train_size], tensor_y[:train_size])
        test_data = TensorDataset(tensor_X[train_size:], tensor_y[train_size:])
        
        trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
        testloader = DataLoader(test_data, batch_size=32, shuffle=False)
        
        num_features = X_train.shape[1]
        num_classes = len(classes)
        
        return IoTClient(trainloader, testloader, num_features, num_classes).to_client()

    strategy = ReputationStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
    )

    print("\nStarting Phase 4: Dynamic Adversarial Simulation...")
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.25},
    )

if __name__ == "__main__":
    main()