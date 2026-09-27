import flwr as fl
import torch
from dataset import load_and_partition_data
from client import IoTClient
from torch.utils.data import DataLoader, TensorDataset

# Custom Federated Strategy with Adaptive Reputation Scoring
class ReputationStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize trust scores for all simulated clients to 1.0 (baseline)
        self.client_reputation = {}

    def aggregate_evaluate(self, server_round, results, failures):
        # Call the parent FedAvg evaluate aggregation
        aggregated_loss, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)
        
        print(f"\n--- Round {server_round} Reputation Update ---")
        for client_proxy, eval_res in results:
            client_id = client_proxy.cid
            accuracy = eval_res.metrics.get("accuracy", 0.0)
            
            # Initialize or update reputation based on threat detection accuracy
            if client_id not in self.client_reputation:
                self.client_reputation[client_id] = 1.0
                
            # Adaptive trust logic: punish low accuracy, reward high accuracy
            if accuracy < 0.70:
                self.client_reputation[client_id] -= 0.2  # Penalize compromised node behavior
            else:
                self.client_reputation[client_id] = min(1.0, self.client_reputation[client_id] + 0.05)
                
            print(f"Client {client_id}: Accuracy = {accuracy:.4f} | New Trust Score = {self.client_reputation[client_id]:.2f}")
            
        return aggregated_loss, metrics_aggregated

def main():
    # 1. Load and partition the IoTID20 dataset
    num_clients = 4
    client_datasets, classes = load_and_partition_data("IoT Network Intrusion Dataset.csv", num_clients=num_clients)
    
    # 2. Define the client builder function for the simulation
    def client_fn(cid: str) -> fl.client.Client:
        # Get the partitioned data for this specific client
        client_idx = int(cid)
        X_train, y_train = client_datasets[client_idx]
        
        # Convert to PyTorch tensors
        tensor_X = torch.tensor(X_train, dtype=torch.float32)
        tensor_y = torch.tensor(y_train, dtype=torch.long)
        
        # Split into training and testing sets (80/20 split)
        dataset_size = len(tensor_X)
        train_size = int(0.8 * dataset_size)
        
        train_data = TensorDataset(tensor_X[:train_size], tensor_y[:train_size])
        test_data = TensorDataset(tensor_X[train_size:], tensor_y[train_size:])
        
        trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
        testloader = DataLoader(test_data, batch_size=32, shuffle=False)
        
        num_features = X_train.shape[1]
        num_classes = len(classes)
        
        return IoTClient(trainloader, testloader, num_features, num_classes)

    # 3. Configure the Adaptive Reputation Strategy
    strategy = ReputationStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
    )

    # 4. Start the Federated Learning Simulation
    print("\nStarting Decentralized Reputation System Simulation...")
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
        # VRAM Management: allocate exactly 25% of the GPU to each of the 4 concurrent clients
        client_resources={"num_cpus": 1, "num_gpus": 0.25},
    )

if __name__ == "__main__":
    main()