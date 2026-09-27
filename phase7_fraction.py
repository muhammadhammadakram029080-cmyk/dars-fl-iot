# phase7_fraction.py
import torch
import numpy as np
from run_manifest import RunManifest, initialize_seeds
from server import DARSServer
from client import create_clients, train_client, evaluate_global

def run_scalability_sweep():
    initialize_seeds(42)
    
    fractions = [0.0, 0.1, 0.2, 0.3, 0.4]
    num_clients = 10
    rounds = 5
    
    for f in fractions:
        num_malicious = int(f * num_clients)
        malicious_indices = list(range(num_clients - num_malicious, num_clients))
        
        manifest = RunManifest(
            run_id=f"phase7_fraction_{f}",
            seed=42,
            dataset="IoTID20",
            hyperparameters={"clients": num_clients, "rounds": rounds, "adversarial_fraction": f}
        )
        
        clients = create_clients(num_clients, malicious_indices=malicious_indices)
        server = DARSServer(omega_1=0.6, omega_2=0.4, alpha=0.6, use_trust=True)
        
        print(f"\n--- Running Adversarial Fraction: {f} ({num_malicious} malicious) ---")
        for r in range(rounds):
            updates = []
            for idx, client in enumerate(clients):
                updates.append((idx, train_client(client)))
                
            global_model, reputations = server.aggregate(updates)
            
        acc, loss = evaluate_global(global_model)
        print(f"Final Accuracy for f={f}: {acc:.4f}")
        
        manifest.write(f"run-manifests/phase7_fraction_{f}.json")

if __name__ == "__main__":
    run_scalability_sweep()
