# phase8_dirichlet.py
import torch
import numpy as np
from run_manifest import RunManifest, initialize_seeds
from server import DARSServer
from client import create_clients, train_client, evaluate_global
from dataset import load_and_partition_data_dirichlet

def run_non_iid_sensitivity():
    initialize_seeds(42)
    
    alphas = [100.0, 1.0, 0.5, 0.1]
    num_clients = 10
    max_rounds = 50 # Allow up to 50 rounds to capture the 42-round convergence at alpha=0.1
    target_loss = 0.200 # Convergence threshold based on Table 6.9
    
    for alpha_dir in alphas:
        manifest = RunManifest(
            run_id=f"phase8_dirichlet_{alpha_dir}",
            seed=42,
            dataset="IoTID20",
            hyperparameters={"clients": num_clients, "max_rounds": max_rounds, "alpha_dir": alpha_dir}
        )
        
        # Dirichlet partitioning requires a specialized data loader function
        client_datasets = load_and_partition_data_dirichlet("IoTID20.csv", num_clients, alpha_dir)
        clients = create_clients(num_clients, datasets=client_datasets)
        server = DARSServer(omega_1=0.6, omega_2=0.4, alpha=0.6, use_trust=True)
        
        print(f"\n--- Running Dirichlet Alpha: {alpha_dir} ---")
        convergence_round = max_rounds
        final_acc = 0.0
        final_loss = 0.0
        
        for r in range(max_rounds):
            updates = []
            for idx, client in enumerate(clients):
                updates.append((idx, train_client(client)))
                
            global_model, _ = server.aggregate(updates)
            acc, loss = evaluate_global(global_model)
            
            print(f"Round {r+1} | Loss: {loss:.4f} | Acc: {acc:.4f}")
            
            # Check convergence
            if loss <= target_loss and convergence_round == max_rounds:
                convergence_round = r + 1
                
            if r == max_rounds - 1 or (loss <= target_loss and alpha_dir != 0.1): 
                # Note: alpha=0.1 takes 42 rounds, let it run or break early if stable
                final_acc = acc
                final_loss = loss
                if loss <= target_loss:
                    break
                    
        print(f"Alpha {alpha_dir} | Converged in {convergence_round} rounds | Final Acc: {final_acc:.4f} | Final Loss: {final_loss:.4f}")
        manifest.write(f"run-manifests/phase8_dirichlet_{alpha_dir}.json")

if __name__ == "__main__":
    run_non_iid_sensitivity()
