# phase6_ablation.py
import torch
import numpy as np
from run_manifest import RunManifest, initialize_seeds
# Assuming server and client modules are implemented in your repository
from dars_server import DARSServer
from client import create_clients, train_client, evaluate_global

def run_ablation_study():
    initialize_seeds(42)
    
    # Variants defined in Table 6.7
    variants = {
        "Full_DARS": {"omega_1": 0.6, "omega_2": 0.4, "alpha": 0.6, "trust_scoring": True},
        "Variant_A_Sdir": {"omega_1": 1.0, "omega_2": 0.0, "alpha": 0.6, "trust_scoring": True},
        "Variant_B_Smag": {"omega_1": 0.0, "omega_2": 1.0, "alpha": 0.6, "trust_scoring": True},
        "Variant_C_NoEMA": {"omega_1": 0.6, "omega_2": 0.4, "alpha": 1.0, "trust_scoring": True},
        "FedAvg": {"omega_1": 0.0, "omega_2": 0.0, "alpha": 1.0, "trust_scoring": False}
    }

    num_clients = 10
    malicious_clients = [9] # 1 targeted adversary as per Phase 4/6
    rounds = 15 # Extended slightly to observe Variant B's 12-round quarantine
    
    for variant_name, params in variants.items():
        manifest = RunManifest(
            run_id=f"phase6_{variant_name}",
            seed=42,
            dataset="IoTID20",
            hyperparameters={"clients": num_clients, "rounds": rounds, **params}
        )
        
        with manifest.time_block("setup"):
            clients = create_clients(num_clients, malicious_indices=malicious_clients)
            server = DARSServer(
                model_dims=65923, # 1D-CNN dimensions
                omega_1=params["omega_1"],
                omega_2=params["omega_2"],
                alpha=params["alpha"],
                use_trust=params["trust_scoring"]
            )
            
        print(f"\n--- Running {variant_name} ---")
        for r in range(rounds):
            updates = []
            with manifest.time_block(f"round_{r}_training"):
                for idx, client in enumerate(clients):
                    update = train_client(client)
                    updates.append((idx, update))
            
            with manifest.time_block(f"round_{r}_aggregation"):
                # Server applies Algorithm 1
                global_model, reputations = server.aggregate(updates)
                
            acc, loss = evaluate_global(global_model)
            print(f"Round {r+1} | Acc: {acc:.4f} | Adv Rep: {reputations.get(9, 'N/A')}")
            
        manifest.write(f"run-manifests/phase6_{variant_name}.json")

if __name__ == "__main__":
    run_ablation_study()
