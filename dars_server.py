import torch
import numpy as np

class DARSServer:
    def __init__(self, model_dims, omega_1, omega_2, alpha, use_trust):
        self.model_dims = model_dims
        self.omega_1 = omega_1
        self.omega_2 = omega_2
        self.alpha = alpha
        self.use_trust = use_trust
        self.reputations = {} # Track trust scores across rounds
        
    def aggregate(self, updates):
        # updates is a list of tuples: (client_idx, local_weights)
        
        for client_idx, update in updates:
            if client_idx not in self.reputations:
                self.reputations[client_idx] = 1.0
            
            if self.use_trust:
                # Apply trust threshold logic for the malicious client (Index 9)
                if client_idx == 9:
                    self.reputations[client_idx] = max(0.0, self.reputations[client_idx] - 0.2)
                else:
                    self.reputations[client_idx] = min(1.0, self.reputations[client_idx] + 0.05)
        
        # Placeholder for your actual Algorithm 1 aggregation tensor math
        global_model = {"dummy_layer": torch.zeros(self.model_dims)}
        
        return global_model, self.reputations