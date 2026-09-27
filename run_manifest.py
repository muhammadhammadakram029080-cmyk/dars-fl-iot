import json
import time
import hashlib
import platform
import sys
import os
from contextlib import contextmanager
import numpy as np
import torch

def initialize_seeds(seed):
    """Initializes deterministic execution seeds across standard libraries."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class RunManifest:
    def __init__(self, run_id, seed, dataset, hyperparameters):
        self.manifest = {
            "run_id": run_id,
            "seed": seed,
            "dataset": dataset,
            "hyperparameters": hyperparameters,
            "environment": self._capture_environment(),
            "timings": {},
            "data_splits": {}
        }

    def _capture_environment(self):
        """Captures hardware and software environment details for reproducibility."""
        env = {
            "python_version": sys.version.split()[0],
            "pytorch_version": torch.__version__,
            "numpy_version": np.__version__,
            "os": platform.system(),
            "os_release": platform.release(),
            "cuda_available": torch.cuda.is_available(),
        }
        if env["cuda_available"]:
            env["cuda_device_name"] = torch.cuda.get_device_name(0)
        return env

    @contextmanager
    def time_block(self, block_name):
        """Context manager to record wall-clock time for specific code blocks."""
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.manifest["timings"][block_name] = elapsed_time

    def record_split(self, split_name, indices, labels=None):
        """Records exact data partition indices, their SHA-256 digest, and optional class distributions."""
        # Ensure indices are a standard numpy array for consistent hashing
        idx_array = np.array(indices, dtype=np.int64)
        
        # Calculate SHA-256 digest
        digest = hashlib.sha256(idx_array.tobytes()).hexdigest()
        
        split_data = {
            "size": len(idx_array),
            "sha256_digest": digest,
            "indices": idx_array.tolist()
        }

        # Calculate class distribution if labels are provided
        if labels is not None:
            lbl_array = np.array(labels)
            unique, counts = np.unique(lbl_array, return_counts=True)
            split_data["class_distribution"] = {str(k): int(v) for k, v in zip(unique, counts)}

        self.manifest["data_splits"][split_name] = split_data

    def write(self, filepath):
        """Writes the compiled manifest to a formatted JSON file."""
        directory = os.path.dirname(filepath)
        if directory:  # Only create directories if a path is specified
            os.makedirs(directory, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.manifest, f, indent=4)
        print(f"Run manifest successfully saved to: {filepath}")
if __name__ == "__main__":
    # Initialize deterministic seeds
    initialize_seeds(42)

    # Create the manifest object with your thesis parameters
    manifest = RunManifest(
        run_id="hec_defense_run",
        seed=42,
        dataset="IoTID20",
        hyperparameters={"rounds": 5, "trust_threshold": 0.4}
    )

    # Example: Pass your dataset indices here to generate the hashes
    # (For your final submission, you will want to replace these with your actual variables from dataset.py)
    train_indices = np.arange(15792)  # 80% of 19740
    test_indices = np.arange(15792, 19740)  # 20% of 19740
    
    # Record the splits to generate the SHA-256 digests
    manifest.record_split("train_split", train_indices)
    manifest.record_split("test_split", test_indices)

    # Save the file to your folder
    manifest.write("run_manifest.json")