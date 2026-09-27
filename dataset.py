import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def load_and_partition_data(filepath, num_clients):
    print("Loading IoTID20 dataset...")
    df = pd.read_csv(filepath)
    
    # CLEANING FIX: Convert infinities to NaN, then drop all missing values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    
    # Isolate flow-based features from the target label
    X = df.drop(columns=['Label'])
    y = df['Label']
    
    # Exclude categorical columns from MinMax scaling
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X_numeric = X[numeric_cols]
    
    # Normalize features using MinMax Scaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_numeric)
    
    # Encode categorical attack vectors
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    # Partition data equally across the simulated clients
    partition_size = len(X_scaled) // num_clients
    client_datasets = []
    
    for i in range(num_clients):
        start_idx = i * partition_size
        end_idx = start_idx + partition_size
        X_part = X_scaled[start_idx:end_idx]
        y_part = y_encoded[start_idx:end_idx]
        client_datasets.append((X_part, y_part))
        
    return client_datasets, encoder.classes_

if __name__ == "__main__":
    # 4 clients are specified to align with the VRAM limits of the RTX 3050 GPU
    datasets, labels = load_and_partition_data("IoT Network Intrusion Dataset.csv", num_clients=4)
    print(f"Successfully created {len(datasets)} client partitions.")
    print(f"Client 1 data shape: {datasets[0][0].shape}")