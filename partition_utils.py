import os
import argparse
import numpy as np
import pandas as pd
from datasets import Dataset
from flwr_datasets.partitioner import (
    IidPartitioner,
    DirichletPartitioner,
    ShardPartitioner,
)

def load_npz_data(numpy_file):
    """Load data from npz file (x: images, y: labels)."""
    data = np.load(numpy_file)
    x = data['x']  # Shape: (50000, 32, 32, 3) HARDCODED FOR CIFAR10!!!
    y = data['y']  # Shape: (50000, 1) HARDCODED FOR CIFAR10!!!
    return x, y

def create_dataset_from_arrays(x, y):
    """Create a HuggingFace Dataset from image arrays and labels."""
    # Flatten images for DataFrame creation
    x_flat = x.reshape(x.shape[0], -1)
    # Create feature column names
    feature_cols = [f'pixel_{i}' for i in range(x_flat.shape[1])]
    # Create DataFrame
    df = pd.DataFrame(x_flat, columns=feature_cols)
    # Add label column - flatten y for DataFrame
    df['label'] = y[:, 0]
    # Create HuggingFace Dataset
    return Dataset.from_pandas(df)

def partition_data(x, y, num_partitions, strategy, alpha=None, num_shards_per_client=None):
    """Create partitions using the specified strategy."""
    # Shuffle dataset (just to handle stl10 labeled data, comment if not using stl10)
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    # Create a Dataset
    dataset = create_dataset_from_arrays(x, y)
    
    # Select partitioner based on strategy
    if strategy == "iid":
        partitioner = IidPartitioner(num_partitions=num_partitions)
    elif strategy == "dirichlet":
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions, 
            alpha=alpha,
            partition_by="label"
        )
    elif strategy == "shard":
        partitioner = ShardPartitioner(
            num_partitions=num_partitions,
            num_shards_per_partition=num_shards_per_client,
            partition_by="label"
        )
    
    # Set the dataset for the partitioner
    partitioner.dataset = dataset
    
    # Load each partition
    partitions = []
    for i in range(num_partitions):
        partition = partitioner.load_partition(partition_id=i)
        partitions.append(partition)
    
    return partitions

def save_partition_to_npz(partition, save_path):
    """Save a partition as an npz file with 'x' and 'y' arrays."""
    # Convert to pandas DataFrame
    df = partition.to_pandas()
    
    # Extract labels and reshape to (n, 1) to match original format
    y = df['label'].values[:, np.newaxis]
    
    # Extract and reshape features back to images
    feature_cols = [col for col in df.columns if col.startswith('pixel_')]
    x_flat = df[feature_cols].values
    x = x_flat.reshape(-1, 32, 32, 3)  # Reshape to original image dimensions HARDCODED FOR CIFAR10!!!
    
    # Save as npz
    np.savez(save_path, x=x, y=y)

def main():
    parser = argparse.ArgumentParser(description="Partition an image dataset")
    parser.add_argument("--data_path", required=True, help="Path to the train.npz file")
    parser.add_argument("--save_dir", required=True, help="Directory to save partitions")
    parser.add_argument("--num_partitions", type=int, default=10, help="Number of partitions")
    parser.add_argument(
        "--strategy", 
        choices=["iid", "dirichlet", "shard"], 
        default="iid",
        help="Partitioning strategy"
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha for Dirichlet partitioning")
    parser.add_argument("--num_shards_per_client", type=int, default=2, help="Shards per client")
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.data_path}")
    x, y = load_npz_data(args.data_path)
    print(f"Loaded {len(x)} images with shape {x.shape[1:]}")
    
    # Create partitions
    print(f"Creating {args.num_partitions} partitions using {args.strategy} strategy")
    partitions = partition_data(
        x, y, 
        args.num_partitions, 
        args.strategy, 
        args.alpha, 
        args.num_shards_per_client
    )
    
    # Save partitions
    print(f"Saving partitions to {args.save_dir}")
    for i, partition in enumerate(partitions):
        save_path = os.path.join(args.save_dir, f"partition_{i}.npz")
        save_partition_to_npz(partition, save_path)
        print(f"Saved partition {i} with {len(partition)} samples")
    
    print("Done!")

if __name__ == "__main__":
    main()