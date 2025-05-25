import os
import argparse
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import yaml

def split_parquet_dataset(
    input_file,
    output_train,
    output_val, 
    val_size=0.2, 
    random_seed=42
):
    print(f"Loading dataset from {input_file}")
    dataset = load_dataset("parquet", data_files={"train": input_file})["train"]
    
    indices = np.arange(len(dataset))
    train_indices, val_indices = train_test_split(
        indices, test_size=val_size, random_state=random_seed
    )
    
    train_dataset = dataset.select(train_indices)
    val_dataset = dataset.select(val_indices)
    
    os.makedirs(os.path.dirname(output_train), exist_ok=True)
    os.makedirs(os.path.dirname(output_val), exist_ok=True)
    
    print(f"Saving training set to {output_train} ({len(train_dataset)} examples)")
    train_dataset.to_parquet(output_train)
    
    print(f"Saving validation set to {output_val} ({len(val_dataset)} examples)")
    val_dataset.to_parquet(output_val)
    
    print("\nStatistics:")
    print(f"  Total examples: {len(dataset)}")
    print(f"  Training examples: {len(train_dataset)} ({len(train_dataset)/len(dataset)*100:.2f}%)")
    print(f"  Validation examples: {len(val_dataset)} ({len(val_dataset)/len(dataset)*100:.2f}%)")
    
    return {
        "train_file": output_train,
        "val_file": output_val,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "total_samples": len(dataset)
    }

def update_config(config_path, split_info):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    for dataset in config['data']:
        if 'paths' in config['data'][dataset] and 'train_file' in config['data'][dataset]['paths']:
            config['data'][dataset]['paths']['val_file'] = split_info['val_file']
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Updated config file: {config_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split a parquet dataset into train and validation sets')
    parser.add_argument('--input', type=str, required=True, 
                        help='Path to input parquet file')
    parser.add_argument('--output-train', type=str, required=True,
                        help='Path to save training parquet file')
    parser.add_argument('--output-val', type=str, required=True, 
                        help='Path to save validation parquet file')
    parser.add_argument('--val-size', type=float, default=0.2,
                        help='Proportion of validation set (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--config', type=str, default='config.yml',
                        help='Path to configuration file to be updated')
    
    args = parser.parse_args()
    
    split_info = split_parquet_dataset(
        args.input, 
        args.output_train,
        args.output_val,
        args.val_size,
        args.seed
    )
   
    update_config(args.config, split_info)