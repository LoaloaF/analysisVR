#!/usr/bin/env python3
"""
Generate synthetic data with the same columns as behavior_glm_input_flat.csv
All numeric values are sampled noisily from a 4-dimensional hypersphere 
and embedded into 11D space, with coordinates spread across the columns.
"""

import numpy as np
import pandas as pd
import argparse


def sample_from_2d_hypersphere(n_samples, radius=1.0, noise_scale=0.1):
    """
    Sample points from a 2D hypersphere with noise.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    radius : float
        Radius of the hypersphere (default: 1.0)
    noise_scale : float
        Scale of Gaussian noise added to the samples (default: 0.1)
    
    Returns:
    --------
    samples : np.ndarray
        Array of shape (n_samples, 2) with samples from the hypersphere
    """
    # Sample uniformly from the surface of a 2D hypersphere
    # Method: sample from 2D standard normal, then normalize to unit sphere, then scale by radius
    samples = np.random.randn(n_samples, 2)
    norms = np.linalg.norm(samples, axis=1, keepdims=True)
    samples = samples / norms * radius

    # Compute the angle theta for each sample (arctan2 returns angle in radians)
    labels = np.arctan2(samples[:, 1], samples[:, 0]) + np.pi/2
    
    # Add Gaussian noise
    noise = np.random.randn(n_samples, 2) * noise_scale
    samples = samples + noise
    
    return samples, labels


def embed_2d_to_d_dimensions(samples_2d, d_dimensions=11, embedding_seed=42):
    """
    Embed 2D hypersphere samples into d_dimensions space using a random linear transformation.
    
    Parameters:
    -----------
    samples_2d : np.ndarray
        Array of shape (n_samples, 2) with 2D hypersphere samples
    embedding_seed : int
        Random seed for the embedding matrix (default: 42)
    
    Returns:
    --------
    samples_d_dimensions : np.ndarray
        Array of shape (n_samples, d_dimensions) with embedded samples
    """
    np.random.seed(embedding_seed)
    # Create a random embedding matrix: d_dimensions x 2
    # This matrix maps 2D vectors to d_dimensions space
    embedding_matrix = np.random.randn(d_dimensions, 2)
    
    # Apply embedding: samples_d_dimensions = samples_2d @ embedding_matrix.T
    samples_d_dimensions = samples_2d @ embedding_matrix.T
    
    return samples_d_dimensions


def generate_synthetic_data(input_csv, output_csv, n_samples=None, radius=1.0, noise_scale=0.1, embedding_seed=42):
    """
    Generate synthetic data with the same columns as the input CSV.
    Samples from a 2D hypersphere, embeds into d_dimensions, and spreads coordinates across columns.
    
    Parameters:
    -----------
    input_csv : str
        Path to the input CSV file to get column names from
    output_csv : str
        Path to save the generated synthetic data
    n_samples : int, optional
        Number of samples to generate. If None, uses the same number as input file.
    radius : float
        Radius of the 2D hypersphere (default: 1.0)
    noise_scale : float
        Scale of noise added to hypersphere samples (default: 0.1)
    embedding_seed : int
        Random seed for the 2D to d_dimensions embedding matrix (default: 42)
    """
    # Read the input CSV to get column names
    print(f"Reading column names from {input_csv}...")
    input_data = pd.read_csv(input_csv, nrows=0)  # Only read header
    
    if n_samples is None:
        # Count rows in original file
        n_samples = sum(1 for _ in open(input_csv)) - 1  # Subtract header
        print(f"Using {n_samples} samples (matching input file size)")
    else:
        print(f"Generating {n_samples} samples")
    
    columns = input_data.columns.tolist()
    n_columns = len(columns)
    d_dimensions = n_columns - 3 # 3 columns are timestamp columns
    
    print(f"Found {n_columns} columns")
    
    # Sample from 4D hypersphere
    print("Sampling from 2D hypersphere...")
    samples_2d, labels = sample_from_2d_hypersphere(n_samples, radius=radius, noise_scale=noise_scale)
    
    # Embed into d_dimensions space
    print(f"Embedding 2D samples into {d_dimensions}D space...")
    samples_d_dimensions = embed_2d_to_d_dimensions(samples_2d, d_dimensions=d_dimensions, embedding_seed=embedding_seed)
    
    # Assign values to each column from the d_dimensions embedded samples
    for i, col in enumerate(columns[3:]):
        values = samples_d_dimensions[:, i]

        input_data.iloc[:, input_data.columns.get_loc(col)] = values

    # Add the session_id column from the original data
    original_data = pd.read_csv(input_csv, usecols=["session_id"])
    input_data["session_id"] = original_data["session_id"].values[:n_samples]
    
    # Ensure timestamp columns are reasonable (positive integers)
    timestamp_cols = [col for col in columns if 'timestamp' in col.lower()]
    for col in timestamp_cols:
        if col in input_data.columns:
            # Make timestamps positive and monotonically increasing
            values = (labels * 1000000).astype(int)
            input_data.iloc[:, input_data.columns.get_loc(col)] = values
    
    # Save to CSV
    print(f"Saving synthetic data to {output_csv}...")
    input_data.to_csv(output_csv, index=False)
    print(f"Done! Generated {n_samples} samples with {d_dimensions}D space.")
    print(f"\nSummary statistics:")
    print(input_data.describe())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic data by embedding 4D hypersphere into 11D and spreading coordinates across columns"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="behavior_glm_input_flat.csv",
        help="Input CSV file to get column names from (default: behavior_glm_input_flat.csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="behavior_glm_input_flat_synthetic.csv",
        help="Output CSV file name (default: behavior_glm_input_flat_synthetic.csv)"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Number of samples to generate (default: same as input file)"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=1.0,
        help="Radius of the 4D hypersphere (default: 1.0)"
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=0.1,
        help="Scale of noise added to hypersphere samples (default: 0.1)"
    )
    parser.add_argument(
        "--embedding_seed",
        type=int,
        default=42,
        help="Random seed for the 4D to 11D embedding matrix (default: 42)"
    )
    
    args = parser.parse_args()
    
    generate_synthetic_data(
        input_csv=args.input,
        output_csv=args.output,
        n_samples=args.n_samples,
        radius=args.radius,
        noise_scale=args.noise_scale,
        embedding_seed=args.embedding_seed
    )

