#!/usr/bin/env python3
"""
Transformer-based model for predicting trend direction in financial indicators.
Uses the last 165 rows, trains on first 150 values, predicts trend between row 151 and last row.
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configuration (will be overridden by command line arguments)
DATA_DIR = "datasindAV"
SEQUENCE_LENGTH = 150
LAST_N_ROWS = 165
PREDICTION_HORIZON = 14  # 165 - 150 - 1 = 14 steps ahead
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.0001
# Device configuration: prioritize MPS (Mac), then CUDA, then CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 4
DIM_FEEDFORWARD = 512
DROPOUT = 0.1
N_DROPOUT_SAMPLES = 50  # Number of forward passes for Bayesian dropout


class TimeSeriesDataset(Dataset):
    """Dataset for time series trend prediction."""
    
    def __init__(self, sequences, labels, column_names):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        self.column_names = column_names
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerTrendPredictor(nn.Module):
    """Lightweight Transformer model for trend prediction."""
    
    def __init__(self, num_targets, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, 
                 dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT):
        super().__init__()
        
        # Input embedding
        self.input_projection = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.dropout_rate = dropout
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers - one classifier per target column
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.num_targets = num_targets
        
        # Separate classifier for each target column
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 2)  # Binary classification: up or down
            ) for _ in range(self.num_targets)
        ])
    
    def enable_dropout(self):
        """Enable dropout for Bayesian inference."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def forward(self, x, target_idx=None):
        # x shape: (batch, sequence_length, 1)
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # Global pooling
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x = self.global_pool(x).squeeze(-1)  # (batch, d_model)
        
        # Classification for specific target or all targets
        if target_idx is not None:
            return self.classifiers[target_idx](x)  # (batch, 2)
        else:
            # Return predictions for all targets
            return [classifier(x) for classifier in self.classifiers]


def load_and_prepare_data(data_dir, target_columns, exclude_year=None):
    """Load Excel files and prepare sequences for training - only for target columns."""
    
    all_files = glob.glob(os.path.join(data_dir, "*.xlsx"))
    
    # Split train/test based on year
    if exclude_year:
        train_files = [f for f in all_files if str(exclude_year) not in f]
        test_files = [f for f in all_files if str(exclude_year) in f]
    else:
        train_files = all_files
        test_files = []
    
    print(f"Train files: {len(train_files)}")
    print(f"Test files: {len(test_files)}")
    print(f"Target columns: {target_columns}")
    
    def process_files(files):
        # Separate data for each target column
        data_by_target = {col: {'sequences': [], 'labels': []} for col in target_columns}
        
        for file_path in tqdm(files, desc="Processing files"):
            try:
                df = pd.read_excel(file_path)
                
                # Get last 165 rows
                if len(df) < LAST_N_ROWS:
                    continue
                
                df_last = df.tail(LAST_N_ROWS)
                
                # Process only target columns
                for col in target_columns:
                    if col not in df_last.columns:
                        continue
                    
                    values = df_last[col].values
                    
                    # Check for valid data
                    if len(values) < LAST_N_ROWS:
                        continue
                    
                    # Get first 150 values as input sequence
                    sequence = values[:SEQUENCE_LENGTH]
                    
                    # Handle NaN values - forward fill then backward fill
                    sequence = pd.Series(sequence).ffill().bfill().values
                    
                    # Skip if still contains NaN
                    if np.any(np.isnan(sequence)):
                        continue
                    
                    # Calculate trend label (last value vs value at position 150)
                    value_at_150 = values[SEQUENCE_LENGTH - 1]
                    last_value = values[-1]
                    
                    # Handle NaN in target values
                    if np.isnan(value_at_150) or np.isnan(last_value):
                        continue
                    
                    # Label: 1 if up, 0 if down
                    label = 1 if last_value > value_at_150 else 0
                    
                    # Normalize sequence (z-score normalization)
                    seq_mean = np.mean(sequence)
                    seq_std = np.std(sequence)
                    if seq_std > 0:
                        sequence = (sequence - seq_mean) / seq_std
                    
                    data_by_target[col]['sequences'].append(sequence)
                    data_by_target[col]['labels'].append(label)
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        # Convert to arrays
        for col in target_columns:
            data_by_target[col]['sequences'] = np.array(data_by_target[col]['sequences'])
            data_by_target[col]['labels'] = np.array(data_by_target[col]['labels'])
        
        return data_by_target
    
    train_data = process_files(train_files)
    test_data = process_files(test_files) if test_files else None
    
    # Print statistics per column
    print(f"\nTraining samples per column:")
    for col in target_columns:
        n_samples = len(train_data[col]['sequences'])
        n_up = np.sum(train_data[col]['labels'])
        print(f"  {col}: {n_samples} samples (Up: {n_up}, Down: {n_samples - n_up})")
    
    if test_data:
        print(f"\nTest samples per column:")
        for col in target_columns:
            n_samples = len(test_data[col]['sequences'])
            n_up = np.sum(test_data[col]['labels'])
            print(f"  {col}: {n_samples} samples (Up: {n_up}, Down: {n_samples - n_up})")
    
    return train_data, test_data


def train_model(model, train_loaders, val_loaders, criterion, optimizer, epochs, writer, target_columns):
    """Train the transformer model with multi-task learning."""
    
    best_val_acc = {col: 0 for col in target_columns}
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_metrics = {col: {'train_loss': 0, 'train_correct': 0, 'train_total': 0} for col in target_columns}
        
        # Train on each target column
        for target_idx, col in enumerate(target_columns):
            train_loader = train_loaders[col]
            
            for sequences, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - {col}"):
                sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(sequences, target_idx=target_idx)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_metrics[col]['train_loss'] += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                epoch_metrics[col]['train_total'] += labels.size(0)
                epoch_metrics[col]['train_correct'] += (predicted == labels).sum().item()
        
        # Validation
        model.eval()
        val_metrics = {col: {'val_correct': 0, 'val_total': 0} for col in target_columns}
        
        with torch.no_grad():
            for target_idx, col in enumerate(target_columns):
                val_loader = val_loaders[col]
                
                for sequences, labels in val_loader:
                    sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
                    outputs = model(sequences, target_idx=target_idx)
                    _, predicted = torch.max(outputs.data, 1)
                    val_metrics[col]['val_total'] += labels.size(0)
                    val_metrics[col]['val_correct'] += (predicted == labels).sum().item()
        
        # Log metrics
        print(f"\nEpoch {epoch+1}/{epochs}:")
        for col in target_columns:
            train_acc = 100 * epoch_metrics[col]['train_correct'] / epoch_metrics[col]['train_total']
            avg_train_loss = epoch_metrics[col]['train_loss'] / len(train_loaders[col])
            val_acc = 100 * val_metrics[col]['val_correct'] / val_metrics[col]['val_total']
            
            print(f"  {col}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
            
            # TensorBoard logging
            writer.add_scalar(f'{col}/train_loss', avg_train_loss, epoch)
            writer.add_scalar(f'{col}/train_acc', train_acc, epoch)
            writer.add_scalar(f'{col}/val_acc', val_acc, epoch)
            
            # Save best model per column
            if val_acc > best_val_acc[col]:
                best_val_acc[col] = val_acc
                torch.save(model.state_dict(), f'best_transformer_{col.replace(" ", "_")}_model.pth')
                print(f"    → Best model for {col} saved with val acc: {val_acc:.2f}%")
        
        # Save overall best model
        avg_val_acc = np.mean([val_metrics[col]['val_correct'] / val_metrics[col]['val_total'] for col in target_columns])
        writer.add_scalar('overall/avg_val_acc', avg_val_acc * 100, epoch)
    
    torch.save(model.state_dict(), 'transformer_trend_final_model.pth')
    return best_val_acc


def bayesian_predict(model, sequences, target_idx, n_samples=N_DROPOUT_SAMPLES):
    """
    Perform Bayesian prediction using Monte Carlo dropout.
    
    Returns:
        predictions: Most common predicted class for each sample
        uncertainties: Entropy-based uncertainty (0 = certain, 1 = uncertain)
        probabilities: Mean probability distribution across samples
    """
    model.eval()
    model.enable_dropout()  # Keep dropout active
    
    batch_size = sequences.size(0)
    all_outputs = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            outputs = model(sequences, target_idx=target_idx)
            probs = torch.softmax(outputs, dim=1)
            all_outputs.append(probs.cpu().numpy())
    
    # Stack all predictions: (n_samples, batch_size, 2)
    all_outputs = np.stack(all_outputs, axis=0)
    
    # Calculate mean probabilities
    mean_probs = np.mean(all_outputs, axis=0)  # (batch_size, 2)
    
    # Predictions based on mean probabilities
    predictions = np.argmax(mean_probs, axis=1)
    
    # Calculate uncertainty using entropy
    epsilon = 1e-10
    entropy = -np.sum(mean_probs * np.log(mean_probs + epsilon), axis=1)
    max_entropy = np.log(2)  # Maximum entropy for binary classification
    uncertainties = entropy / max_entropy  # Normalize to [0, 1]
    
    # Calculate confidence as 1 - uncertainty
    confidences = 1 - uncertainties
    
    return predictions, confidences, mean_probs


def plot_precision_vs_confidence(labels, predictions, confidences, column_name, writer):
    """
    Create histogram showing precision as a function of confidence bins.
    """
    # Define confidence bins
    bins = np.linspace(0, 1, 11)  # 10 bins from 0 to 1
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    precisions = []
    counts = []
    
    for i in range(len(bins) - 1):
        # Get samples in this confidence bin
        mask = (confidences >= bins[i]) & (confidences < bins[i+1])
        
        if np.sum(mask) == 0:
            precisions.append(0)
            counts.append(0)
            continue
        
        bin_labels = labels[mask]
        bin_predictions = predictions[mask]
        
        # Calculate precision for this bin
        correct = np.sum(bin_labels == bin_predictions)
        total = len(bin_labels)
        precision = correct / total if total > 0 else 0
        
        precisions.append(precision)
        counts.append(total)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot precision vs confidence
    ax1.bar(bin_centers, precisions, width=0.08, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Confidence (1 - Uncertainty)', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title(f'{column_name} - Precision vs Confidence', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1.05])
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random baseline')
    
    # Add values on bars
    for i, (x, y, count) in enumerate(zip(bin_centers, precisions, counts)):
        if count > 0:
            ax1.text(x, y + 0.02, f'{y:.2f}\n(n={count})', ha='center', va='bottom', fontsize=8)
    
    ax1.legend()
    
    # Plot sample distribution
    ax2.bar(bin_centers, counts, width=0.08, alpha=0.7, color='coral', edgecolor='black')
    ax2.set_xlabel('Confidence (1 - Uncertainty)', fontsize=12)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title(f'{column_name} - Sample Distribution by Confidence', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Log to TensorBoard
    writer.add_figure(f'{column_name}/precision_vs_confidence', fig, 0)
    
    # Save figure
    fig.savefig(f'precision_vs_confidence_{column_name.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return precisions, counts


def evaluate_model(model, test_loaders, writer, target_columns):
    """Evaluate model on test set for each target column using Bayesian dropout."""
    
    results = {}
    
    for target_idx, col in enumerate(target_columns):
        test_loader = test_loaders[col]
        all_predictions = []
        all_labels = []
        all_confidences = []
        
        # Bayesian inference with dropout
        for sequences, labels in tqdm(test_loader, desc=f"Evaluating {col} with Bayesian dropout"):
            sequences = sequences.to(DEVICE)
            
            # Get Bayesian predictions
            predictions, confidences, _ = bayesian_predict(model, sequences, target_idx)
            
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences)
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_confidences = np.array(all_confidences)
        
        # Calculate metrics
        test_correct = np.sum(all_predictions == all_labels)
        test_total = len(all_labels)
        test_acc = 100 * test_correct / test_total
        
        tp = np.sum((all_predictions == 1) & (all_labels == 1))
        tn = np.sum((all_predictions == 0) & (all_labels == 0))
        fp = np.sum((all_predictions == 1) & (all_labels == 0))
        fn = np.sum((all_predictions == 0) & (all_labels == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate mean confidence
        mean_confidence = np.mean(all_confidences)
        
        print(f"\n{col} - Test Results (Bayesian Dropout):")
        print(f"  Accuracy: {test_acc:.2f}%")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Mean Confidence: {mean_confidence:.4f}")
        print(f"  TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        
        # Log to TensorBoard
        writer.add_scalar(f'{col}/test_acc', test_acc, 0)
        writer.add_scalar(f'{col}/test_precision', precision, 0)
        writer.add_scalar(f'{col}/test_recall', recall, 0)
        writer.add_scalar(f'{col}/test_f1', f1, 0)
        writer.add_scalar(f'{col}/mean_confidence', mean_confidence, 0)
        
        # Plot precision vs confidence histogram
        print(f"  Generating precision vs confidence plot...")
        plot_precision_vs_confidence(all_labels, all_predictions, all_confidences, col, writer)
        
        results[col] = {
            'accuracy': test_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Transformer model for time series trend prediction'
    )
    parser.add_argument(
        '--target_columns',
        type=str,
        nargs='+',
        default=['Stoch RL', 'close'],
        help='List of column names to predict (default: Stoch RL close)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=DATA_DIR,
        help=f'Directory containing Excel files (default: {DATA_DIR})'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=EPOCHS,
        help=f'Number of training epochs (default: {EPOCHS})'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=BATCH_SIZE,
        help=f'Batch size (default: {BATCH_SIZE})'
    )
    parser.add_argument(
        '--n_dropout_samples',
        type=int,
        default=N_DROPOUT_SAMPLES,
        help=f'Number of forward passes for Bayesian dropout (default: {N_DROPOUT_SAMPLES})'
    )
    
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    
    print(f"Device: {DEVICE}")
    print(f"Target columns: {args.target_columns}")
    print(f"Loading data from {args.data_dir}...")
    
    # Update global variables
    global N_DROPOUT_SAMPLES, BATCH_SIZE, EPOCHS
    N_DROPOUT_SAMPLES = args.n_dropout_samples
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    
    # Initialize TensorBoard writer
    writer = SummaryWriter('runs_transformer_trend')
    
    # Load and prepare data
    train_data, test_data = load_and_prepare_data(args.data_dir, args.target_columns, exclude_year=2024)
    
    # Create datasets and dataloaders for each target column
    train_loaders = {}
    val_loaders = {}
    test_loaders = {}
    
    for col in args.target_columns:
        # Get data for this column
        X = train_data[col]['sequences']
        y = train_data[col]['labels']
        
        # Reshape for model input
        X = X.reshape(-1, SEQUENCE_LENGTH, 1)
        
        # Split into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        
        print(f"\n{col} - Final split:")
        print(f"  Train: {len(X_train)}")
        print(f"  Val: {len(X_val)}")
        
        # Create datasets
        train_dataset = TimeSeriesDataset(X_train, y_train, [col] * len(X_train))
        val_dataset = TimeSeriesDataset(X_val, y_val, [col] * len(X_val))
        
        train_loaders[col] = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loaders[col] = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        # Test data
        if test_data and col in test_data:
            X_test = test_data[col]['sequences'].reshape(-1, SEQUENCE_LENGTH, 1)
            y_test = test_data[col]['labels']
            print(f"  Test: {len(X_test)}")
            
            test_dataset = TimeSeriesDataset(X_test, y_test, [col] * len(X_test))
            test_loaders[col] = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Create model
    print("\nInitializing Transformer model...")
    model = TransformerTrendPredictor(num_targets=len(args.target_columns)).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train model
    print("\nStarting training...")
    best_val_acc = train_model(
        model, train_loaders, val_loaders, criterion, optimizer, args.epochs, writer, args.target_columns
    )
    
    # Evaluate on test set
    if test_loaders:
        print(f"\nEvaluating on test set (2024 data) with Bayesian dropout ({args.n_dropout_samples} samples)...")
        model.load_state_dict(torch.load('transformer_trend_final_model.pth'))
        results = evaluate_model(model, test_loaders, writer, args.target_columns)
    
    writer.close()
    
    print("\nTraining complete!")
    print(f"Models saved:")
    for col in args.target_columns:
        print(f"  - best_transformer_{col.replace(' ', '_')}_model.pth")
    print(f"  - transformer_trend_final_model.pth")
    print(f"TensorBoard logs: runs_transformer_trend/")


if __name__ == "__main__":
    main()
