#!/usr/bin/env python3
"""
Transformer-based model for predicting trend direction in financial indicators.
Uses ALL columns as input to predict trend for ALL target columns simultaneously.
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


class MultiTimeSeriesDataset(Dataset):
    """Dataset for multi-input time series trend prediction."""
    
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)  # (N, seq_len, num_features)
        self.labels = torch.LongTensor(labels)  # (N, num_targets)
    
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


class MultiInputTransformerTrendPredictor(nn.Module):
    """Transformer model for multi-input multi-output trend prediction."""
    
    def __init__(self, num_features, num_targets, d_model=D_MODEL, nhead=NHEAD, 
                 num_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT):
        super().__init__()
        
        self.num_features = num_features
        self.num_targets = num_targets
        
        # Input embedding: project all features to d_model
        self.input_projection = nn.Linear(num_features, d_model)
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
        
        # Separate classifier for each target column
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 2)  # Binary classification: up or down
            ) for _ in range(num_targets)
        ])
    
    def enable_dropout(self):
        """Enable dropout for Bayesian inference."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def forward(self, x):
        # x shape: (batch, sequence_length, num_features)
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # Global pooling
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x = self.global_pool(x).squeeze(-1)  # (batch, d_model)
        
        # Classification for all targets
        outputs = [classifier(x) for classifier in self.classifiers]  # List of (batch, 2)
        return torch.stack(outputs, dim=1)  # (batch, num_targets, 2)


def load_and_prepare_data(data_dir, exclude_year=None):
    """Load Excel files and prepare sequences for training - using ALL columns."""
    
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
    
    def process_files(files):
        sequences = []
        labels = []
        column_names = None
        skipped = 0
        
        for file_path in tqdm(files, desc="Processing files"):
            try:
                df = pd.read_excel(file_path)
                
                # Get last 165 rows
                if len(df) < LAST_N_ROWS:
                    skipped += 1
                    continue
                
                df_last = df.tail(LAST_N_ROWS)
                
                # Get column names (excluding non-numeric columns)
                if column_names is None:
                    # Select only numeric columns
                    numeric_cols = df_last.select_dtypes(include=[np.number]).columns.tolist()
                    column_names = numeric_cols
                    print(f"Using {len(column_names)} numeric columns as features and targets")
                
                # Get values for all columns
                values = df_last[column_names].values  # (165, num_columns)
                
                # Check if we have enough data
                if len(values) < LAST_N_ROWS:
                    skipped += 1
                    continue
                
                # Get first 150 values as input sequence
                sequence = values[:SEQUENCE_LENGTH, :]  # (150, num_columns)
                
                # Handle NaN values - forward fill then backward fill
                sequence = pd.DataFrame(sequence).ffill().bfill().values
                
                # Skip if still contains NaN
                if np.any(np.isnan(sequence)):
                    skipped += 1
                    continue
                
                # Calculate trend labels for each column
                values_at_150 = values[SEQUENCE_LENGTH - 1, :]  # Values at position 150
                last_values = values[-1, :]  # Last values
                
                # Handle NaN in target values
                if np.any(np.isnan(values_at_150)) or np.any(np.isnan(last_values)):
                    skipped += 1
                    continue
                
                # Label: 1 if up, 0 if down for each column
                label = (last_values > values_at_150).astype(int)  # (num_columns,)
                
                # Normalize sequence (z-score normalization per column)
                seq_mean = np.mean(sequence, axis=0, keepdims=True)
                seq_std = np.std(sequence, axis=0, keepdims=True)
                seq_std[seq_std == 0] = 1  # Avoid division by zero
                sequence = (sequence - seq_mean) / seq_std
                
                sequences.append(sequence)
                labels.append(label)
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                skipped += 1
                continue
        
        print(f"Skipped {skipped} files due to insufficient data or errors")
        
        # Convert to arrays
        sequences = np.array(sequences)  # (N, 150, num_columns)
        labels = np.array(labels)  # (N, num_columns)
        
        return sequences, labels, column_names
    
    train_sequences, train_labels, column_names = process_files(train_files)
    
    if test_files:
        test_sequences, test_labels, _ = process_files(test_files)
    else:
        test_sequences, test_labels = None, None
    
    # Print statistics
    print(f"\nTraining samples: {len(train_sequences)}")
    if column_names:
        print(f"Number of features/targets: {len(column_names)}")
        print(f"Sequence shape: (batch, {SEQUENCE_LENGTH}, {len(column_names)})")
        print(f"Labels shape: (batch, {len(column_names)})")
        
        # Print distribution per column
        print(f"\nTraining label distribution (Up/Down):")
        for i, col in enumerate(column_names):
            n_up = np.sum(train_labels[:, i])
            n_down = len(train_labels) - n_up
            print(f"  {col}: Up={n_up}, Down={n_down}")
    
    if test_sequences is not None:
        print(f"\nTest samples: {len(test_sequences)}")
        print(f"\nTest label distribution (Up/Down):")
        for i, col in enumerate(column_names):
            n_up = np.sum(test_labels[:, i])
            n_down = len(test_labels) - n_up
            print(f"  {col}: Up={n_up}, Down={n_down}")
    
    return train_sequences, train_labels, test_sequences, test_labels, column_names


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, writer, target_columns):
    """Train the transformer model with multi-task learning."""
    
    best_val_acc = {col: 0 for col in target_columns}
    best_overall_val_acc = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = {i: 0 for i in range(len(target_columns))}
        train_total = 0
        
        for sequences, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(sequences)  # (batch, num_targets, 2)
            
            # Calculate loss for all targets
            loss = 0
            for i in range(len(target_columns)):
                loss += criterion(outputs[:, i, :], labels[:, i])
            loss = loss / len(target_columns)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy per target
            for i in range(len(target_columns)):
                _, predicted = torch.max(outputs[:, i, :].data, 1)
                train_correct[i] += (predicted == labels[:, i]).sum().item()
            
            train_total += labels.size(0)
        
        # Validation
        model.eval()
        val_correct = {i: 0 for i in range(len(target_columns))}
        val_total = 0
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
                outputs = model(sequences)
                
                for i in range(len(target_columns)):
                    _, predicted = torch.max(outputs[:, i, :].data, 1)
                    val_correct[i] += (predicted == labels[:, i]).sum().item()
                
                val_total += labels.size(0)
        
        # Log metrics
        avg_train_loss = train_loss / len(train_loader)
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Average Train Loss: {avg_train_loss:.4f}")
        
        val_accs = []
        for i, col in enumerate(target_columns):
            train_acc = 100 * train_correct[i] / train_total
            val_acc = 100 * val_correct[i] / val_total
            val_accs.append(val_acc)
            
            print(f"  {col}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
            
            # TensorBoard logging
            writer.add_scalar(f'{col}/train_acc', train_acc, epoch)
            writer.add_scalar(f'{col}/val_acc', val_acc, epoch)
            
            # Save best model per column
            if val_acc > best_val_acc[col]:
                best_val_acc[col] = val_acc
        
        # Overall metrics
        overall_val_acc = np.mean(val_accs)
        writer.add_scalar('overall/train_loss', avg_train_loss, epoch)
        writer.add_scalar('overall/avg_val_acc', overall_val_acc, epoch)
        print(f"  Overall Val Acc: {overall_val_acc:.2f}%")
        
        # Save best overall model
        if overall_val_acc > best_overall_val_acc:
            best_overall_val_acc = overall_val_acc
            torch.save(model.state_dict(), 'best_transformer_multi_trend_model.pth')
            print(f"  → Best model saved with overall val acc: {overall_val_acc:.2f}%")
    
    torch.save(model.state_dict(), 'transformer_multi_trend_final_model.pth')
    return best_val_acc


def bayesian_predict(model, sequences, n_samples=N_DROPOUT_SAMPLES):
    """
    Perform Bayesian prediction using Monte Carlo dropout for all targets.
    
    Returns:
        predictions: Most common predicted class for each sample and target (batch, num_targets)
        confidences: Entropy-based confidence (batch, num_targets)
        probabilities: Mean probability distribution (batch, num_targets, 2)
    """
    model.eval()
    model.enable_dropout()  # Keep dropout active
    
    batch_size = sequences.size(0)
    all_outputs = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            outputs = model(sequences)  # (batch, num_targets, 2)
            probs = torch.softmax(outputs, dim=2)
            all_outputs.append(probs.cpu().numpy())
    
    # Stack all predictions: (n_samples, batch, num_targets, 2)
    all_outputs = np.stack(all_outputs, axis=0)
    
    # Calculate mean probabilities
    mean_probs = np.mean(all_outputs, axis=0)  # (batch, num_targets, 2)
    
    # Predictions based on mean probabilities
    predictions = np.argmax(mean_probs, axis=2)  # (batch, num_targets)
    
    # Calculate uncertainty using entropy
    epsilon = 1e-10
    entropy = -np.sum(mean_probs * np.log(mean_probs + epsilon), axis=2)  # (batch, num_targets)
    max_entropy = np.log(2)  # Maximum entropy for binary classification
    uncertainties = entropy / max_entropy  # Normalize to [0, 1]
    
    # Calculate confidence as 1 - uncertainty
    confidences = 1 - uncertainties  # (batch, num_targets)
    
    return predictions, confidences, mean_probs


def plot_precision_vs_confidence(labels, predictions, confidences, column_name, writer):
    """Create histogram showing precision as a function of confidence bins."""
    bins = np.linspace(0, 1, 11)  # 10 bins from 0 to 1
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    precisions = []
    counts = []
    
    for i in range(len(bins) - 1):
        mask = (confidences >= bins[i]) & (confidences < bins[i+1])
        
        if np.sum(mask) == 0:
            precisions.append(0)
            counts.append(0)
            continue
        
        bin_labels = labels[mask]
        bin_predictions = predictions[mask]
        
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
    fig.savefig(f'precision_vs_confidence_multi_{column_name.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return precisions, counts


def evaluate_model(model, test_loader, writer, target_columns):
    """Evaluate model on test set for all target columns using Bayesian dropout."""
    
    results = {}
    all_predictions = []
    all_labels = []
    all_confidences = []
    
    # Bayesian inference with dropout
    for sequences, labels in tqdm(test_loader, desc=f"Evaluating with Bayesian dropout"):
        sequences = sequences.to(DEVICE)
        
        # Get Bayesian predictions
        predictions, confidences, _ = bayesian_predict(model, sequences)
        
        all_predictions.append(predictions)
        all_labels.append(labels.cpu().numpy())
        all_confidences.append(confidences)
    
    # Convert to numpy arrays
    all_predictions = np.vstack(all_predictions)  # (N, num_targets)
    all_labels = np.vstack(all_labels)  # (N, num_targets)
    all_confidences = np.vstack(all_confidences)  # (N, num_targets)
    
    # Calculate metrics per column
    for i, col in enumerate(target_columns):
        predictions = all_predictions[:, i]
        labels = all_labels[:, i]
        confidences = all_confidences[:, i]
        
        test_correct = np.sum(predictions == labels)
        test_total = len(labels)
        test_acc = 100 * test_correct / test_total
        
        tp = np.sum((predictions == 1) & (labels == 1))
        tn = np.sum((predictions == 0) & (labels == 0))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        mean_confidence = np.mean(confidences)
        
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
        plot_precision_vs_confidence(labels, predictions, confidences, col, writer)
        
        results[col] = {
            'accuracy': test_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    # Overall metrics
    overall_acc = 100 * np.sum(all_predictions == all_labels) / all_labels.size
    print(f"\nOverall Test Accuracy: {overall_acc:.2f}%")
    writer.add_scalar('overall/test_acc', overall_acc, 0)
    
    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Multi-Input Transformer model for time series trend prediction'
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
    print(f"Loading data from {args.data_dir}...")
    
    # Update global variables
    global N_DROPOUT_SAMPLES, BATCH_SIZE, EPOCHS
    N_DROPOUT_SAMPLES = args.n_dropout_samples
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    
    # Initialize TensorBoard writer
    writer = SummaryWriter('runs_transformer_trend_multi')
    
    # Load and prepare data
    train_sequences, train_labels, test_sequences, test_labels, column_names = \
        load_and_prepare_data(args.data_dir, exclude_year=2024)
    
    num_features = len(column_names)
    num_targets = len(column_names)
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        train_sequences, train_labels, test_size=0.15, random_state=42
    )
    
    print(f"\nFinal split:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val: {len(X_val)}")
    if test_sequences is not None:
        print(f"  Test: {len(test_sequences)}")
    
    # Create datasets
    train_dataset = MultiTimeSeriesDataset(X_train, y_train)
    val_dataset = MultiTimeSeriesDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    test_loader = None
    if test_sequences is not None:
        test_dataset = MultiTimeSeriesDataset(test_sequences, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Create model
    print("\nInitializing Multi-Input Transformer model...")
    model = MultiInputTransformerTrendPredictor(
        num_features=num_features, 
        num_targets=num_targets
    ).to(DEVICE)
    
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
        model, train_loader, val_loader, criterion, optimizer, args.epochs, writer, column_names
    )
    
    # Evaluate on test set
    if test_loader:
        print(f"\nEvaluating on test set (2024 data) with Bayesian dropout ({args.n_dropout_samples} samples)...")
        model.load_state_dict(torch.load('transformer_multi_trend_final_model.pth'))
        results = evaluate_model(model, test_loader, writer, column_names)
    
    writer.close()
    
    print("\nTraining complete!")
    print(f"Models saved:")
    print(f"  - best_transformer_multi_trend_model.pth")
    print(f"  - transformer_multi_trend_final_model.pth")
    print(f"TensorBoard logs: runs_transformer_trend_multi/")


if __name__ == "__main__":
    main()
