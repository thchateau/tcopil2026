#!/usr/bin/env python3
"""
Streamlit interface for Multi-Input Transformer-based trend prediction.
Uses ALL columns as input to predict ALL target trends simultaneously.
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from tqdm import tqdm
import time
from datetime import datetime
import subprocess
import warnings
from io import BytesIO
import json
import hashlib
warnings.filterwarnings('ignore')

# Device configuration
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Model hyperparameters (defaults, can be overridden via UI)
D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 4
DIM_FEEDFORWARD = 512
DROPOUT = 0.1


# Cache directory
CACHE_DIR = "cached_datasets"

def get_cache_config(data_dir, exclude_year, file_percentage, sequence_length, prediction_horizon):
    """Generate cache configuration and hash for unique identification."""
    config = {
        'data_dir': data_dir,
        'exclude_year': exclude_year,
        'file_percentage': file_percentage,
        'sequence_length': sequence_length,
        'prediction_horizon': prediction_horizon
    }
    # Create hash from config
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    return config, config_hash

def get_cache_paths(config_hash):
    """Get paths for cached files."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    base_path = os.path.join(CACHE_DIR, f"cache_{config_hash}")
    return {
        'config': f"{base_path}_config.json",
        'train_sequences': f"{base_path}_train_sequences.npy",
        'train_labels': f"{base_path}_train_labels.npy",
        'test_sequences': f"{base_path}_test_sequences.npy",
        'test_labels': f"{base_path}_test_labels.npy",
        'metadata': f"{base_path}_metadata.json"
    }

def save_cache(cache_paths, train_sequences, train_labels, test_sequences, test_labels, 
               column_names, config, n_train_files, n_test_files, train_skipped, test_skipped, train_padded=0, test_padded=0):
    """Save processed data to cache files."""
    # Save config
    with open(cache_paths['config'], 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save sequences and labels
    np.save(cache_paths['train_sequences'], train_sequences)
    np.save(cache_paths['train_labels'], train_labels)
    
    if test_sequences is not None:
        np.save(cache_paths['test_sequences'], test_sequences)
        np.save(cache_paths['test_labels'], test_labels)
    
    # Save metadata
    metadata = {
        'column_names': column_names,
        'n_train_files': n_train_files,
        'n_test_files': n_test_files,
        'train_skipped': train_skipped,
        'test_skipped': test_skipped,
        'train_padded': train_padded,
        'test_padded': test_padded,
        'train_shape': train_sequences.shape,
        'test_shape': test_sequences.shape if test_sequences is not None else None
    }
    with open(cache_paths['metadata'], 'w') as f:
        json.dump(metadata, f, indent=2)

def load_cache(cache_paths):
    """Load processed data from cache files."""
    # Check if all required files exist
    required_files = ['config', 'train_sequences', 'train_labels', 'metadata']
    if not all(os.path.exists(cache_paths[f]) for f in required_files):
        return None
    
    # Load metadata
    with open(cache_paths['metadata'], 'r') as f:
        metadata = json.load(f)
    
    # Load sequences and labels
    train_sequences = np.load(cache_paths['train_sequences'])
    train_labels = np.load(cache_paths['train_labels'])
    
    # Load test data if exists
    if os.path.exists(cache_paths['test_sequences']) and os.path.exists(cache_paths['test_labels']):
        test_sequences = np.load(cache_paths['test_sequences'])
        test_labels = np.load(cache_paths['test_labels'])
    else:
        test_sequences = None
        test_labels = None
    
    return (train_sequences, train_labels, test_sequences, test_labels,
            metadata['column_names'], metadata['n_train_files'], metadata['n_test_files'],
            metadata['train_skipped'], metadata['test_skipped'],
            metadata.get('train_padded', 0), metadata.get('test_padded', 0))

class MultiTimeSeriesDataset(Dataset):
    """Dataset for multi-input time series trend prediction."""
    
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
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
        
        self.input_projection = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.dropout_rate = dropout
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 2)
            ) for _ in range(num_targets)
        ])
    
    def enable_dropout(self):
        """Enable dropout for Bayesian inference."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.transpose(1, 2)
        x = self.global_pool(x).squeeze(-1)
        
        outputs = [classifier(x) for classifier in self.classifiers]
        return torch.stack(outputs, dim=1)


def load_and_prepare_data(data_dir, exclude_year=None, progress_bar=None, file_percentage=100, sequence_length=150, prediction_horizon=15):
    """Load Excel files and prepare sequences - using ALL columns."""
    
    last_n_rows = sequence_length + prediction_horizon
    all_files = glob.glob(os.path.join(data_dir, "*.xlsx"))
    
    if exclude_year:
        train_files = [f for f in all_files if str(exclude_year) not in f]
        test_files = [f for f in all_files if str(exclude_year) in f]
    else:
        train_files = all_files
        test_files = []
    
    # Apply file percentage BEFORE loading files
    if file_percentage < 100:
        np.random.seed(42)
        if train_files:
            n_train_keep = max(1, int(len(train_files) * file_percentage / 100))
            train_files = list(np.random.choice(train_files, size=n_train_keep, replace=False))
        if test_files:
            n_test_keep = max(1, int(len(test_files) * file_percentage / 100))
            test_files = list(np.random.choice(test_files, size=n_test_keep, replace=False))
    
    def process_files(files):
        sequences = []
        labels = []
        column_names = None
        skipped = 0
        padded = 0
        
        for idx, file_path in enumerate(files):
            if progress_bar:
                progress_bar.progress((idx + 1) / len(files), f"Processing file {idx+1}/{len(files)}")
            
            try:
                df = pd.read_excel(file_path)
                
                # Require at least prediction_horizon points (minimum for creating a label)
                if len(df) < prediction_horizon:
                    skipped += 1
                    continue
                
                # Get numeric columns if not yet defined
                if column_names is None:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    column_names = numeric_cols
                
                values = df[column_names].values
                
                # Check for NaN values
                if np.any(np.isnan(values)):
                    values = pd.DataFrame(values).ffill().bfill().values
                
                if np.any(np.isnan(values)):
                    skipped += 1
                    continue
                
                # Handle cases with fewer than last_n_rows points
                n_rows = len(values)
                
                if n_rows < last_n_rows:
                    # Pad with last row repeated (append at end to preserve real data at beginning)
                    pad_length = last_n_rows - n_rows
                    pad = np.tile(values[-1, :], (pad_length, 1))
                    values = np.vstack([values, pad])
                    padded += 1
                
                # Get the last last_n_rows points
                df_last = values[-last_n_rows:, :]
                
                # Extract sequence and prediction point
                sequence = df_last[:sequence_length, :]
                values_at_seq_end = df_last[sequence_length - 1, :]
                last_values = df_last[-1, :]
                
                # Create label: 1 if increased, 0 if decreased
                label = (last_values > values_at_seq_end).astype(int)
                
                # Normalize sequence
                seq_mean = np.mean(sequence, axis=0, keepdims=True)
                seq_std = np.std(sequence, axis=0, keepdims=True)
                seq_std[seq_std == 0] = 1
                sequence = (sequence - seq_mean) / seq_std
                
                sequences.append(sequence)
                labels.append(label)
                    
            except Exception as e:
                skipped += 1
                continue
        
        sequences = np.array(sequences)
        labels = np.array(labels)
        
        return sequences, labels, column_names, skipped, padded
    
    train_sequences, train_labels, column_names, train_skipped, train_padded = process_files(train_files)
    
    if test_files:
        test_sequences, test_labels, _, test_skipped, test_padded = process_files(test_files)
    else:
        test_sequences, test_labels, test_skipped, test_padded = None, None, 0, 0
    
    return (train_sequences, train_labels, test_sequences, test_labels, 
            column_names, len(train_files), len(test_files), train_skipped, test_skipped, train_padded, test_padded)


def bayesian_predict(model, sequences, n_samples=50):
    """Perform Bayesian prediction for all targets."""
    model.eval()
    model.enable_dropout()
    
    all_outputs = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            outputs = model(sequences)
            probs = torch.softmax(outputs, dim=2)
            all_outputs.append(probs.cpu().numpy())
    
    all_outputs = np.stack(all_outputs, axis=0)
    mean_probs = np.mean(all_outputs, axis=0)
    predictions = np.argmax(mean_probs, axis=2)
    
    epsilon = 1e-10
    entropy = -np.sum(mean_probs * np.log(mean_probs + epsilon), axis=2)
    max_entropy = np.log(2)
    uncertainties = entropy / max_entropy
    confidences = 1 - uncertainties
    
    return predictions, confidences, mean_probs


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, 
                target_columns, progress_placeholder, metrics_placeholder, writer=None, test_loader=None):
    """Train the multi-input transformer model with live updates."""
    
    best_val_acc = {col: 0 for col in target_columns}
    best_overall_val_acc = 0
    history = {col: {'train_acc': [], 'val_acc': [], 'test_acc': []} for col in target_columns}
    history['overall'] = {'train_loss': [], 'test_loss': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = {i: 0 for i in range(len(target_columns))}
        train_total = 0
        
        # Training
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            
            loss = 0
            for i in range(len(target_columns)):
                loss += criterion(outputs[:, i, :], labels[:, i])
            loss = loss / len(target_columns)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
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
        
        # Test set evaluation (if available)
        test_correct = {i: 0 for i in range(len(target_columns))}
        test_total = 0
        test_loss = 0
        
        if test_loader is not None:
            with torch.no_grad():
                for sequences, labels in test_loader:
                    sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
                    outputs = model(sequences)
                    
                    # Calculate test loss
                    loss = 0
                    for i in range(len(target_columns)):
                        loss += criterion(outputs[:, i, :], labels[:, i])
                    loss = loss / len(target_columns)
                    test_loss += loss.item()
                    
                    # Calculate test accuracy
                    for i in range(len(target_columns)):
                        _, predicted = torch.max(outputs[:, i, :].data, 1)
                        test_correct[i] += (predicted == labels[:, i]).sum().item()
                    
                    test_total += labels.size(0)
        
        # Update history
        avg_train_loss = train_loss / len(train_loader)
        history['overall']['train_loss'].append(avg_train_loss)
        
        if test_loader is not None:
            avg_test_loss = test_loss / len(test_loader)
            history['overall']['test_loss'].append(avg_test_loss)
        
        val_accs = []
        test_accs = []
        for i, col in enumerate(target_columns):
            train_acc = 100 * train_correct[i] / train_total
            val_acc = 100 * val_correct[i] / val_total
            val_accs.append(val_acc)
            
            history[col]['train_acc'].append(train_acc)
            history[col]['val_acc'].append(val_acc)
            
            if writer:
                writer.add_scalar(f'{col}/train_acc', train_acc, epoch)
                writer.add_scalar(f'{col}/val_acc', val_acc, epoch)
            
            # Log test accuracy per column
            if test_loader is not None and test_total > 0:
                test_acc = 100 * test_correct[i] / test_total
                test_accs.append(test_acc)
                history[col]['test_acc'].append(test_acc)
                
                if writer:
                    writer.add_scalar(f'{col}/test_acc', test_acc, epoch)
            
            if val_acc > best_val_acc[col]:
                best_val_acc[col] = val_acc
        
        overall_val_acc = np.mean(val_accs)
        if writer:
            writer.add_scalar('overall/train_loss', avg_train_loss, epoch)
            writer.add_scalar('overall/avg_val_acc', overall_val_acc, epoch)
            
            # Log overall test metrics
            if test_loader is not None and test_total > 0:
                writer.add_scalar('overall/test_loss', avg_test_loss, epoch)
                overall_test_acc = np.mean(test_accs)
                writer.add_scalar('overall/avg_test_acc', overall_test_acc, epoch)
        
        if overall_val_acc > best_overall_val_acc:
            best_overall_val_acc = overall_val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'column_names': target_columns,
                'num_features': len(target_columns),
                'num_targets': len(target_columns)
            }, 'best_transformer_multi_trend_model.pth')
        
        # Update progress
        progress_placeholder.progress((epoch + 1) / epochs, f"Epoch {epoch+1}/{epochs}")
        
        # Display metrics
        display_training_metrics(history, target_columns, metrics_placeholder, test_available=(test_loader is not None))
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'column_names': target_columns,
        'num_features': len(target_columns),
        'num_targets': len(target_columns)
    }, 'transformer_multi_trend_final_model.pth')
    return history, best_val_acc


def display_training_metrics(history, target_columns, placeholder, test_available=False):
    """Display training metrics with plotly charts."""
    
    n_cols = len(target_columns)
    
    # Show first 6 columns max to avoid overcrowding
    display_cols = target_columns[:6]
    n_display = len(display_cols)
    
    if n_display > 0:
        fig = make_subplots(
            rows=n_display, cols=2,
            subplot_titles=[f'{col} - Accuracy' for col in display_cols for _ in range(2)][::2] + 
                          ['Overall Loss'] if n_display > 0 else [],
            vertical_spacing=0.1
        )
        
        for idx, col in enumerate(display_cols):
            row = idx + 1
            epochs = list(range(1, len(history[col]['train_acc']) + 1))
            
            # Accuracy plot
            fig.add_trace(
                go.Scatter(x=epochs, y=history[col]['train_acc'], 
                          name=f'{col} Train', mode='lines+markers',
                          line=dict(color='blue')),
                row=row, col=1
            )
            fig.add_trace(
                go.Scatter(x=epochs, y=history[col]['val_acc'], 
                          name=f'{col} Val', mode='lines+markers',
                          line=dict(color='orange')),
                row=row, col=1
            )
            
            # Add test accuracy if available
            if test_available and len(history[col]['test_acc']) > 0:
                fig.add_trace(
                    go.Scatter(x=epochs, y=history[col]['test_acc'], 
                              name=f'{col} Test', mode='lines+markers',
                              line=dict(color='green')),
                    row=row, col=1
                )
            
            # Loss plot (only for first row)
            if idx == 0:
                fig.add_trace(
                    go.Scatter(x=epochs, y=history['overall']['train_loss'], 
                              name='Train Loss', mode='lines+markers',
                              line=dict(color='red')),
                    row=row, col=2
                )
                
                # Add test loss if available
                if test_available and len(history['overall']['test_loss']) > 0:
                    fig.add_trace(
                        go.Scatter(x=epochs, y=history['overall']['test_loss'], 
                                  name='Test Loss', mode='lines+markers',
                                  line=dict(color='darkgreen')),
                        row=row, col=2
                    )
        
        fig.update_layout(height=300*n_display, showlegend=True)
        placeholder.plotly_chart(fig, use_container_width=True)
    
    if n_cols > 6:
        placeholder.info(f"ℹ️ Showing first 6/{n_cols} columns. Check TensorBoard for all metrics.")


def evaluate_and_visualize(model, test_loader, target_columns, n_dropout_samples, writer=None):
    """Evaluate model and create visualizations."""
    
    results = {}
    all_predictions = []
    all_labels = []
    all_confidences = []
    
    progress = st.progress(0, "Evaluating...")
    total_batches = len(test_loader)
    
    for batch_idx, (sequences, labels) in enumerate(test_loader):
        sequences = sequences.to(DEVICE)
        predictions, confidences, _ = bayesian_predict(model, sequences, n_dropout_samples)
        
        all_predictions.append(predictions)
        all_labels.append(labels.cpu().numpy())
        all_confidences.append(confidences)
        
        progress.progress((batch_idx + 1) / total_batches)
    
    progress.empty()
    
    all_predictions = np.vstack(all_predictions)
    all_labels = np.vstack(all_labels)
    all_confidences = np.vstack(all_confidences)
    
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
        
        if writer:
            writer.add_scalar(f'{col}/test_acc', test_acc, 0)
            writer.add_scalar(f'{col}/test_precision', precision, 0)
            writer.add_scalar(f'{col}/test_recall', recall, 0)
            writer.add_scalar(f'{col}/test_f1', f1, 0)
            writer.add_scalar(f'{col}/mean_confidence', mean_confidence, 0)
        
        results[col] = {
            'accuracy': test_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mean_confidence': mean_confidence,
            'predictions': predictions,
            'labels': labels,
            'confidences': confidences,
            'confusion': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}
        }
    
    overall_acc = 100 * np.sum(all_predictions == all_labels) / all_labels.size
    if writer:
        writer.add_scalar('overall/test_acc', overall_acc, 0)
    results['overall_accuracy'] = overall_acc
    
    return results


def plot_precision_vs_confidence(labels, predictions, confidences, column_name, writer=None):
    """Create histogram showing precision as a function of confidence bins."""
    
    bins = np.linspace(0, 1, 11)
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
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'{column_name} - Precision vs Confidence', 
                       f'{column_name} - Sample Distribution'),
        vertical_spacing=0.15
    )
    
    fig.add_trace(
        go.Bar(x=bin_centers, y=precisions, name='Precision',
               marker_color='steelblue',
               text=[f'{p:.2f}<br>(n={c})' for p, c in zip(precisions, counts)],
               textposition='outside'),
        row=1, col=1
    )
    
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", 
                  annotation_text="Random baseline", row=1, col=1)
    
    fig.add_trace(
        go.Bar(x=bin_centers, y=counts, name='Sample Count',
               marker_color='coral',
               text=counts, textposition='outside'),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Confidence (1 - Uncertainty)", row=2, col=1)
    fig.update_yaxes(title_text="Precision", range=[0, 1.1], row=1, col=1)
    fig.update_yaxes(title_text="Number of Samples", row=2, col=1)
    
    fig.update_layout(height=700, showlegend=False)
    
    return fig


def export_results_to_excel(history, best_val_acc, results, overall_acc, 
                            config_params, test_available=True):
    """Export training history and test results to Excel file."""
    
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet 1: Configuration
        config_df = pd.DataFrame([
            {'Parameter': 'Sequence Length', 'Value': config_params['sequence_length']},
            {'Parameter': 'Prediction Horizon', 'Value': config_params['prediction_horizon']},
            {'Parameter': 'Total Data Points', 'Value': config_params['sequence_length'] + config_params['prediction_horizon']},
            {'Parameter': 'Epochs', 'Value': config_params['epochs']},
            {'Parameter': 'Batch Size', 'Value': config_params['batch_size']},
            {'Parameter': 'Learning Rate', 'Value': config_params['learning_rate']},
            {'Parameter': 'Test Year', 'Value': config_params['test_year']},
            {'Parameter': 'Bayesian Samples', 'Value': config_params['n_dropout_samples']},
            {'Parameter': 'File Percentage', 'Value': f"{config_params['file_percentage']}%"},
            {'Parameter': 'Data Directory', 'Value': config_params['data_dir']},
            {'Parameter': 'Log Directory', 'Value': config_params['log_dir']},
            {'Parameter': 'Training Date', 'Value': config_params['timestamp']},
            {'Parameter': 'Device', 'Value': str(config_params['device'])},
            {'Parameter': 'Total Parameters', 'Value': config_params['total_params']},
            {'Parameter': 'Number of Features', 'Value': config_params['num_features']},
        ])
        config_df.to_excel(writer, sheet_name='Configuration', index=False)
        
        # Sheet 2: Training History - Overall Loss
        if 'overall' in history and 'train_loss' in history['overall']:
            loss_df = pd.DataFrame({
                'Epoch': list(range(1, len(history['overall']['train_loss']) + 1)),
                'Train Loss': history['overall']['train_loss']
            })
            loss_df.to_excel(writer, sheet_name='Training Loss', index=False)
        
        # Sheet 3: Best Validation Accuracies
        best_val_df = pd.DataFrame([
            {'Column': col, 'Best Val Accuracy (%)': acc}
            for col, acc in sorted(best_val_acc.items(), key=lambda x: x[1], reverse=True)
        ])
        best_val_df.to_excel(writer, sheet_name='Best Validation Acc', index=False)
        
        # Sheet 4: Training History per Column (first 50 columns to avoid Excel limits)
        train_history_data = []
        columns_to_export = list(history.keys())
        columns_to_export = [c for c in columns_to_export if c != 'overall'][:50]
        
        for epoch_idx in range(len(history[columns_to_export[0]]['train_acc']) if columns_to_export else 0):
            row = {'Epoch': epoch_idx + 1}
            for col in columns_to_export:
                if col in history:
                    row[f'{col}_Train_Acc'] = history[col]['train_acc'][epoch_idx]
                    row[f'{col}_Val_Acc'] = history[col]['val_acc'][epoch_idx]
            train_history_data.append(row)
        
        if train_history_data:
            train_history_df = pd.DataFrame(train_history_data)
            train_history_df.to_excel(writer, sheet_name='Training History', index=False)
        
        # Sheet 5: Test Results Summary (if available)
        if test_available and results:
            test_summary = []
            for col, result in results.items():
                test_summary.append({
                    'Column': col,
                    'Accuracy (%)': result['accuracy'],
                    'Precision': result['precision'],
                    'Recall': result['recall'],
                    'F1 Score': result['f1'],
                    'Mean Confidence': result['mean_confidence'],
                    'True Positives': result['confusion']['tp'],
                    'True Negatives': result['confusion']['tn'],
                    'False Positives': result['confusion']['fp'],
                    'False Negatives': result['confusion']['fn']
                })
            
            test_summary_df = pd.DataFrame(test_summary)
            test_summary_df.to_excel(writer, sheet_name='Test Results Summary', index=False)
            
            # Sheet 6: Overall Test Accuracy
            overall_df = pd.DataFrame([{
                'Metric': 'Overall Test Accuracy',
                'Value (%)': overall_acc
            }])
            overall_df.to_excel(writer, sheet_name='Overall Test Accuracy', index=False)
        
        # Sheet 7: Confusion Matrices (if test available)
        if test_available and results:
            confusion_data = []
            for col, result in results.items():
                conf = result['confusion']
                confusion_data.append({
                    'Column': col,
                    'True Negatives (TN)': conf['tn'],
                    'False Positives (FP)': conf['fp'],
                    'False Negatives (FN)': conf['fn'],
                    'True Positives (TP)': conf['tp'],
                    'Total Predictions': conf['tn'] + conf['fp'] + conf['fn'] + conf['tp']
                })
            
            confusion_df = pd.DataFrame(confusion_data)
            confusion_df.to_excel(writer, sheet_name='Confusion Matrices', index=False)
    
    output.seek(0)
    return output


def main():
    st.set_page_config(page_title="Multi-Input Transformer Trend Predictor", layout="wide")
    
    st.title("🤖 Multi-Input Transformer Time Series Trend Prediction")
    st.markdown("**Uses ALL columns as input to predict ALL trends simultaneously**")
    
    # Architecture explanation section
    with st.expander("📐 Architecture Explanation", expanded=False):
        st.markdown("""
        ### Architecture du réseau MultiInputTransformerTrendPredictor
        
        Ce modèle utilise un **Transformer multi-entrées/multi-sorties** pour prédire simultanément la tendance de tous les indicateurs techniques.
        
        **Composants principaux :**
        
        1. **Input Projection** : Projette les features d'entrée (N indicateurs) vers l'espace de dimension d_model=128
        
        2. **Positional Encoding** : Ajoute l'information de position temporelle aux embeddings via des encodages sinusoïdaux
        
        3. **Transformer Encoder** (4 couches) :
           - **Multi-Head Self-Attention** (8 têtes) : Permet au modèle de capturer les relations temporelles et entre indicateurs
           - **Feed Forward Network** : Réseau dense (128→512→128) avec ReLU et Dropout
           - Connexions résiduelles + Layer Normalization pour stabiliser l'apprentissage
        
        4. **Global Average Pooling** : Agrège la séquence temporelle (150 pas de temps) en un vecteur unique de dimension 128
        
        5. **Classificateurs Parallèles** : Un classificateur indépendant pour chaque indicateur cible
           - Architecture par classificateur : Linear(128→64) → ReLU → Dropout → Linear(64→2)
           - Sortie binaire : [Baisse, Hausse] pour chaque indicateur
        
        **Spécificité** : Le modèle prend en entrée **toutes les colonnes** (indicateurs techniques) sur 150 pas de temps, 
        et prédit simultanément la tendance (hausse/baisse) pour **chacune de ces colonnes** à horizon de 15 pas de temps.
        
        **Inférence Bayésienne** : Le dropout reste actif pendant l'inférence pour estimer l'incertitude des prédictions 
        (Monte Carlo Dropout avec 50 échantillons par défaut).
        """)
        
        # Display architecture diagram if exists
        if os.path.exists('transformer_architecture.svg'):
            with open('transformer_architecture.svg', 'r') as f:
                svg_content = f.read()
            st.markdown(svg_content, unsafe_allow_html=True)
        else:
            st.info("📊 Architecture diagram not found. Run the script to generate it.")
    
    st.markdown("---")
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Data directory with folder picker
    st.sidebar.subheader("📁 Data Directory")
    
    # Initialize session state for data_dir if not exists
    if 'data_dir' not in st.session_state:
        st.session_state.data_dir = "datasindAV"
    
    # List available directories
    available_dirs = [d for d in glob.glob("*/") + glob.glob("*/*/") if os.path.isdir(d.rstrip('/'))]
    available_dirs = sorted(set([d.rstrip('/') for d in available_dirs]))
    
    # Add current directory options
    base_dirs = ["datasindAV", "datasindAV2025", "datasindAV2025short", "datasindAV", "datas", "datas2025"]
    available_dirs = sorted(set(base_dirs + available_dirs))
    
    # Folder selection dropdown
    selected_dir = st.sidebar.selectbox(
        "Select Directory",
        options=available_dirs,
        index=available_dirs.index(st.session_state.data_dir) if st.session_state.data_dir in available_dirs else 0,
        help="Choose from available directories"
    )
    
    # Manual input option
    manual_dir = st.sidebar.text_input(
        "Or enter path manually",
        value=selected_dir,
        help="You can override the selection by typing a custom path"
    )
    
    # Use manual input if different from selection
    data_dir = manual_dir if manual_dir != selected_dir else selected_dir
    st.session_state.data_dir = data_dir
    
    # Data parameters
    st.sidebar.subheader("Data Parameters")
    sequence_length = st.sidebar.number_input(
        "Sequence Length (Input Window)",
        min_value=50, max_value=500, value=150, step=10,
        help="Number of time steps used as input for prediction"
    )
    prediction_horizon = st.sidebar.number_input(
        "Prediction Horizon (Forecast Window)",
        min_value=5, max_value=100, value=15, step=5,
        help="Number of time steps ahead to predict the trend"
    )
    
    # Training parameters
    st.sidebar.subheader("Training Parameters")
    epochs = st.sidebar.slider("Number of Epochs", min_value=1, max_value=100, value=5)
    batch_size = st.sidebar.slider("Batch Size", min_value=16, max_value=256, value=64, step=16)
    learning_rate = st.sidebar.select_slider(
        "Learning Rate",
        options=[0.00001, 0.00005, 0.0001, 0.0005, 0.001],
        value=0.0001,
        format_func=lambda x: f"{x:.5f}"
    )
    
    # Bayesian dropout parameters
    st.sidebar.subheader("Bayesian Inference")
    n_dropout_samples = st.sidebar.slider(
        "Dropout Samples (for uncertainty)",
        min_value=10, max_value=100, value=50, step=10
    )
    
    # Exclude year for test set
    exclude_year = st.sidebar.number_input("Test Year (to exclude from training)", 
                                          min_value=2000, max_value=2030, value=2020)
    
    # Dataset size percentage
    st.sidebar.subheader("Dataset Size")
    file_percentage = st.sidebar.slider(
        "File Loading Percentage (%)",
        min_value=10, max_value=100, value=100, step=5,
        help="Load only a percentage of files (applied before loading to save memory)"
    )
    
    # TensorBoard log directory
    st.sidebar.subheader("📊 Logging")
    current_datetime = datetime.now().strftime("%Y%m%d_%H-%M")
    default_log_dir = f"runs_transformer_trend_multi_{exclude_year}_{current_datetime}"
    log_dir = st.sidebar.text_input(
        "TensorBoard Log Directory",
        value=default_log_dir,
        help="Directory where TensorBoard logs will be saved"
    )
    
    # TensorBoard launch button
    if 'tensorboard_launched' not in st.session_state:
        st.session_state.tensorboard_launched = False
    
    if st.sidebar.button("🚀 Launch TensorBoard", help="Start TensorBoard server in background"):
        try:
            # Launch TensorBoard in detached mode
            port = 6006
            tensorboard_cmd = f"tensorboard --logdir={log_dir} --port={port} --host=0.0.0.0"
            
            # Use subprocess to launch in true background (detached, no I/O blocking)
            process = subprocess.Popen(
                tensorboard_cmd.split(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True
            )
            
            st.session_state.tensorboard_launched = True
            st.session_state.tensorboard_pid = process.pid
            
        except Exception as e:
            st.sidebar.error(f"❌ Failed to start TensorBoard: {str(e)}")
            st.sidebar.caption(f"Try running manually: `tensorboard --logdir={log_dir}`")
    
    if st.session_state.tensorboard_launched:
        st.sidebar.success(f"✅ TensorBoard started!")
        st.sidebar.info(f"🌐 Access at: http://localhost:6006")
        if 'tensorboard_pid' in st.session_state:
            st.sidebar.caption(f"PID: {st.session_state.tensorboard_pid}")
    
    # Display device info
    st.sidebar.markdown("---")
    st.sidebar.info(f"🖥️ **Device:** {DEVICE}")
    
    # TensorBoard info
    if os.path.exists(log_dir):
        st.sidebar.warning(f"⚠️ Log directory exists. New logs will be added.")
    else:
        st.sidebar.info(f"📁 New log directory will be created.")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Launch TensorBoard:**")
    st.sidebar.code(f"tensorboard --logdir={log_dir}", language="bash")
    
    # Cache management section
    st.sidebar.markdown("---")
    st.sidebar.subheader("💾 Cache Management")
    
    # Show cache info
    config, config_hash = get_cache_config(data_dir, exclude_year, file_percentage, sequence_length, prediction_horizon)
    cache_paths = get_cache_paths(config_hash)
    
    if os.path.exists(cache_paths['config']):
        st.sidebar.success(f"✅ Cache exists")
        st.sidebar.caption(f"Hash: {config_hash}")
        if st.sidebar.button("🗑️ Clear this cache"):
            for path in cache_paths.values():
                if os.path.exists(path):
                    os.remove(path)
            st.sidebar.success("Cache cleared!")
            st.rerun()
    else:
        st.sidebar.info(f"📦 No cache for current config")
        st.sidebar.caption(f"Hash: {config_hash}")
    
    # Clear all caches button
    if os.path.exists(CACHE_DIR):
        cache_files = glob.glob(os.path.join(CACHE_DIR, "*"))
        if cache_files:
            st.sidebar.caption(f"Total cache files: {len(cache_files)}")
            if st.sidebar.button("🗑️ Clear all caches"):
                for f in cache_files:
                    os.remove(f)
                st.sidebar.success("All caches cleared!")
                st.rerun()
    
    # Clear temp files button to force Excel reload
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Clear temp files", help="Force reload of all Excel files by clearing Streamlit cache"):
        st.cache_data.clear()
        st.sidebar.success("Streamlit cache cleared! Excel files will be reloaded.")
        st.rerun()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Configuration Summary")
        st.write(f"**Input:** All numeric columns from Excel files")
        st.write(f"**Output:** Trend prediction for all columns")
        st.write(f"**Sequence Length:** {sequence_length} | **Prediction Horizon:** {prediction_horizon}")
        st.write(f"**Epochs:** {epochs} | **Batch Size:** {batch_size} | **Learning Rate:** {learning_rate}")
        st.write(f"**Test Year:** {exclude_year} | **Bayesian Samples:** {n_dropout_samples}")
        st.write(f"**File Percentage:** {file_percentage}% | **Log Directory:** `{log_dir}`")
    
    with col2:
        st.subheader("📁 Data Info")
        if os.path.exists(data_dir):
            n_files = len(glob.glob(os.path.join(data_dir, "*.xlsx")))
            n_test_files = len([f for f in glob.glob(os.path.join(data_dir, "*.xlsx")) 
                               if str(exclude_year) in f])
            st.metric("Total Files", n_files)
            st.metric("Test Files", n_test_files)
        else:
            st.error("❌ Data directory not found!")
            return
    
    st.markdown("---")
    
    # Initialize training state
    if 'training_completed' not in st.session_state:
        st.session_state.training_completed = False
    if 'model_state' not in st.session_state:
        st.session_state.model_state = None
    if 'column_names' not in st.session_state:
        st.session_state.column_names = None
    if 'test_loader_data' not in st.session_state:
        st.session_state.test_loader_data = None
    
    # Training button
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        # Reset training state
        st.session_state.training_completed = False
        st.session_state.model_state = None
        
        # Load data
        st.subheader("📥 Loading Data")
        progress_bar = st.progress(0)
        
        # Check cache first
        config, config_hash = get_cache_config(data_dir, exclude_year, file_percentage, sequence_length, prediction_horizon)
        cache_paths = get_cache_paths(config_hash)
        
        cached_data = load_cache(cache_paths)
        
        if cached_data is not None:
            st.info(f"✅ Loading from cache (hash: {config_hash})...")
            (train_sequences, train_labels, test_sequences, test_labels, 
             column_names, n_train_files, n_test_files, train_skipped, test_skipped, train_padded, test_padded) = cached_data
            st.success(f"✅ Loaded from cache in milliseconds!")
        else:
            with st.spinner("Loading and preprocessing data from Excel files..."):
                (train_sequences, train_labels, test_sequences, test_labels, 
                 column_names, n_train_files, n_test_files, train_skipped, test_skipped, train_padded, test_padded) = \
                    load_and_prepare_data(data_dir, exclude_year, progress_bar, file_percentage, sequence_length, prediction_horizon)
            
            # Save to cache
            st.info(f"💾 Saving to cache (hash: {config_hash})...")
            save_cache(cache_paths, train_sequences, train_labels, test_sequences, test_labels,
                      column_names, config, n_train_files, n_test_files, train_skipped, test_skipped, train_padded, test_padded)
            st.success(f"✅ Cache saved for future use!")
        
        progress_bar.empty()
        
        if len(column_names) == 0:
            st.error("❌ No valid data found!")
            return
        
        # Display data statistics
        st.success(f"✅ Loaded {n_train_files} training files ({train_skipped} skipped, {train_padded} padded) and {n_test_files} test files ({test_skipped} skipped, {test_padded} padded)")
        st.info(f"📊 **Features/Targets:** {len(column_names)} columns")
        st.info(f"📊 **Shape:** Input: ({len(train_sequences)}, {sequence_length}, {len(column_names)}) → Output: ({len(train_sequences)}, {len(column_names)})")
        st.info(f"📊 **Data Window:** Using last {sequence_length + prediction_horizon} points = {sequence_length} input + {prediction_horizon} forecast horizon")
        
        if train_padded > 0 or test_padded > 0:
            st.warning(f"⚠️ **Padding Applied:** {train_padded + test_padded} files were padded because they had fewer than {sequence_length + prediction_horizon} points")
        
        # Show column names
        with st.expander("📋 Column Names"):
            col_df = pd.DataFrame({'Column Name': column_names})
            st.dataframe(col_df, use_container_width=True)
        
        # Show label distribution
        with st.expander("📊 Label Distribution (Up/Down)"):
            label_stats = []
            for i, col in enumerate(column_names):
                n_up = np.sum(train_labels[:, i])
                n_down = len(train_labels) - n_up
                label_stats.append({
                    'Column': col,
                    'Up': n_up,
                    'Down': n_down,
                    'Balance': f"{100*n_up/len(train_labels):.1f}% / {100*n_down/len(train_labels):.1f}%"
                })
            st.dataframe(pd.DataFrame(label_stats), use_container_width=True)
        
        # Prepare dataloaders
        st.subheader("🔧 Preparing Model")
        
        # Use test set as validation set
        if test_sequences is not None and test_labels is not None:
            X_train, y_train = train_sequences, train_labels
            X_val, y_val = test_sequences, test_labels
        else:
            # Fallback to split if no test set
            X_train, X_val, y_train, y_val = train_test_split(
                train_sequences, train_labels, test_size=0.15, random_state=42
            )
        
        st.info(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(test_sequences) if test_sequences is not None else 0}")
        
        train_dataset = MultiTimeSeriesDataset(X_train, y_train)
        val_dataset = MultiTimeSeriesDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        test_loader = None
        if test_sequences is not None:
            test_dataset = MultiTimeSeriesDataset(test_sequences, test_labels)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Create model
        num_features = len(column_names)
        num_targets = len(column_names)
        
        model = MultiInputTransformerTrendPredictor(
            num_features=num_features, 
            num_targets=num_targets
        ).to(DEVICE)
        
        total_params = sum(p.numel() for p in model.parameters())
        st.info(f"🧠 Model initialized with **{total_params:,}** parameters")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Initialize TensorBoard writer
        writer = SummaryWriter(log_dir)
        st.info(f"📊 TensorBoard logging to: `{log_dir}`")
        
        # Training
        st.markdown("---")
        st.subheader("📈 Training Progress")
        
        progress_placeholder = st.empty()
        metrics_placeholder = st.empty()
        
        history, best_val_acc = train_model(
            model, train_loader, val_loader, criterion, optimizer, epochs,
            column_names, progress_placeholder, metrics_placeholder, writer, test_loader
        )
        
        st.success("✅ Training completed!")
        
        # Store results in session state for export
        if 'training_results' not in st.session_state:
            st.session_state.training_results = {}
        
        st.session_state.training_results['history'] = history
        st.session_state.training_results['best_val_acc'] = best_val_acc
        st.session_state.training_results['config'] = {
            'sequence_length': sequence_length,
            'prediction_horizon': prediction_horizon,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'test_year': exclude_year,
            'n_dropout_samples': n_dropout_samples,
            'file_percentage': file_percentage,
            'data_dir': data_dir,
            'log_dir': log_dir,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'device': DEVICE,
            'total_params': total_params,
            'num_features': len(column_names)
        }
        
        # Save important data for persistence
        st.session_state.column_names = column_names
        st.session_state.model_state = model.state_dict()
        if test_loader:
            st.session_state.test_loader_data = (test_sequences, test_labels)
        st.session_state.training_completed = True
        
        # Display best validation accuracies (top 10)
        st.subheader("🏆 Best Validation Accuracies (Top 10)")
        sorted_acc = sorted(best_val_acc.items(), key=lambda x: x[1], reverse=True)[:10]
        best_acc_data = [{'Column': col, 'Best Val Accuracy': f"{acc:.2f}%"} 
                        for col, acc in sorted_acc]
        st.dataframe(pd.DataFrame(best_acc_data), use_container_width=True)
        
        # Evaluation on test set
        if test_loader:
            st.markdown("---")
            st.subheader("🧪 Test Set Evaluation")
            
            with st.spinner(f"Evaluating with Bayesian dropout ({n_dropout_samples} samples)..."):
                checkpoint = torch.load('transformer_multi_trend_final_model.pth')
                model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
                results = evaluate_and_visualize(model, test_loader, column_names, n_dropout_samples, writer)
            
            overall_acc = results.pop('overall_accuracy')
            st.success(f"🎯 **Overall Test Accuracy:** {overall_acc:.2f}%")
            
            # Store test results in session state
            st.session_state.training_results['test_results'] = results
            st.session_state.training_results['overall_test_acc'] = overall_acc
            
            # Close TensorBoard writer
            writer.close()
            
            st.success("🎉 Evaluation completed! Models saved to disk.")
            st.info(f"📊 TensorBoard logs saved to: `{log_dir}`\n\nView with: `tensorboard --logdir={log_dir}`")
    
    # Display results section (persists after training)
    if st.session_state.training_completed and 'training_results' in st.session_state:
        st.markdown("---")
        st.header("📊 Training & Test Results")
        
        # Display best validation accuracies (top 10)
        st.subheader("🏆 Best Validation Accuracies (Top 10)")
        best_val_acc = st.session_state.training_results['best_val_acc']
        sorted_acc = sorted(best_val_acc.items(), key=lambda x: x[1], reverse=True)[:10]
        best_acc_data = [{'Column': col, 'Best Val Accuracy': f"{acc:.2f}%"} 
                        for col, acc in sorted_acc]
        st.dataframe(pd.DataFrame(best_acc_data), use_container_width=True)
        
        # Export button for training results (always available after training)
        st.markdown("---")
        st.subheader("📊 Export Training Results")
        
        col_export_train = st.columns([2, 1])
        
        with col_export_train[0]:
            st.write("Export training history and validation results to Excel file")
            
        with col_export_train[1]:
            if st.button("📥 Export Training to Excel", use_container_width=True):
                with st.spinner("Generating Excel file..."):
                    excel_data = export_results_to_excel(
                        history=st.session_state.training_results['history'],
                        best_val_acc=st.session_state.training_results['best_val_acc'],
                        results={},
                        overall_acc=0,
                        config_params=st.session_state.training_results['config'],
                        test_available=False
                    )
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"training_results_{timestamp}.xlsx"
                    
                    st.download_button(
                        label="💾 Download Training Report",
                        data=excel_data,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_train_only"
                    )
                    st.success(f"✅ Excel file ready for download: {filename}")
        
        # Test results section (if available)
        if 'test_results' in st.session_state.training_results:
            st.markdown("---")
            st.subheader("🧪 Test Set Results")
            
            results = st.session_state.training_results['test_results']
            overall_acc = st.session_state.training_results['overall_test_acc']
            
            st.success(f"🎯 **Overall Test Accuracy:** {overall_acc:.2f}%")
            
            # Display results - show top performers and allow selection
            st.markdown("### 📊 Detailed Results")
            
            # Summary table
            summary_data = []
            for col, result in results.items():
                summary_data.append({
                    'Column': col,
                    'Accuracy': f"{result['accuracy']:.2f}%",
                    'F1 Score': f"{result['f1']:.3f}",
                    'Mean Confidence': f"{result['mean_confidence']:.3f}"
                })
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Select columns to visualize in detail
            selected_cols = st.multiselect(
                "Select columns to visualize in detail (max 3):",
                options=list(results.keys()),
                default=list(results.keys())[:3],
                key="column_selector"
            )
            
            # Display detailed results for selected columns
            for col in selected_cols[:3]:  # Limit to 3 to avoid overcrowding
                st.markdown(f"### 📊 {col}")
                
                result = results[col]
                
                # Metrics in columns
                metric_cols = st.columns(5)
                metric_cols[0].metric("Accuracy", f"{result['accuracy']:.2f}%")
                metric_cols[1].metric("Precision", f"{result['precision']:.3f}")
                metric_cols[2].metric("Recall", f"{result['recall']:.3f}")
                metric_cols[3].metric("F1 Score", f"{result['f1']:.3f}")
                metric_cols[4].metric("Mean Confidence", f"{result['mean_confidence']:.3f}")
                
                # Confusion matrix
                conf = result['confusion']
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.write("**Confusion Matrix:**")
                    conf_df = pd.DataFrame({
                        'Predicted Down': [conf['tn'], conf['fn']],
                        'Predicted Up': [conf['fp'], conf['tp']]
                    }, index=['Actual Down', 'Actual Up'])
                    st.dataframe(conf_df, use_container_width=True)
                
                with col_b:
                    fig_conf = go.Figure(data=go.Heatmap(
                        z=[[conf['tn'], conf['fp']], [conf['fn'], conf['tp']]],
                        x=['Predicted Down', 'Predicted Up'],
                        y=['Actual Down', 'Actual Up'],
                        text=[[conf['tn'], conf['fp']], [conf['fn'], conf['tp']]],
                        texttemplate='%{text}',
                        colorscale='Blues'
                    ))
                    fig_conf.update_layout(title="Confusion Matrix Heatmap", height=300)
                    st.plotly_chart(fig_conf, use_container_width=True)
                
                # Precision vs Confidence plot
                st.write("**Precision vs Confidence:**")
                fig_prec = plot_precision_vs_confidence(
                    result['labels'], result['predictions'], 
                    result['confidences'], col, writer=None
                )
                st.plotly_chart(fig_prec, use_container_width=True)
                
                st.markdown("---")
            
            # Export to Excel button with test results
            st.markdown("---")
            st.subheader("📊 Export Complete Results (Training + Test)")
            
            col_export = st.columns([2, 1])
            
            with col_export[0]:
                st.write("Export training history AND test results to Excel file")
                
            with col_export[1]:
                if st.button("📥 Export All to Excel", type="primary", use_container_width=True):
                    with st.spinner("Generating complete Excel file..."):
                        excel_data = export_results_to_excel(
                            history=st.session_state.training_results['history'],
                            best_val_acc=st.session_state.training_results['best_val_acc'],
                            results=st.session_state.training_results.get('test_results', {}),
                            overall_acc=st.session_state.training_results.get('overall_test_acc', 0),
                            config_params=st.session_state.training_results['config'],
                            test_available=True
                        )
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"complete_results_{timestamp}.xlsx"
                        
                        st.download_button(
                            label="💾 Download Complete Report",
                            data=excel_data,
                            file_name=filename,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="download_complete"
                        )
                        st.success(f"✅ Complete Excel file ready for download: {filename}")
            
            st.markdown("---")
            
            # Download buttons
            st.subheader("💾 Download Models")
            col_dl = st.columns(2)
            
            if os.path.exists('best_transformer_multi_trend_model.pth'):
                with open('best_transformer_multi_trend_model.pth', 'rb') as f:
                    col_dl[0].download_button(
                        label="📥 Best Model",
                        data=f,
                        file_name='best_transformer_multi_trend_model.pth',
                        mime="application/octet-stream"
                    )
            
            if os.path.exists('transformer_multi_trend_final_model.pth'):
                with open('transformer_multi_trend_final_model.pth', 'rb') as f:
                    col_dl[1].download_button(
                        label="📥 Final Model",
                        data=f,
                        file_name='transformer_multi_trend_final_model.pth',
                        mime="application/octet-stream"
                    )


if __name__ == "__main__":
    main()
