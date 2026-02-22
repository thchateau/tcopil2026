#!/usr/bin/env python3
"""
Streamlit interface for Transformer-based trend prediction.
Provides interactive UI for training and visualizing results.
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
import warnings
warnings.filterwarnings('ignore')

# Device configuration
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Model hyperparameters
SEQUENCE_LENGTH = 150
LAST_N_ROWS = 165
D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 4
DIM_FEEDFORWARD = 512
DROPOUT = 0.1


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
        
        self.input_projection = nn.Linear(1, d_model)
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
        self.num_targets = num_targets
        
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 2)
            ) for _ in range(self.num_targets)
        ])
    
    def enable_dropout(self):
        """Enable dropout for Bayesian inference."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def forward(self, x, target_idx=None):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.transpose(1, 2)
        x = self.global_pool(x).squeeze(-1)
        
        if target_idx is not None:
            return self.classifiers[target_idx](x)
        else:
            return [classifier(x) for classifier in self.classifiers]


def load_and_prepare_data(data_dir, target_columns, exclude_year=None, progress_bar=None):
    """Load Excel files and prepare sequences for training."""
    
    all_files = glob.glob(os.path.join(data_dir, "*.xlsx"))
    
    if exclude_year:
        train_files = [f for f in all_files if str(exclude_year) not in f]
        test_files = [f for f in all_files if str(exclude_year) in f]
    else:
        train_files = all_files
        test_files = []
    
    def process_files(files):
        data_by_target = {col: {'sequences': [], 'labels': []} for col in target_columns}
        
        for idx, file_path in enumerate(files):
            if progress_bar:
                progress_bar.progress((idx + 1) / len(files), f"Processing file {idx+1}/{len(files)}")
            
            try:
                df = pd.read_excel(file_path)
                
                if len(df) < LAST_N_ROWS:
                    continue
                
                df_last = df.tail(LAST_N_ROWS)
                
                for col in target_columns:
                    if col not in df_last.columns:
                        continue
                    
                    values = df_last[col].values
                    
                    if len(values) < LAST_N_ROWS:
                        continue
                    
                    sequence = values[:SEQUENCE_LENGTH]
                    sequence = pd.Series(sequence).ffill().bfill().values
                    
                    if np.any(np.isnan(sequence)):
                        continue
                    
                    value_at_150 = values[SEQUENCE_LENGTH - 1]
                    last_value = values[-1]
                    
                    if np.isnan(value_at_150) or np.isnan(last_value):
                        continue
                    
                    label = 1 if last_value > value_at_150 else 0
                    
                    seq_mean = np.mean(sequence)
                    seq_std = np.std(sequence)
                    if seq_std > 0:
                        sequence = (sequence - seq_mean) / seq_std
                    
                    data_by_target[col]['sequences'].append(sequence)
                    data_by_target[col]['labels'].append(label)
                    
            except Exception as e:
                continue
        
        for col in target_columns:
            data_by_target[col]['sequences'] = np.array(data_by_target[col]['sequences'])
            data_by_target[col]['labels'] = np.array(data_by_target[col]['labels'])
        
        return data_by_target
    
    train_data = process_files(train_files)
    test_data = process_files(test_files) if test_files else None
    
    return train_data, test_data, len(train_files), len(test_files)


def bayesian_predict(model, sequences, target_idx, n_samples=50):
    """Perform Bayesian prediction using Monte Carlo dropout."""
    model.eval()
    model.enable_dropout()
    
    all_outputs = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            outputs = model(sequences, target_idx=target_idx)
            probs = torch.softmax(outputs, dim=1)
            all_outputs.append(probs.cpu().numpy())
    
    all_outputs = np.stack(all_outputs, axis=0)
    mean_probs = np.mean(all_outputs, axis=0)
    predictions = np.argmax(mean_probs, axis=1)
    
    epsilon = 1e-10
    entropy = -np.sum(mean_probs * np.log(mean_probs + epsilon), axis=1)
    max_entropy = np.log(2)
    uncertainties = entropy / max_entropy
    confidences = 1 - uncertainties
    
    return predictions, confidences, mean_probs


def train_model(model, train_loaders, val_loaders, criterion, optimizer, epochs, 
                target_columns, progress_placeholder, metrics_placeholder, writer=None):
    """Train the transformer model with live updates."""
    
    best_val_acc = {col: 0 for col in target_columns}
    history = {col: {'train_loss': [], 'train_acc': [], 'val_acc': []} for col in target_columns}
    
    for epoch in range(epochs):
        model.train()
        epoch_metrics = {col: {'train_loss': 0, 'train_correct': 0, 'train_total': 0} 
                        for col in target_columns}
        
        # Training
        for target_idx, col in enumerate(target_columns):
            train_loader = train_loaders[col]
            
            for batch_idx, (sequences, labels) in enumerate(train_loader):
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
        
        # Update history and display
        for col in target_columns:
            train_acc = 100 * epoch_metrics[col]['train_correct'] / epoch_metrics[col]['train_total']
            avg_train_loss = epoch_metrics[col]['train_loss'] / len(train_loaders[col])
            val_acc = 100 * val_metrics[col]['val_correct'] / val_metrics[col]['val_total']
            
            history[col]['train_loss'].append(avg_train_loss)
            history[col]['train_acc'].append(train_acc)
            history[col]['val_acc'].append(val_acc)
            
            # Log to TensorBoard if writer is provided
            if writer:
                writer.add_scalar(f'{col}/train_loss', avg_train_loss, epoch)
                writer.add_scalar(f'{col}/train_acc', train_acc, epoch)
                writer.add_scalar(f'{col}/val_acc', val_acc, epoch)
            
            if val_acc > best_val_acc[col]:
                best_val_acc[col] = val_acc
                torch.save(model.state_dict(), f'best_transformer_{col.replace(" ", "_")}_model.pth')
        
        # Update progress
        progress_placeholder.progress((epoch + 1) / epochs, f"Epoch {epoch+1}/{epochs}")
        
        # Display metrics
        display_training_metrics(history, target_columns, metrics_placeholder)
        
        # Log overall average validation accuracy
        if writer:
            avg_val_acc = np.mean([val_metrics[col]['val_correct'] / val_metrics[col]['val_total'] 
                                   for col in target_columns])
            writer.add_scalar('overall/avg_val_acc', avg_val_acc * 100, epoch)
    
    torch.save(model.state_dict(), 'transformer_trend_final_model.pth')
    return history, best_val_acc


def display_training_metrics(history, target_columns, placeholder):
    """Display training metrics with plotly charts."""
    
    # Create subplots for each target column
    n_cols = len(target_columns)
    fig = make_subplots(
        rows=n_cols, cols=2,
        subplot_titles=[f'{col} - Loss' if i % 2 == 0 else f'{col} - Accuracy' 
                       for col in target_columns for i in range(2)],
        vertical_spacing=0.1
    )
    
    for idx, col in enumerate(target_columns):
        row = idx + 1
        epochs = list(range(1, len(history[col]['train_loss']) + 1))
        
        # Loss plot
        fig.add_trace(
            go.Scatter(x=epochs, y=history[col]['train_loss'], 
                      name=f'{col} Train Loss', mode='lines+markers',
                      line=dict(color='blue')),
            row=row, col=1
        )
        
        # Accuracy plot
        fig.add_trace(
            go.Scatter(x=epochs, y=history[col]['train_acc'], 
                      name=f'{col} Train Acc', mode='lines+markers',
                      line=dict(color='blue')),
            row=row, col=2
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=history[col]['val_acc'], 
                      name=f'{col} Val Acc', mode='lines+markers',
                      line=dict(color='orange')),
            row=row, col=2
        )
    
    fig.update_layout(height=300*n_cols, showlegend=True)
    placeholder.plotly_chart(fig, use_container_width=True)


def evaluate_and_visualize(model, test_loaders, target_columns, n_dropout_samples, writer=None):
    """Evaluate model and create visualizations."""
    
    results = {}
    
    for target_idx, col in enumerate(target_columns):
        test_loader = test_loaders[col]
        all_predictions = []
        all_labels = []
        all_confidences = []
        
        progress = st.progress(0, f"Evaluating {col}...")
        total_batches = len(test_loader)
        
        for batch_idx, (sequences, labels) in enumerate(test_loader):
            sequences = sequences.to(DEVICE)
            predictions, confidences, _ = bayesian_predict(model, sequences, target_idx, n_dropout_samples)
            
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences)
            
            progress.progress((batch_idx + 1) / total_batches)
        
        progress.empty()
        
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
        mean_confidence = np.mean(all_confidences)
        
        # Log to TensorBoard if writer is provided
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
            'predictions': all_predictions,
            'labels': all_labels,
            'confidences': all_confidences,
            'confusion': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}
        }
    
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
    
    # Create plotly figure with two subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'{column_name} - Precision vs Confidence', 
                       f'{column_name} - Sample Distribution'),
        vertical_spacing=0.15
    )
    
    # Precision plot
    fig.add_trace(
        go.Bar(x=bin_centers, y=precisions, name='Precision',
               marker_color='steelblue',
               text=[f'{p:.2f}<br>(n={c})' for p, c in zip(precisions, counts)],
               textposition='outside'),
        row=1, col=1
    )
    
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", 
                  annotation_text="Random baseline", row=1, col=1)
    
    # Sample distribution plot
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
    
    # Log to TensorBoard if writer is provided
    if writer:
        writer.add_figure(f'{column_name}/precision_vs_confidence', 
                         plotly_to_matplotlib(fig), 0)
    
    return fig


def plotly_to_matplotlib(plotly_fig):
    """Convert plotly figure to matplotlib for TensorBoard logging."""
    import matplotlib.pyplot as plt
    from io import BytesIO
    from PIL import Image
    
    # Save plotly figure as image
    img_bytes = plotly_fig.to_image(format="png", width=1000, height=700)
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 7))
    img = Image.open(BytesIO(img_bytes))
    ax.imshow(img)
    ax.axis('off')
    
    return fig


def main():
    st.set_page_config(page_title="Transformer Trend Predictor", layout="wide")
    
    st.title("🤖 Transformer-Based Time Series Trend Prediction")
    st.markdown("---")
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Data directory
    data_dir = st.sidebar.text_input("Data Directory", value="datasindAV")
    
    # Get available columns from a sample file
    sample_files = glob.glob(os.path.join(data_dir, "*.xlsx"))
    available_columns = []
    if sample_files:
        try:
            sample_df = pd.read_excel(sample_files[0])
            available_columns = [col for col in sample_df.columns if col.lower() != 'date']
        except:
            available_columns = ['Stoch RL', 'close', 'MACD', 'RSI', 'CCI(20)', 'ADX(14)']
    else:
        available_columns = ['Stoch RL', 'close', 'MACD', 'RSI', 'CCI(20)', 'ADX(14)']
    
    # Target columns selection
    default_columns = ['Stoch RL', 'close'] if 'Stoch RL' in available_columns and 'close' in available_columns else available_columns[:2]
    target_columns = st.sidebar.multiselect(
        "Target Columns to Predict",
        options=available_columns,
        default=default_columns
    )
    
    if not target_columns:
        st.warning("⚠️ Please select at least one target column")
        return
    
    # Training parameters
    st.sidebar.subheader("Training Parameters")
    epochs = st.sidebar.slider("Number of Epochs", min_value=1, max_value=20, value=5)
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
                                          min_value=2000, max_value=2030, value=2024)
    
    # TensorBoard log directory
    st.sidebar.subheader("📊 Logging")
    default_log_dir = f"runs_transformer_trend_{exclude_year}"
    log_dir = st.sidebar.text_input(
        "TensorBoard Log Directory",
        value=default_log_dir,
        help="Directory where TensorBoard logs will be saved"
    )
    
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
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Configuration Summary")
        st.write(f"**Target Columns:** {', '.join(target_columns)}")
        st.write(f"**Epochs:** {epochs} | **Batch Size:** {batch_size} | **Learning Rate:** {learning_rate}")
        st.write(f"**Test Year:** {exclude_year} | **Bayesian Samples:** {n_dropout_samples}")
        st.write(f"**Log Directory:** `{log_dir}`")
    
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
    
    # Training button
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        
        # Load data
        st.subheader("📥 Loading Data")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("Loading and preprocessing data..."):
            train_data, test_data, n_train_files, n_test_files = load_and_prepare_data(
                data_dir, target_columns, exclude_year, progress_bar
            )
        
        progress_bar.empty()
        
        # Display data statistics
        st.success(f"✅ Loaded {n_train_files} training files and {n_test_files} test files")
        
        data_stats = []
        for col in target_columns:
            n_samples = len(train_data[col]['sequences'])
            n_up = np.sum(train_data[col]['labels'])
            data_stats.append({
                'Column': col,
                'Train Samples': n_samples,
                'Up': n_up,
                'Down': n_samples - n_up,
                'Balance': f"{100*n_up/n_samples:.1f}% / {100*(n_samples-n_up)/n_samples:.1f}%"
            })
        
        st.dataframe(pd.DataFrame(data_stats), use_container_width=True)
        
        # Prepare dataloaders
        st.subheader("🔧 Preparing Model")
        
        train_loaders = {}
        val_loaders = {}
        test_loaders = {}
        
        for col in target_columns:
            X = train_data[col]['sequences'].reshape(-1, SEQUENCE_LENGTH, 1)
            y = train_data[col]['labels']
            
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.15, random_state=42, stratify=y
            )
            
            train_dataset = TimeSeriesDataset(X_train, y_train, [col] * len(X_train))
            val_dataset = TimeSeriesDataset(X_val, y_val, [col] * len(X_val))
            
            train_loaders[col] = DataLoader(train_dataset, batch_size=batch_size, 
                                           shuffle=True, num_workers=0)
            val_loaders[col] = DataLoader(val_dataset, batch_size=batch_size, 
                                         shuffle=False, num_workers=0)
            
            if test_data and col in test_data:
                X_test = test_data[col]['sequences'].reshape(-1, SEQUENCE_LENGTH, 1)
                y_test = test_data[col]['labels']
                test_dataset = TimeSeriesDataset(X_test, y_test, [col] * len(X_test))
                test_loaders[col] = DataLoader(test_dataset, batch_size=batch_size, 
                                              shuffle=False, num_workers=0)
        
        # Create model
        model = TransformerTrendPredictor(num_targets=len(target_columns)).to(DEVICE)
        
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
            model, train_loaders, val_loaders, criterion, optimizer, epochs,
            target_columns, progress_placeholder, metrics_placeholder, writer
        )
        
        st.success("✅ Training completed!")
        
        # Display best validation accuracies
        st.subheader("🏆 Best Validation Accuracies")
        best_acc_data = [{'Column': col, 'Best Val Accuracy': f"{acc:.2f}%"} 
                        for col, acc in best_val_acc.items()]
        st.dataframe(pd.DataFrame(best_acc_data), use_container_width=True)
        
        # Evaluation on test set
        if test_loaders:
            st.markdown("---")
            st.subheader("🧪 Test Set Evaluation")
            
            with st.spinner(f"Evaluating with Bayesian dropout ({n_dropout_samples} samples)..."):
                model.load_state_dict(torch.load('transformer_trend_final_model.pth'))
                results = evaluate_and_visualize(model, test_loaders, target_columns, n_dropout_samples, writer)
            
            # Display results for each column
            for col in target_columns:
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
                    # Confusion matrix heatmap
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
                    result['confidences'], col, writer
                )
                st.plotly_chart(fig_prec, use_container_width=True)
                
                # Confidence distribution
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=result['confidences'],
                    nbinsx=30,
                    name='Confidence Distribution',
                    marker_color='lightblue'
                ))
                fig_dist.update_layout(
                    title=f"{col} - Confidence Distribution",
                    xaxis_title="Confidence",
                    yaxis_title="Count",
                    height=400
                )
                st.plotly_chart(fig_dist, use_container_width=True)
                
                st.markdown("---")
            
            # Summary
            st.subheader("📝 Summary")
            summary_data = []
            for col, result in results.items():
                summary_data.append({
                    'Column': col,
                    'Accuracy': f"{result['accuracy']:.2f}%",
                    'F1 Score': f"{result['f1']:.3f}",
                    'Mean Confidence': f"{result['mean_confidence']:.3f}"
                })
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
            
            # Close TensorBoard writer
            writer.close()
            
            st.success("🎉 Evaluation completed! Models saved to disk.")
            st.info(f"📊 TensorBoard logs saved to: `{log_dir}`\n\nView with: `tensorboard --logdir={log_dir}`")
            
            # Download buttons
            st.subheader("💾 Download Models")
            col_dl = st.columns(len(target_columns) + 1)
            for idx, col in enumerate(target_columns):
                model_file = f'best_transformer_{col.replace(" ", "_")}_model.pth'
                if os.path.exists(model_file):
                    with open(model_file, 'rb') as f:
                        col_dl[idx].download_button(
                            label=f"📥 {col}",
                            data=f,
                            file_name=model_file,
                            mime="application/octet-stream"
                        )
            
            if os.path.exists('transformer_trend_final_model.pth'):
                with open('transformer_trend_final_model.pth', 'rb') as f:
                    col_dl[-1].download_button(
                        label="📥 Final Model",
                        data=f,
                        file_name='transformer_trend_final_model.pth',
                        mime="application/octet-stream"
                    )


if __name__ == "__main__":
    main()
