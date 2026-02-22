#!/usr/bin/env python3
"""
Classe d'inférence pour le modèle Multi-Input Transformer.
Permet de faire des prédictions à partir d'un modèle entrainé.
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


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
    
    def __init__(self, num_features, num_targets, d_model=128, nhead=8, 
                 num_layers=4, dim_feedforward=512, dropout=0.1):
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


class MultiTransformerInference:
    """
    Classe d'inférence pour le modèle Multi-Input Transformer.
    
    Args:
        model_path: Chemin vers le fichier du modèle (.pth)
        sequence_length: Longueur de la séquence d'entrée (default: 150)
        prediction_horizon: Horizon de prédiction (default: 15)
        d_model: Dimension du modèle (default: 128)
        nhead: Nombre de têtes d'attention (default: 8)
        num_layers: Nombre de couches du transformer (default: 4)
        dim_feedforward: Dimension du feedforward (default: 512)
        dropout: Taux de dropout (default: 0.1)
        device: Device PyTorch à utiliser (default: auto-detect)
    """
    
    def __init__(
        self,
        model_path: str,
        sequence_length: int = 150,
        prediction_horizon: int = 15,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        device: Optional[torch.device] = None
    ):
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.last_n_rows = sequence_length + prediction_horizon
        
        # Device configuration
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
        
        # Model hyperparameters
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        
        # Will be initialized when loading first data
        self.model = None
        self.num_features = None
        self.num_targets = None
        self.column_names = None
    
    def _initialize_model(self, num_features: int, num_targets: int):
        """Initialize the model architecture."""
        self.num_features = num_features
        self.num_targets = num_targets
        
        self.model = MultiInputTransformerTrendPredictor(
            num_features=num_features,
            num_targets=num_targets,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout
        ).to(self.device)
        
        # Load model weights and metadata
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Handle both old format (state_dict only) and new format (dict with metadata)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            # Load column names from checkpoint if available
            if 'column_names' in checkpoint and self.column_names is None:
                self.column_names = checkpoint['column_names']
                print(f"   Loaded column names from model: {len(self.column_names)} columns")
        else:
            # Old format: checkpoint is the state_dict directly
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        
        print(f"✅ Model loaded successfully from {self.model_path}")
        print(f"   Device: {self.device}")
        print(f"   Features: {num_features}, Targets: {num_targets}")
        if self.column_names:
            print(f"   Columns: {self.column_names[:5]}{'...' if len(self.column_names) > 5 else ''}")
    
    def _prepare_dataframe(self, df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Prepare a DataFrame for inference.
        
        Returns:
            Tuple of (sequence, label) or (None, None) if preparation fails
        """
        if len(df) < self.last_n_rows:
            print(f"⚠️  DataFrame too short: {len(df)} < {self.last_n_rows}")
            return None, None
        
        df_last = df.tail(self.last_n_rows)
        
        # Get numeric columns
        numeric_cols = df_last.select_dtypes(include=[np.number]).columns.tolist()
        
        # Initialize column names on first call if not loaded from model
        if self.column_names is None:
            self.column_names = numeric_cols
            print(f"⚠️  Column names not found in model, using columns from data: {len(self.column_names)} columns")
        
        # Verify that required columns exist in the DataFrame
        missing_cols = set(self.column_names) - set(numeric_cols)
        if missing_cols:
            print(f"❌ Missing columns in DataFrame: {missing_cols}")
            return None, None
        
        # Use the same columns as training, in the SAME ORDER
        # This is critical: pandas will reorder columns to match self.column_names
        values = df_last[self.column_names].values
        
        if len(values) < self.last_n_rows:
            return None, None
        
        # Extract sequence
        sequence = values[:self.sequence_length, :]
        sequence = pd.DataFrame(sequence).ffill().bfill().values
        
        if np.any(np.isnan(sequence)):
            print("⚠️  NaN values found in sequence")
            return None, None
        
        # Calculate label
        values_at_150 = values[self.sequence_length - 1, :]
        last_values = values[-1, :]
        
        if np.any(np.isnan(values_at_150)) or np.any(np.isnan(last_values)):
            print("⚠️  NaN values found in label calculation")
            return None, None
        
        label = (last_values > values_at_150).astype(int)
        
        # Normalize sequence
        seq_mean = np.mean(sequence, axis=0, keepdims=True)
        seq_std = np.std(sequence, axis=0, keepdims=True)
        seq_std[seq_std == 0] = 1
        sequence = (sequence - seq_mean) / seq_std
        
        return sequence, label
    
    def _bayesian_predict(
        self, 
        sequences: torch.Tensor, 
        n_samples: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform Bayesian prediction with uncertainty estimation.
        
        Args:
            sequences: Input sequences tensor
            n_samples: Number of dropout samples for uncertainty estimation
        
        Returns:
            Tuple of (predictions, confidences, mean_probs)
        """
        self.model.eval()
        self.model.enable_dropout()
        
        all_outputs = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                outputs = self.model(sequences)
                probs = torch.softmax(outputs, dim=2)
                all_outputs.append(probs.cpu().numpy())
        
        all_outputs = np.stack(all_outputs, axis=0)
        mean_probs = np.mean(all_outputs, axis=0)
        predictions = np.argmax(mean_probs, axis=2)
        
        # Calculate uncertainty
        epsilon = 1e-10
        entropy = -np.sum(mean_probs * np.log(mean_probs + epsilon), axis=2)
        max_entropy = np.log(2)
        uncertainties = entropy / max_entropy
        confidences = 1 - uncertainties
        
        return predictions, confidences, mean_probs
    
    def inference_dataframe(
        self, 
        df: pd.DataFrame, 
        n_dropout_samples: int = 50
    ) -> Optional[Dict]:
        """
        Perform inference on a single DataFrame.
        
        Args:
            df: Input DataFrame with numeric columns
            n_dropout_samples: Number of dropout samples for uncertainty estimation
        
        Returns:
            Dictionary containing predictions, labels, confidences, and metrics
            or None if inference fails
        """
        # Prepare data
        sequence, label = self._prepare_dataframe(df)
        
        if sequence is None or label is None:
            return None
        
        # Initialize model if needed
        if self.model is None:
            self._initialize_model(sequence.shape[1], label.shape[0])
        
        # Convert to tensor
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        # Perform prediction
        predictions, confidences, mean_probs = self._bayesian_predict(
            sequence_tensor, 
            n_dropout_samples
        )
        
        # Remove batch dimension
        predictions = predictions[0]
        confidences = confidences[0]
        mean_probs = mean_probs[0]
        
        # Calculate metrics per target
        results = {}
        for i, col in enumerate(self.column_names):
            pred = predictions[i]
            true_label = label[i]
            conf = confidences[i]
            prob = mean_probs[i]
            
            correct = (pred == true_label)
            
            results[col] = {
                'prediction': int(pred),
                'true_label': int(true_label),
                'correct': bool(correct),
                'confidence': float(conf),
                'probability_class_0': float(prob[0]),
                'probability_class_1': float(prob[1])
            }
        
        # Overall accuracy
        correct_count = sum(1 for r in results.values() if r['correct'])
        overall_accuracy = correct_count / len(results) if results else 0
        
        return {
            'results': results,
            'overall_accuracy': overall_accuracy,
            'predictions': predictions,
            'labels': label,
            'confidences': confidences
        }
    
    def inference_excel_file(
        self, 
        excel_path: str, 
        n_dropout_samples: int = 50
    ) -> Optional[Dict]:
        """
        Perform inference on a single Excel file.
        
        Args:
            excel_path: Path to the Excel file
            n_dropout_samples: Number of dropout samples for uncertainty estimation
        
        Returns:
            Dictionary containing predictions, labels, confidences, and metrics
            or None if inference fails
        """
        try:
            df = pd.read_excel(excel_path)
            result = self.inference_dataframe(df, n_dropout_samples)
            
            if result:
                result['file_path'] = excel_path
                result['file_name'] = os.path.basename(excel_path)
            
            return result
        
        except Exception as e:
            print(f"❌ Error processing {excel_path}: {str(e)}")
            return None
    
    def inference_folder(
        self, 
        folder_path: str, 
        n_dropout_samples: int = 50,
        progress_callback = None
    ) -> Dict:
        """
        Perform inference on all Excel files in a folder.
        
        Args:
            folder_path: Path to the folder containing Excel files
            n_dropout_samples: Number of dropout samples for uncertainty estimation
            progress_callback: Optional callback function(current, total, filename)
        
        Returns:
            Dictionary containing results for all files and aggregated metrics
        """
        excel_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
        
        if not excel_files:
            print(f"⚠️  No Excel files found in {folder_path}")
            return {'files': [], 'summary': None}
        
        print(f"📁 Found {len(excel_files)} Excel files in {folder_path}")
        
        all_results = []
        all_predictions = []
        all_labels = []
        all_confidences = []
        
        for idx, file_path in enumerate(excel_files):
            if progress_callback:
                progress_callback(idx + 1, len(excel_files), os.path.basename(file_path))
            
            result = self.inference_excel_file(file_path, n_dropout_samples)
            
            if result:
                all_results.append(result)
                all_predictions.append(result['predictions'])
                all_labels.append(result['labels'])
                all_confidences.append(result['confidences'])
        
        if not all_results:
            print("❌ No files successfully processed")
            return {'files': [], 'summary': None}
        
        # Convert to arrays for aggregated metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_confidences = np.array(all_confidences)
        
        # Calculate aggregated metrics per column
        summary_results = {}
        for i, col in enumerate(self.column_names):
            predictions = all_predictions[:, i]
            labels = all_labels[:, i]
            confidences = all_confidences[:, i]
            
            correct = np.sum(predictions == labels)
            total = len(labels)
            accuracy = 100 * correct / total if total > 0 else 0
            
            tp = np.sum((predictions == 1) & (labels == 1))
            tn = np.sum((predictions == 0) & (labels == 0))
            fp = np.sum((predictions == 1) & (labels == 0))
            fn = np.sum((predictions == 0) & (labels == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            mean_confidence = np.mean(confidences)
            
            summary_results[col] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'mean_confidence': mean_confidence,
                'confusion': {
                    'tp': int(tp),
                    'tn': int(tn),
                    'fp': int(fp),
                    'fn': int(fn)
                }
            }
        
        # Overall accuracy
        overall_accuracy = 100 * np.sum(all_predictions == all_labels) / all_labels.size
        
        print(f"✅ Processed {len(all_results)}/{len(excel_files)} files successfully")
        print(f"📊 Overall Accuracy: {overall_accuracy:.2f}%")
        
        return {
            'files': all_results,
            'summary': {
                'results': summary_results,
                'overall_accuracy': overall_accuracy,
                'total_files': len(excel_files),
                'processed_files': len(all_results),
                'failed_files': len(excel_files) - len(all_results)
            }
        }
