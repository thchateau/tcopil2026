# Multi-Input Transformer Time Series Trend Predictor

A sophisticated machine learning system for predicting financial time series trends using a Transformer-based architecture with multi-input/multi-output (MIMO) capabilities.

## 🎯 Overview

This project implements a **Transformer-based neural network** that simultaneously predicts trend directions (up/down) for **all technical indicators** across multiple financial instruments. The system uses:

- **Input:** All numeric columns from Excel time series data (default: 150 time steps)
- **Output:** Binary trend predictions for all columns (up/down) at 15 time steps horizon
- **Architecture:** Multi-head Transformer with parallel binary classifiers
- **Inference:** Bayesian Monte Carlo Dropout for uncertainty quantification
- **✨ NEW:** Automatic padding for short sequences (v2.0) - files with < 150 points are now supported!

## 📊 Core Files

### Main Application
- **`streamlit_predict_multi.py`** (1421 lines)
  - Interactive Streamlit web interface for training and inference
  - Multi-input Transformer model implementation
  - Data loading, preprocessing, and caching system
  - Training pipeline with TensorBoard integration
  - Bayesian inference with confidence/uncertainty metrics
  - Results export to Excel with comprehensive metrics

### Testing & Validation
- **`test_inference_multi.py`** (118 lines)
  - Unit tests for the MultiTransformerInference class
  - Tests single file and batch folder inference
  - Validates model predictions and accuracy metrics

## 🏗️ Model Architecture

```
INPUT (All Columns × 150 timesteps)
    ↓
Input Projection (Linear: N → 128)
    ↓
Positional Encoding (Sinusoidal)
    ↓
Transformer Encoder (4 layers, 8 heads, 512 ff_dim)
    • Multi-Head Self-Attention (8 heads)
    • Feed Forward Network (128 → 512 → 128)
    • Residual Connections + Layer Norm
    ↓
Global Average Pooling (150 → 1)
    ↓
Parallel Classifiers (One per column)
    • Each: Linear(128→64) → ReLU → Dropout → Linear(64→2)
    ↓
OUTPUT (Binary predictions for ALL columns)
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch streamlit pandas numpy scikit-learn openpyxl plotly tensorboard tqdm
```

### Running the Web Interface
```bash
streamlit run streamlit_predict_multi.py
```

### Running Tests
```bash
python test_inference_multi.py
```

## 📋 Features

### Data Processing
- **Intelligent caching system** with MD5 hashing for fast reloading
- **Automatic Excel file loading** from configurable directories
- **Train/test split** based on year exclusion (e.g., hold out 2020 for testing)
- **Multi-file support** with selective loading (% of dataset)
- **Normalization** per-sequence (z-score) to prevent data leakage
- **✨ NEW (v2.0): Automatic padding** for short sequences
  - Files with < 150 points are automatically padded and used
  - Minimum requirement: 15 points per file
  - Padding tracked in statistics and cache metadata
  - See `PADDING_FEATURE.md` and `CHANGELOG_PADDING.md` for details

### Training
- **Configurable hyperparameters** via Streamlit UI:
  - Sequence length (50-500 timesteps)
  - Prediction horizon (5-100 timesteps)
  - Epochs, batch size, learning rate
  - File percentage for dataset size control
  
- **Multi-output learning:**
  - Single loss aggregation across all columns
  - Per-column accuracy tracking
  - Separate best model checkpoints

- **TensorBoard integration:**
  - Real-time metric visualization
  - Per-column accuracy curves
  - Training/validation/test loss tracking
  - One-click TensorBoard launcher

### Inference
- **Bayesian dropout** for uncertainty estimation:
  - Monte Carlo Dropout with configurable samples (10-100)
  - Entropy-based confidence scores
  - Reliability binning (precision vs confidence)

- **Comprehensive metrics:**
  - Accuracy, Precision, Recall, F1-Score
  - Confusion matrices
  - Per-column and overall metrics

### Export
- **Multi-sheet Excel reports:**
  - Configuration parameters
  - Training loss history
  - Best validation accuracies
  - Detailed test results per column
  - Confusion matrices
  - Sample count by confidence bins

## 🎮 Web Interface Guide

### Configuration Sidebar
1. **Data Directory** - Select or enter path to Excel files
2. **Data Parameters** - Sequence length and prediction horizon
3. **Training Parameters** - Epochs, batch size, learning rate
4. **Bayesian Inference** - Dropout samples for uncertainty
5. **Test Year** - Year to exclude from training (used as test set)
6. **Cache Management** - View, clear, or manage cached datasets
7. **TensorBoard** - Launch monitoring dashboard

### Main Workflow
1. Select data directory and configure parameters
2. Click **"🚀 Start Training"**
3. System loads/caches data automatically
4. Training progress displayed in real-time
5. View training curves and best accuracies
6. Evaluate on test set (if available)
7. Export results to Excel or download model files

## 💾 Data Format

### Input Requirements
- **Location:** Excel files (.xlsx) in configured directory
- **Format:** Each file contains time series data with numeric columns
- **Minimum size:** sequence_length + prediction_horizon rows
- **Columns:** All numeric columns are used as both inputs and targets

### Output Behavior
- Files excluded from training (e.g., containing "2020") form the test set
- Labels are binary: 1 if value increased, 0 if decreased
- Comparison: value at position (sequence_length) vs value at end of window

## 🔧 Key Configuration Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Sequence Length | 50-500 | 150 | Input window size (timesteps) |
| Prediction Horizon | 5-100 | 15 | Forecast window size (timesteps) |
| Epochs | 1-100 | 5 | Training iterations |
| Batch Size | 16-256 | 64 | Samples per batch |
| Learning Rate | 0.00001-0.001 | 0.0001 | Adam optimizer rate |
| Test Year | 2000-2030 | 2020 | Year to exclude for testing |
| Dropout Samples | 10-100 | 50 | MC Dropout iterations for inference |
| File Percentage | 10-100 | 100 | % of files to load |

## 📈 Performance Metrics

The system tracks and exports:
- **Per-column metrics:** Accuracy, Precision, Recall, F1-Score
- **Ensemble metrics:** Overall accuracy across all columns
- **Uncertainty metrics:** Mean confidence per column
- **Calibration:** Precision binned by confidence levels
- **Confusion matrices:** TP, TN, FP, FN per column

## 🧠 Model Specifications

- **Total Parameters:** ~500K (configurable based on data)
- **Device Support:** MPS (Apple Silicon), CUDA (NVIDIA), CPU
- **Input Shape:** (batch, 150, num_features)
- **Output Shape:** (batch, num_features, 2) → binary classification per column
- **Dropout Rate:** 0.1 (used during inference for Bayesian predictions)

## 📁 Cache System

Cached datasets include:
- `train_sequences.npy` - Input training data
- `train_labels.npy` - Training labels
- `test_sequences.npy` - Input test data
- `test_labels.npy` - Test labels
- `metadata.json` - Column names, file counts, shape info
- `config.json` - Full configuration for reproducibility

**Hash:** MD5(config) → unique cache per configuration

## 🔌 Dependencies

| Library | Purpose |
|---------|---------|
| torch | Deep learning framework |
| streamlit | Web interface |
| pandas | Data manipulation |
| numpy | Numerical computing |
| scikit-learn | Train/test split, metrics |
| openpyxl | Excel I/O |
| plotly | Interactive visualizations |
| tensorboard | Training monitoring |
| tqdm | Progress bars |

## 📝 Usage Examples

### Example 1: Basic Training
```python
# Via Streamlit UI
streamlit run streamlit_predict_multi.py
# - Select "datasindAV" folder
# - Set sequence_length=150, prediction_horizon=15
# - Set epochs=5, batch_size=64
# - Click "Start Training"
```

### Example 2: Test Inference
```bash
python test_inference_multi.py
# Runs inference on test data and validates metrics
```

### Example 3: Custom Dataset
```
1. Place Excel files in a new folder (e.g., "my_data/")
2. In Streamlit: enter "my_data" in data directory field
3. Adjust sequence_length and prediction_horizon
4. Train and evaluate
```

## 🎯 Performance Notes

- **Training time:** ~5-10 minutes for 1000 files (150 epochs, 64 batch)
- **Inference:** ~50ms per file (CPU), ~10ms per file (GPU)
- **Memory:** ~4GB for 1000 files with 150 sequence length
- **Cache efficiency:** Reloading cached data takes <1 second

## 🔐 Model Persistence

Models are automatically saved:
- **Best model:** `best_transformer_multi_trend_model.pth`
  - Saved when validation accuracy improves
- **Final model:** `transformer_multi_trend_final_model.pth`
  - Saved after training completes
- **Format:** PyTorch state dict with metadata (column names, shapes)

## 📊 Export Formats

### Excel Report Sheets
1. **Configuration** - All hyperparameters and settings
2. **Training Loss** - Loss per epoch
3. **Best Validation Acc** - Peak validation accuracy per column
4. **Training History** - Per-epoch accuracy for each column (first 50)
5. **Test Results Summary** - Final metrics per column
6. **Overall Test Accuracy** - Global test accuracy
7. **Confusion Matrices** - TP/TN/FP/FN per column

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Data directory not found" | Verify path exists and contains .xlsx files |
| "No valid data found" | Check Excel files have min rows (sequence_length + horizon) |
| Out of memory | Reduce file_percentage or sequence_length |
| Slow training | Enable GPU (CUDA/MPS) or reduce batch_size |
| NaN in predictions | Ensure input data has no all-zero columns |

## 📚 Related Files

- `inference_multi.py` - Inference class for standalone predictions
- `README_inference.md` - Detailed inference documentation
- `run_inference_streamlit.sh` - Shell script to launch Streamlit
- `FIX_COLUMN_ORDER_REPORT.md` - Column ordering notes

## 🔄 Workflow Summary

```
Data Loading
    ↓
[Cache Check] → Cached? → Load from cache
    ↓ No
    Data Preprocessing
    ↓
    [Cache Save]
    ↓
    Train/Val/Test Split
    ↓
    Model Initialization
    ↓
    Training Loop (with TensorBoard logging)
    ↓
    Model Evaluation
    ↓
    Results Export (Excel, Models, TensorBoard logs)
```

## 📞 Support

For issues with:
- **Streamlit interface:** Check browser console and terminal output
- **Model training:** Review TensorBoard logs at configured log_dir
- **Data:** Verify Excel format (numeric columns, sufficient rows)
- **Inference:** Run test_inference_multi.py for diagnostics

## 📄 License

[Add license info if applicable]

---

**Created:** 2026
**Python Version:** 3.8+
**PyTorch Version:** 1.9+
**Streamlit Version:** 1.0+
