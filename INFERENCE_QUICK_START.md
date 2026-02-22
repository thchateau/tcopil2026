# 🔮 Quick Start: Using the Inference Interface

How to use the trained model to make predictions on new data.

---

## 📥 Step 1: Have a Trained Model

You need a trained model file (`.pth`):
- `best_transformer_multi_trend_model.pth` (best validation accuracy)
- `transformer_multi_trend_final_model.pth` (after training completion)

These are created by running `streamlit_predict_multi.py` and training.

---

## 🚀 Step 2: Launch Inference Interface

```bash
streamlit run streamlit_inference_multi.py
```

Browser opens automatically at: `http://localhost:8501`

---

## 🎯 Step 3: Make Predictions

### Option A: Predict on Single Excel File (5 minutes)

1. **Select Model**
   - Browse or enter path to trained model (`.pth` file)
   - Example: `best_transformer_multi_trend_model.pth`

2. **Configure Parameters**
   - Sequence Length: 150 (input window)
   - Prediction Horizon: 15 (forecast window)
   - Dropout Samples: 50 (uncertainty iterations)

3. **Select Data**
   - Browse or enter path to Excel file
   - File requirements:
     - Numeric columns only
     - Minimum 15 rows (padded if needed)
     - Time series order (oldest → newest)

4. **Run Prediction**
   - Click "🔮 Predict on File"
   - View results:
     - Predictions (Up/Down)
     - Confidence scores
     - Accuracy (if labels available)

5. **Export Results**
   - Click "📥 Export to Excel"
   - Download detailed report

### Option B: Batch Process Folder (10 minutes)

1. **Select Model** (same as above)

2. **Select Folder**
   - Contains multiple Excel files
   - Each file processed independently
   - Results aggregated

3. **Process**
   - Click "🔮 Predict on Folder"
   - Progress bar shows status
   - Files processed in sequence

4. **Review Results**
   - File-by-file accuracy
   - Summary statistics
   - Per-column metrics

5. **Export**
   - Click "📥 Export to Excel"
   - 6-sheet comprehensive report

### Option C: Live Market Data (2 minutes)

1. **Select Model** (same as above)

2. **Enter Ticker**
   - Stock symbol (e.g., "TSLA", "AAPL", "MSFT")

3. **Select Interval**
   - 1min, 5min, 30min, 1hour, 4hour, 1day

4. **Fetch Data**
   - Click "📊 Fetch from FMP API"
   - System fetches OHLCV data
   - Calculates technical indicators

5. **Predict**
   - Model predicts trend for next 15 periods
   - Compare with latest price movement
   - See confidence scores

---

## 📊 Understanding Results

### Single File Output Example

```
File: stock_data.xlsx
Overall Accuracy: 75.23%

Column: Close
├─ Prediction: ↗️ UP (probability: 85%)
├─ Confidence: 0.87
├─ True Label: UP
└─ Result: ✅ CORRECT

Column: Volume
├─ Prediction: ↘️ DOWN (probability: 62%)
├─ Confidence: 0.62
├─ True Label: DOWN
└─ Result: ✅ CORRECT
```

### Key Metrics

| Metric | Meaning | Good Value |
|--------|---------|-----------|
| **Confidence** | Model certainty (0-1) | > 0.7 |
| **Accuracy** | % correct predictions | > 70% |
| **Precision** | Of predicted positives, % correct | > 0.7 |
| **Recall** | Of actual positives, % found | > 0.7 |
| **F1 Score** | Balance of precision/recall | > 0.7 |

---

## 📈 Visualizations

The interface displays:

1. **Confusion Matrix** - Prediction accuracy breakdown
2. **Metrics Charts** - Accuracy, precision, recall, F1 per column
3. **Confidence Distribution** - Model certainty histogram
4. **Per-Column Results** - Individual indicator performance

---

## 💾 Excel Export Sheets

Downloaded Excel file contains:

| Sheet | Content |
|-------|---------|
| **Files Summary** | One row per file, overall accuracy |
| **Detailed Predictions** | All predictions with true labels |
| **Summary Metrics** | Accuracy, precision, recall, F1 per column |
| **Overall Statistics** | Total files, processed, accuracy |
| **Global Precision** | Per-column performance metrics |
| **Global Average** | Average metrics across all indicators |

---

## ⚡ Quick Examples

### Example 1: Single Stock File

```bash
# 1. Launch
streamlit run streamlit_inference_multi.py

# 2. In UI:
#   - Model: best_model.pth
#   - File: TSLA_data.xlsx
#   - Click "Predict on File"

# 3. Download results
#   - Click "Export to Excel"
#   - Get TSLA_predictions.xlsx
```

### Example 2: Batch Testing

```bash
# 1. Launch inference

# 2. In UI:
#   - Model: best_model.pth
#   - Folder: ~/stocks/test_data/
#   - Click "Predict on Folder"

# 3. Wait for processing (progress shows %)

# 4. Export:
#   - Multi-file Excel report
#   - Summary statistics
```

### Example 3: Live Trading Analysis

```bash
# 1. Launch inference

# 2. In UI:
#   - Model: best_model.pth
#   - Ticker: MSFT
#   - Interval: 1day
#   - Click "Fetch from FMP API"

# 3. Results:
#   - Next 15-day trend predictions
#   - Confidence for each indicator
#   - Compare with actual price movement
```

---

## 🔧 Troubleshooting

### "Model file not found"
- Check file path is correct
- Use absolute path (e.g., `/Users/name/models/best_model.pth`)
- Ensure file ends with `.pth`

### "ValueError: Expected 150 features"
- Model trained with specific number of inputs
- Your file has different number of columns
- Match column count to training data

### "No data from API"
- Check ticker symbol is valid
- Verify internet connection
- Check FMP API key (if not embedded)

### "NaN in results"
- File contains non-numeric columns
- Data has all-zero columns
- Check for missing values

### "Memory error"
- Processing too many files
- Reduce dropout samples (e.g., 30 instead of 50)
- Process files in smaller batches

---

## 🎯 Best Practices

1. **Model Selection**
   - Use `best_transformer_multi_trend_model.pth` for highest accuracy
   - Match sequence length to training (usually 150)

2. **Data Preparation**
   - Ensure time series order (oldest first)
   - Only numeric columns
   - Minimum 15 rows

3. **Confidence Interpretation**
   - > 0.8: High confidence, reliable predictions
   - 0.6-0.8: Moderate confidence, consider with other signals
   - < 0.6: Low confidence, treat with caution

4. **Batch Processing**
   - Process 100+ files efficiently
   - Dropout samples 30-50 balances speed/accuracy
   - Monitor memory usage

5. **Results Analysis**
   - Check accuracy per indicator (per-column results)
   - Compare with baseline (50% random)
   - Validate on real trading data

---

## 📋 File Requirements

### Excel File Format
```
Minimum: 15 rows
Columns: All numeric (no text, dates, etc.)
Order: Time series (oldest to newest)

Example:
Date       | Close | Volume  | RSI | MACD
2024-01-01 | 100   | 1000000 | 50  | 0.5
2024-01-02 | 102   | 950000  | 52  | 0.6
...
2024-02-22 | 115   | 1200000 | 62  | 1.2
```

### Model File Format
- PyTorch checkpoint (`.pth`)
- Contains:
  - `model_state_dict`: Weights and parameters
  - `column_names`: Technical indicators
  - `num_features`: Number of inputs
  - `num_targets`: Number of outputs

---

## 🔐 API Setup (Optional)

For live market data fetching:

1. Get API key: https://financialmodelingprep.com/
2. Add to code (line 231):
   ```python
   api_key = 'YOUR_API_KEY_HERE'
   ```
3. Or use environment variable:
   ```bash
   export FMP_API_KEY='YOUR_API_KEY_HERE'
   ```

---

## 📈 Performance Tips

| Setting | Speed | Accuracy |
|---------|-------|----------|
| Dropout samples: 10 | ⚡ Fast | ⭐ Low |
| Dropout samples: 30 | ⚡⚡ Good | ⭐⭐ Fair |
| Dropout samples: 50 | ⚡⚡⚡ Balanced | ⭐⭐⭐ Good |
| Dropout samples: 100 | ⚡⚡⚡⚡ Slow | ⭐⭐⭐⭐ Best |

**Recommendation:** Use 50 for good balance

---

## 🚀 Next Steps

1. **First Run**
   - Launch interface
   - Select trained model
   - Predict on test file
   - Review results

2. **Evaluate Performance**
   - Check accuracy per indicator
   - Compare confidence with correctness
   - Identify strengths/weaknesses

3. **Batch Processing**
   - Process multiple files
   - Generate summary report
   - Export to Excel

4. **Live Trading (if applicable)**
   - Fetch live market data
   - Get trend predictions
   - Make informed decisions

---

## 📚 See Also

- **Training:** `streamlit_predict_multi.py`
- **Core inference:** `inference_multi.py`
- **Full documentation:** `README_INFERENCE.md`
- **Project overview:** `README.md`

---

**Version:** 2.0  
**Date:** 2026-02-22  
**Status:** ✅ Production Ready
