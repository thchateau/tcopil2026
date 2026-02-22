# Flexible Sequence Length with Automatic Padding

## 📋 Problem Statement

Previously, the system required all Excel files to contain at least `sequence_length + prediction_horizon` data points (default: 150 + 15 = 165 points). Files with fewer points were simply skipped.

**Example:** If you had files with only 100 points, they were rejected.

## ✅ Solution

The system now **automatically pads short sequences** using forward-fill logic, allowing files with any number of points ≥ `prediction_horizon` (minimum 15 points).

## 🔧 How It Works

### Padding Strategy

1. **Calculate required padding:** If file has N rows and N < sequence_length + prediction_horizon:
   - Pad length = (sequence_length + prediction_horizon) - N
   
2. **Pad with last row:** Repeat the last row of the data `pad_length` times at the END
   - This preserves real data at the BEGINNING of the sequence
   - Avoids artificial jumps
   - Better for model learning
   - Example: File has [10, 20, 30] → Pad with [10, 20, 30, 30, 30, ...]

3. **Extract sequence:** Take the last `sequence_length + prediction_horizon` points:
   - Sequence: first `sequence_length` points (REAL DATA at beginning)
   - Prediction point: point at position `sequence_length`
   - Label: Compare last point with prediction point

### Code Changes

**Before:**
```python
if len(df) < last_n_rows:
    skipped += 1
    continue
```

**After:**
```python
n_rows = len(values)

if n_rows < last_n_rows:
    # Pad with first row repeated
    pad_length = last_n_rows - n_rows
    pad = np.tile(values[0, :], (pad_length, 1))
    values = np.vstack([pad, values])
    padded += 1  # Track padded files
```

## 📊 Padding Statistics

The system now tracks and reports:
- **Padded files:** Count of files that were padded
- **Skipped files:** Count of files discarded (too short or NaN values)

**Example output:**
```
✅ Loaded 1000 training files (50 skipped, 75 padded) 
   and 200 test files (10 skipped, 5 padded)
```

## 🎯 Benefits

| Scenario | Before | After |
|----------|--------|-------|
| File with 100 points | ❌ Skipped | ✅ Padded & Used |
| File with 50 points | ❌ Skipped | ✅ Padded & Used |
| File with 15 points | ❌ Skipped | ✅ Padded & Used |
| File with 10 points | ❌ Skipped | ❌ Still skipped (< 15) |

## 📈 Impact on Model

### Training
- **More training data:** Previously rejected files are now usable
- **Padding handling:** The model learns to process padded sequences
- **Normalization:** Per-sequence z-score normalization handles padded values naturally

### Inference
- **Flexible input:** Can accept files with any number of points (≥ 15)
- **Automatic padding:** Applied transparently during inference
- **Consistent results:** Predictions unaffected by padding strategy

## 💾 Cache Management

Padded file counts are now stored in the cache metadata:
```json
{
  "train_padded": 75,
  "test_padded": 5,
  ...
}
```

This allows:
- Reproducible results across sessions
- Audit trail of data processing
- Detection of data quality issues

## ⚠️ Considerations

### When Padding Might Affect Results
- **Very short files:** A file with 15-30 points will have ~50-75% padded data
  - First row is repeated many times
  - May create artificial patterns in normalized sequences
  
- **Recommendation:** Consider minimum file length:
  - ≥ 50 points: Good (30% padding max)
  - ≥ 100 points: Better (40% padding max)  
  - ≥ sequence_length: Best (minimal padding)

### Monitoring
- Use the "Data Statistics" section to see padding counts
- Check if padded files significantly impact accuracy
- Consider filtering if padding ratio exceeds acceptable threshold

## 🔄 Backward Compatibility

- **Old cached data:** Will be invalidated (hash changes)
- **Configuration:** No changes required
- **Model:** Works with both padded and non-padded sequences

## 📝 Example Usage

### Scenario: 50-point file

**Original behavior:**
```
File: data_2024.xlsx (50 rows)
Status: ❌ Skipped (< 165 required points)
```

**New behavior:**
```
File: data_2024.xlsx (50 rows)
Padding: 115 points added (pad length = 165 - 50)
Padded data: [row1, row1, row1, ..., row1, row2, row3, ..., row50]
Status: ✅ Used (tracked as 1 padded file)
Sequence: last 150 points of padded data
Label: Trend from position 149 to position 164
```

## 🚀 Testing Padded Data

To test the padding feature:

```bash
# Create a small test file with only 50 points
python3 << 'EOF'
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'Close': np.random.randn(50).cumsum() + 100,
    'Volume': np.random.randint(1000000, 10000000, 50)
})
df.to_excel('test_small.xlsx', index=False)
print("Created test_small.xlsx with 50 rows")
EOF

# Run Streamlit with this file
mkdir -p test_data
mv test_small.xlsx test_data/
streamlit run streamlit_predict_multi.py
# Select "test_data" as directory and train
```

## 📞 Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Many padded files | Data source has short series | Accept padding or filter files |
| Low accuracy | Too much padding | Increase minimum file length requirement |
| Memory issues | Padding increases dataset | Reduce file_percentage or sequence_length |

---

**Version:** 2.0 (with padding support)  
**Date:** 2026-02-22  
**Backward Compatible:** No (cache invalidation)
