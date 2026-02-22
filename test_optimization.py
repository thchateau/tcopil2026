"""
Test script to validate that the optimized indicators produce identical results
to the original implementation.
"""

import pandas as pd
import numpy as np
import time
from indicateurs import Indicator as IndicatorOriginal
from indicateurs_opt import Indicator as IndicatorOptimized


def compare_arrays(arr1, arr2, name, tolerance=1e-6):
    """Compare two numpy arrays with tolerance for floating point errors"""
    # Handle NaN values
    mask = ~(np.isnan(arr1) | np.isnan(arr2))
    
    if not np.allclose(arr1[mask], arr2[mask], rtol=tolerance, atol=tolerance):
        diff = np.abs(arr1[mask] - arr2[mask])
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        print(f"❌ {name}: MISMATCH (max diff: {max_diff:.10f}, mean diff: {mean_diff:.10f})")
        # Show first few differences
        idx = np.where(diff > tolerance)[0][:5]
        for i in idx:
            actual_idx = np.where(mask)[0][i]
            print(f"   Index {actual_idx}: {arr1[actual_idx]:.10f} vs {arr2[actual_idx]:.10f}")
        return False
    else:
        print(f"✅ {name}: MATCH")
        return True


def test_indicators(df):
    """Test all indicators and compare results"""
    print("\n" + "="*60)
    print("TESTING INDICATORS - Original vs Optimized")
    print("="*60)
    
    # Test data
    df_test = df[["open", "high", "low", "close"]].copy()
    
    results = {
        'original_time': {},
        'optimized_time': {},
        'match': {}
    }
    
    # Test MACD
    print("\n--- Testing MACD ---")
    df1 = df_test.copy()
    df2 = df_test.copy()
    
    ind1 = IndicatorOriginal(df1)
    ind2 = IndicatorOptimized(df2)
    
    start = time.time()
    ind1.macd()
    results['original_time']['macd'] = time.time() - start
    
    start = time.time()
    ind2.macd()
    results['optimized_time']['macd'] = time.time() - start
    
    results['match']['macd'] = (
        compare_arrays(ind1.df["macd"].values, ind2.df["macd"].values, "MACD") and
        compare_arrays(ind1.df["macd_signal1"].values, ind2.df["macd_signal1"].values, "MACD Signal1") and
        compare_arrays(ind1.df["macd_signal2"].values, ind2.df["macd_signal2"].values, "MACD Signal2")
    )
    
    # Test Stochastic
    print("\n--- Testing Stochastic ---")
    df1 = df_test.copy()
    df2 = df_test.copy()
    
    ind1 = IndicatorOriginal(df1)
    ind2 = IndicatorOptimized(df2)
    
    start = time.time()
    ind1.stochastic()
    results['original_time']['stochastic'] = time.time() - start
    
    start = time.time()
    ind2.stochastic()
    results['optimized_time']['stochastic'] = time.time() - start
    
    results['match']['stochastic'] = (
        compare_arrays(ind1.df["stochRf"].values, ind2.df["stochRf"].values, "Stoch Rf") and
        compare_arrays(ind1.df["stochRf2"].values, ind2.df["stochRf2"].values, "Stoch Rf2") and
        compare_arrays(ind1.df["stochRL"].values, ind2.df["stochRL"].values, "Stoch RL")
    )
    
    # Test RSI
    print("\n--- Testing RSI ---")
    df1 = df_test.copy()
    df2 = df_test.copy()
    
    ind1 = IndicatorOriginal(df1)
    ind2 = IndicatorOptimized(df2)
    
    start = time.time()
    ind1.rsi()
    results['original_time']['rsi'] = time.time() - start
    
    start = time.time()
    ind2.rsi()
    results['optimized_time']['rsi'] = time.time() - start
    
    results['match']['rsi'] = compare_arrays(ind1.df["rsi"].values, ind2.df["rsi"].values, "RSI")
    
    # Test ADX
    print("\n--- Testing ADX ---")
    df1 = df_test.copy()
    df2 = df_test.copy()
    
    ind1 = IndicatorOriginal(df1)
    ind2 = IndicatorOptimized(df2)
    
    start = time.time()
    ind1.adx()
    results['original_time']['adx'] = time.time() - start
    
    start = time.time()
    ind2.adx()
    results['optimized_time']['adx'] = time.time() - start
    
    results['match']['adx'] = compare_arrays(ind1.df["adx"].values, ind2.df["adx"].values, "ADX")
    
    # Test CCI
    print("\n--- Testing CCI ---")
    df1 = df_test.copy()
    df2 = df_test.copy()
    
    ind1 = IndicatorOriginal(df1)
    ind2 = IndicatorOptimized(df2)
    
    start = time.time()
    ind1.cci()
    results['original_time']['cci'] = time.time() - start
    
    start = time.time()
    ind2.cci()
    results['optimized_time']['cci'] = time.time() - start
    
    results['match']['cci'] = compare_arrays(ind1.df["cci"].values, ind2.df["cci"].values, "CCI")
    
    # Summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"{'Indicator':<15} {'Original (s)':<15} {'Optimized (s)':<15} {'Speedup':<10} {'Match'}")
    print("-"*60)
    
    for indicator in results['original_time'].keys():
        orig_time = results['original_time'][indicator]
        opt_time = results['optimized_time'][indicator]
        speedup = orig_time / opt_time if opt_time > 0 else float('inf')
        match = '✅' if results['match'][indicator] else '❌'
        print(f"{indicator:<15} {orig_time:<15.6f} {opt_time:<15.6f} {speedup:<10.2f}x {match}")
    
    total_orig = sum(results['original_time'].values())
    total_opt = sum(results['optimized_time'].values())
    total_speedup = total_orig / total_opt if total_opt > 0 else float('inf')
    
    print("-"*60)
    print(f"{'TOTAL':<15} {total_orig:<15.6f} {total_opt:<15.6f} {total_speedup:<10.2f}x")
    
    all_match = all(results['match'].values())
    print("\n" + "="*60)
    if all_match:
        print("✅ ALL TESTS PASSED - Results are identical!")
    else:
        print("❌ SOME TESTS FAILED - Results differ!")
    print("="*60 + "\n")
    
    return all_match


def main():
    print("Loading test data...")
    
    # Try to load the test file
    try:
        df = pd.read_excel("Données_Source_Python2.xlsx")
        print(f"Loaded {len(df)} rows of data")
    except FileNotFoundError:
        print("Test file 'Données_Source_Python2.xlsx' not found.")
        print("Creating synthetic test data...")
        # Create synthetic data
        np.random.seed(42)
        n = 500
        df = pd.DataFrame({
            'open': 100 + np.cumsum(np.random.randn(n) * 0.5),
            'high': 100 + np.cumsum(np.random.randn(n) * 0.5) + 1,
            'low': 100 + np.cumsum(np.random.randn(n) * 0.5) - 1,
            'close': 100 + np.cumsum(np.random.randn(n) * 0.5)
        })
        # Ensure high is max and low is min
        df['high'] = df[['open', 'high', 'close']].max(axis=1) + 0.5
        df['low'] = df[['open', 'low', 'close']].min(axis=1) - 0.5
    
    # Run tests
    success = test_indicators(df)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
