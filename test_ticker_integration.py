#!/usr/bin/env python3
"""
Script de test pour valider l'intégration du ticker dans streamlit_inference_multi.py
"""

import sys
import pandas as pd
from fmpapi import get_fmp_data
from indicateurs_opt import Indicator


def test_ticker_integration(ticker='TSLA', interval='30min'):
    """Test the ticker data fetching and indicator calculation"""
    
    print(f"\n{'='*60}")
    print(f"TEST: Récupération et traitement des données pour {ticker}")
    print(f"{'='*60}\n")
    
    # Step 1: Fetch data from FMP
    print(f"📡 Étape 1: Récupération des données via FMP API...")
    try:
        df = get_fmp_data(ticker, interval=interval)
        if df is None or df.empty:
            print(f"❌ Erreur: Aucune donnée reçue pour {ticker}")
            return False
        
        print(f"✅ Données récupérées: {len(df)} lignes")
        print(f"   Colonnes: {list(df.columns)}")
        print(f"   Première date: {df['date'].iloc[0]}")
        print(f"   Dernière date: {df['date'].iloc[-1]}")
        
    except Exception as e:
        print(f"❌ Erreur lors de la récupération: {str(e)}")
        return False
    
    # Step 2: Validate required columns
    print(f"\n📋 Étape 2: Validation des colonnes requises...")
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ Colonnes manquantes: {missing_cols}")
        return False
    
    print(f"✅ Toutes les colonnes requises sont présentes")
    
    # Step 3: Calculate indicators
    print(f"\n🔢 Étape 3: Calcul des indicateurs techniques...")
    try:
        indicator = Indicator(df[required_cols])
        
        # Calculate all indicators
        print("   - Calcul MACD...")
        indicator.macd()
        
        print("   - Calcul Stochastic...")
        indicator.stochastic()
        
        print("   - Calcul RSI...")
        indicator.rsi()
        
        print("   - Calcul ADX...")
        indicator.adx()
        
        print("   - Calcul CCI...")
        indicator.cci()
        
        print(f"✅ Indicateurs calculés avec succès")
        
        # Get result dataframe
        df_with_indicators = indicator.df.copy()
        
        # Add date column
        if 'date' in df.columns:
            df_with_indicators.insert(0, 'date', df['date'].values)
        
        # Add volume if exists
        if 'volume' in df.columns:
            df_with_indicators['volume'] = df['volume'].values
        
        print(f"   Colonnes finales: {list(df_with_indicators.columns)}")
        print(f"   Nombre total de colonnes: {len(df_with_indicators.columns)}")
        
    except Exception as e:
        print(f"❌ Erreur lors du calcul des indicateurs: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Check for NaN values
    print(f"\n🔍 Étape 4: Vérification des valeurs NaN...")
    nan_counts = df_with_indicators.isna().sum()
    total_nans = nan_counts.sum()
    
    if total_nans > 0:
        print(f"⚠️  {total_nans} valeurs NaN trouvées:")
        for col, count in nan_counts[nan_counts > 0].items():
            print(f"   - {col}: {count} NaN")
    else:
        print(f"✅ Aucune valeur NaN détectée")
    
    # Step 5: Display sample data
    print(f"\n📊 Étape 5: Aperçu des dernières données:")
    print(df_with_indicators.tail(5).to_string())
    
    # Step 6: Save to Excel for manual verification
    output_file = f"test_ticker_{ticker}_{interval}.xlsx"
    print(f"\n💾 Étape 6: Sauvegarde dans {output_file}...")
    try:
        df_with_indicators.to_excel(output_file, index=False)
        print(f"✅ Fichier sauvegardé avec succès")
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde: {str(e)}")
        return False
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✅ TEST RÉUSSI!")
    print(f"{'='*60}")
    print(f"Résumé:")
    print(f"  - Ticker: {ticker}")
    print(f"  - Intervalle: {interval}")
    print(f"  - Lignes: {len(df_with_indicators)}")
    print(f"  - Colonnes: {len(df_with_indicators.columns)}")
    print(f"  - Fichier: {output_file}")
    print(f"{'='*60}\n")
    
    return True


def main():
    """Run tests"""
    print("\n🧪 TESTS D'INTÉGRATION - TICKER + INDICATEURS\n")
    
    # Test 1: TSLA with 30min interval
    success1 = test_ticker_integration('TSLA', '30min')
    
    # Test 2: AAPL with 5min interval (optional)
    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        print("\n" + "="*60)
        print("TEST SUPPLÉMENTAIRE")
        print("="*60 + "\n")
        success2 = test_ticker_integration('AAPL', '5min')
        
        if success1 and success2:
            print("\n🎉 TOUS LES TESTS RÉUSSIS!")
            return 0
        else:
            print("\n❌ CERTAINS TESTS ONT ÉCHOUÉ")
            return 1
    
    return 0 if success1 else 1


if __name__ == "__main__":
    exit(main())
