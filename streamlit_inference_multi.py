#!/usr/bin/env python3
"""
Interface Streamlit pour l'inférence avec le modèle Multi-Input Transformer.
Permet de charger un modèle et faire des prédictions sur des fichiers Excel.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from inference_multi import MultiTransformerInference
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from io import BytesIO
from fmpapi import get_fmp_data
from indicateurs_opt import Indicator


def export_results_to_excel(results):
    """Export inference results to Excel file."""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        if 'files' in results and results['files']:
            # Sheet 1: File-by-file results
            file_data = []
            for file_result in results['files']:
                file_data.append({
                    'File': file_result['file_name'],
                    'Overall Accuracy': f"{file_result['overall_accuracy']*100:.2f}%",
                    'Path': file_result['file_path']
                })
            
            files_df = pd.DataFrame(file_data)
            files_df.to_excel(writer, sheet_name='Files Summary', index=False)
            
            # Sheet 2: Detailed predictions per file and column
            detailed_data = []
            for file_result in results['files']:
                for col, col_result in file_result['results'].items():
                    detailed_data.append({
                        'File': file_result['file_name'],
                        'Column': col,
                        'Prediction': col_result['prediction'],
                        'True Label': col_result['true_label'],
                        'Correct': col_result['correct'],
                        'Confidence': f"{col_result['confidence']:.4f}",
                        'Prob Class 0': f"{col_result['probability_class_0']:.4f}",
                        'Prob Class 1': f"{col_result['probability_class_1']:.4f}"
                    })
            
            detailed_df = pd.DataFrame(detailed_data)
            detailed_df.to_excel(writer, sheet_name='Detailed Predictions', index=False)
        
        # Sheet 3: Summary metrics (if folder inference)
        if 'summary' in results and results['summary']:
            summary_data = []
            for col, metrics in results['summary']['results'].items():
                summary_data.append({
                    'Column': col,
                    'Accuracy (%)': f"{metrics['accuracy']:.2f}",
                    'Precision': f"{metrics['precision']:.4f}",
                    'Recall': f"{metrics['recall']:.4f}",
                    'F1 Score': f"{metrics['f1']:.4f}",
                    'Mean Confidence': f"{metrics['mean_confidence']:.4f}",
                    'True Positives': metrics['confusion']['tp'],
                    'True Negatives': metrics['confusion']['tn'],
                    'False Positives': metrics['confusion']['fp'],
                    'False Negatives': metrics['confusion']['fn']
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary Metrics', index=False)
            
            # Sheet 4: Overall statistics
            overall_data = [{
                'Metric': 'Overall Accuracy',
                'Value': f"{results['summary']['overall_accuracy']:.2f}%"
            }, {
                'Metric': 'Total Files',
                'Value': results['summary']['total_files']
            }, {
                'Metric': 'Processed Files',
                'Value': results['summary']['processed_files']
            }, {
                'Metric': 'Failed Files',
                'Value': results['summary']['failed_files']
            }]
            
            overall_df = pd.DataFrame(overall_data)
            overall_df.to_excel(writer, sheet_name='Overall Statistics', index=False)
            
            # Sheet 5: Global precision per column
            if results['summary']['results']:
                global_precision_data = []
                for col, metrics in results['summary']['results'].items():
                    global_precision_data.append({
                        'Column': col,
                        'Global Precision (%)': f"{metrics['accuracy']:.2f}",
                        'Precision Score': f"{metrics['precision']:.4f}",
                        'Recall Score': f"{metrics['recall']:.4f}",
                        'F1 Score': f"{metrics['f1']:.4f}"
                    })
                
                global_precision_df = pd.DataFrame(global_precision_data)
                global_precision_df.to_excel(writer, sheet_name='Global Precision by Column', index=False)
                
                # Sheet 6: Global average statistics across all columns
                avg_accuracy = np.mean([m['accuracy'] for m in results['summary']['results'].values()])
                avg_precision = np.mean([m['precision'] for m in results['summary']['results'].values()])
                avg_recall = np.mean([m['recall'] for m in results['summary']['results'].values()])
                avg_f1 = np.mean([m['f1'] for m in results['summary']['results'].values()])
                avg_confidence = np.mean([m['mean_confidence'] for m in results['summary']['results'].values()])
                
                total_tp = sum([m['confusion']['tp'] for m in results['summary']['results'].values()])
                total_tn = sum([m['confusion']['tn'] for m in results['summary']['results'].values()])
                total_fp = sum([m['confusion']['fp'] for m in results['summary']['results'].values()])
                total_fn = sum([m['confusion']['fn'] for m in results['summary']['results'].values()])
                
                global_avg_data = [{
                    'Metric': 'Average Accuracy',
                    'Value': f"{avg_accuracy:.2f}%"
                }, {
                    'Metric': 'Average Precision',
                    'Value': f"{avg_precision:.4f}"
                }, {
                    'Metric': 'Average Recall',
                    'Value': f"{avg_recall:.4f}"
                }, {
                    'Metric': 'Average F1 Score',
                    'Value': f"{avg_f1:.4f}"
                }, {
                    'Metric': 'Average Confidence',
                    'Value': f"{avg_confidence:.4f}"
                }, {
                    'Metric': 'Total True Positives',
                    'Value': total_tp
                }, {
                    'Metric': 'Total True Negatives',
                    'Value': total_tn
                }, {
                    'Metric': 'Total False Positives',
                    'Value': total_fp
                }, {
                    'Metric': 'Total False Negatives',
                    'Value': total_fn
                }, {
                    'Metric': 'Total Predictions',
                    'Value': total_tp + total_tn + total_fp + total_fn
                }]
                
                global_avg_df = pd.DataFrame(global_avg_data)
                global_avg_df.to_excel(writer, sheet_name='Global Average Stats', index=False)
    
    output.seek(0)
    return output


def plot_confusion_matrix(confusion, column_name):
    """Create a confusion matrix heatmap."""
    conf_matrix = np.array([
        [confusion['tn'], confusion['fp']],
        [confusion['fn'], confusion['tp']]
    ])
    
    fig = go.Figure(data=go.Heatmap(
        z=conf_matrix,
        x=['Predicted 0', 'Predicted 1'],
        y=['Actual 0', 'Actual 1'],
        text=conf_matrix,
        texttemplate='%{text}',
        textfont={"size": 16},
        colorscale='Blues',
        showscale=True
    ))
    
    fig.update_layout(
        title=f'Confusion Matrix - {column_name}',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=400
    )
    
    return fig


def plot_metrics_per_column(summary_results):
    """Create bar plots for metrics per column."""
    columns = list(summary_results.keys())[:20]  # Show first 20 columns max
    
    accuracies = [summary_results[col]['accuracy'] for col in columns]
    precisions = [summary_results[col]['precision'] * 100 for col in columns]
    recalls = [summary_results[col]['recall'] * 100 for col in columns]
    f1_scores = [summary_results[col]['f1'] * 100 for col in columns]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1 Score (%)'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    fig.add_trace(
        go.Bar(x=columns, y=accuracies, name='Accuracy', marker_color='steelblue'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=columns, y=precisions, name='Precision', marker_color='coral'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=columns, y=recalls, name='Recall', marker_color='lightgreen'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(x=columns, y=f1_scores, name='F1 Score', marker_color='plum'),
        row=2, col=2
    )
    
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(range=[0, 100])
    fig.update_layout(height=700, showlegend=False)
    
    return fig


def fetch_and_process_ticker_data(ticker, interval='30min', api_key='TxlfXZeoBpDIPHnFcoWD8pv3KQ3zuJ5V'):
    """
    Fetch data from FMP API and calculate technical indicators.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'TSLA')
        interval: Time interval ('1min', '5min', '30min', '1hour', '4hour', '1day')
        api_key: FMP API key
    
    Returns:
        DataFrame with OHLCV data and technical indicators
    """
    try:
        # Fetch data from FMP
        df = get_fmp_data(ticker, interval=interval, APIKEY=api_key)
        
        if df.empty:
            return None, "No data received from FMP API"
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return None, f"Missing required columns. Got: {df.columns.tolist()}"
        
        # Calculate technical indicators using optimized class
        indicator = Indicator(df[required_cols])
        df_indicators = indicator.construire_arnaud(df)
        return df_indicators, None
        
    except Exception as e:
        return None, f"Error processing ticker data: {str(e)}"


def main():
    st.set_page_config(page_title="Multi-Transformer Inference", layout="wide")
    
    st.title("🔮 Multi-Input Transformer - Inférence")
    st.markdown("Interface pour effectuer des prédictions avec un modèle entrainé.")
    
    # Sidebar - Model Configuration
    with st.sidebar:
        st.header("⚙️ Configuration du Modèle")
        
        model_path = st.text_input(
            "Chemin du modèle (.pth)",
            value="best_transformer_multi_trend_model.pth",
            help="Chemin vers le fichier du modèle entrainé"
        )
        
        st.subheader("Paramètres du modèle")
        
        sequence_length = st.number_input(
            "Sequence Length",
            min_value=10,
            max_value=500,
            value=150,
            help="Longueur de la séquence d'entrée"
        )
        
        prediction_horizon = st.number_input(
            "Prediction Horizon",
            min_value=1,
            max_value=100,
            value=15,
            help="Horizon de prédiction"
        )
        
        n_dropout_samples = st.number_input(
            "Bayesian Samples",
            min_value=10,
            max_value=200,
            value=50,
            help="Nombre d'échantillons pour l'estimation d'incertitude"
        )
        
        with st.expander("Architecture avancée"):
            d_model = st.number_input("D Model", value=128, min_value=32, max_value=512)
            nhead = st.number_input("N Head", value=8, min_value=1, max_value=16)
            num_layers = st.number_input("Num Layers", value=4, min_value=1, max_value=12)
            dim_feedforward = st.number_input("Dim Feedforward", value=512, min_value=64, max_value=2048)
            dropout = st.slider("Dropout", 0.0, 0.5, 0.1)
        
        # Device selection
        device_options = ["auto"]
        if torch.cuda.is_available():
            device_options.append("cuda")
        if torch.backends.mps.is_available():
            device_options.append("mps")
        device_options.append("cpu")
        
        device_str = st.selectbox("Device", device_options)
        if device_str == "auto":
            device = None
        else:
            device = torch.device(device_str)
    
    # Main area
    st.header("📥 Chargement du Modèle")
    
    if not os.path.exists(model_path):
        st.error(f"❌ Le fichier modèle '{model_path}' n'existe pas!")
        st.stop()
    
    # Initialize inference object
    if 'inference' not in st.session_state or st.sidebar.button("🔄 Recharger le modèle"):
        with st.spinner("Chargement du modèle..."):
            try:
                st.session_state.inference = MultiTransformerInference(
                    model_path=model_path,
                    sequence_length=sequence_length,
                    prediction_horizon=prediction_horizon,
                    d_model=d_model,
                    nhead=nhead,
                    num_layers=num_layers,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    device=device
                )
                st.success("✅ Configuration chargée avec succès!")
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement: {str(e)}")
                st.stop()
    
    inference = st.session_state.inference
    
    # Tabs for different inference modes
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Fichier unique", "📁 Dossier", "📈 Ticker en temps réel", "📊 Résultats sauvegardés"])
    
    with tab1:
        st.subheader("Inférence sur un fichier Excel")
        
        uploaded_file = st.file_uploader(
            "Charger un fichier Excel",
            type=['xlsx'],
            help="Sélectionnez un fichier Excel du même format que les données d'entraînement"
        )
        
        if uploaded_file is not None:
            if st.button("🚀 Lancer l'inférence", key="single_file"):
                with st.spinner("Traitement en cours..."):
                    # Save uploaded file temporarily
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    try:
                        result = inference.inference_excel_file(temp_path, n_dropout_samples)
                        
                        if result:
                            st.success("✅ Inférence terminée!")
                            
                            # Display overall accuracy
                            st.metric("Précision Globale", f"{result['overall_accuracy']*100:.2f}%")
                            
                            # Display results per column
                            st.subheader("📊 Résultats par colonne")
                            
                            results_data = []
                            for col, col_result in result['results'].items():
                                results_data.append({
                                    'Colonne': col,
                                    'Prédiction': '↗️ Hausse' if col_result['prediction'] == 1 else '↘️ Baisse',
                                    'Vérité': '↗️ Hausse' if col_result['true_label'] == 1 else '↘️ Baisse',
                                    'Correct': '✅' if col_result['correct'] else '❌',
                                    'Confiance': f"{col_result['confidence']:.4f}",
                                    'Prob. Baisse': f"{col_result['probability_class_0']:.4f}",
                                    'Prob. Hausse': f"{col_result['probability_class_1']:.4f}"
                                })
                            
                            results_df = pd.DataFrame(results_data)
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Statistics
                            correct_count = sum(1 for r in result['results'].values() if r['correct'])
                            total_count = len(result['results'])
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Prédictions Correctes", f"{correct_count}/{total_count}")
                            with col2:
                                avg_confidence = np.mean([r['confidence'] for r in result['results'].values()])
                                st.metric("Confiance Moyenne", f"{avg_confidence:.4f}")
                            with col3:
                                st.metric("Précision", f"{result['overall_accuracy']*100:.2f}%")
                        
                        else:
                            st.error("❌ Impossible de traiter le fichier")
                    
                    except Exception as e:
                        st.error(f"❌ Erreur: {str(e)}")
                    
                    finally:
                        # Clean up temp file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
    
    with tab2:
        st.subheader("Inférence sur un dossier")
        
        folder_path = st.text_input(
            "Chemin du dossier",
            value="datasindAV2025/",
            help="Chemin vers le dossier contenant les fichiers Excel"
        )
        
        if st.button("🚀 Lancer l'inférence sur le dossier", key="folder"):
            if not os.path.isdir(folder_path):
                st.error(f"❌ Le dossier '{folder_path}' n'existe pas!")
            else:
                progress_bar = st.progress(0, "Initialisation...")
                status_text = st.empty()
                
                def progress_callback(current, total, filename):
                    progress_bar.progress(current / total, f"Traitement: {filename} ({current}/{total})")
                    status_text.text(f"Fichier en cours: {filename}")
                
                try:
                    results = inference.inference_folder(
                        folder_path, 
                        n_dropout_samples,
                        progress_callback=progress_callback
                    )
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    if results['summary']:
                        st.success("✅ Inférence terminée!")
                        
                        # Save results in session state
                        st.session_state.folder_results = results
                        
                        # Display summary metrics
                        st.subheader("📈 Métriques Globales")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Précision Globale", f"{results['summary']['overall_accuracy']:.2f}%")
                        with col2:
                            st.metric("Fichiers Traités", results['summary']['processed_files'])
                        with col3:
                            st.metric("Fichiers Totaux", results['summary']['total_files'])
                        with col4:
                            st.metric("Échecs", results['summary']['failed_files'])
                        
                        # Global precision per column summary
                        st.subheader("🎯 Précision Globale par Colonne")
                        precision_summary_data = []
                        for col, metrics in results['summary']['results'].items():
                            precision_summary_data.append({
                                'Colonne': col,
                                'Précision Globale (%)': f"{metrics['accuracy']:.2f}",
                                'Precision': f"{metrics['precision']:.4f}",
                                'Recall': f"{metrics['recall']:.4f}",
                                'F1 Score': f"{metrics['f1']:.4f}"
                            })
                        
                        precision_summary_df = pd.DataFrame(precision_summary_data)
                        st.dataframe(precision_summary_df, use_container_width=True)
                        
                        # Plot metrics per column
                        st.subheader("📊 Métriques par Colonne")
                        fig = plot_metrics_per_column(results['summary']['results'])
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display detailed metrics table
                        st.subheader("📋 Détail des Métriques")
                        
                        metrics_data = []
                        for col, metrics in results['summary']['results'].items():
                            metrics_data.append({
                                'Colonne': col,
                                'Accuracy (%)': f"{metrics['accuracy']:.2f}",
                                'Precision': f"{metrics['precision']:.4f}",
                                'Recall': f"{metrics['recall']:.4f}",
                                'F1 Score': f"{metrics['f1']:.4f}",
                                'Confiance Moy.': f"{metrics['mean_confidence']:.4f}",
                                'TP': metrics['confusion']['tp'],
                                'TN': metrics['confusion']['tn'],
                                'FP': metrics['confusion']['fp'],
                                'FN': metrics['confusion']['fn']
                            })
                        
                        metrics_df = pd.DataFrame(metrics_data)
                        st.dataframe(metrics_df, use_container_width=True)
                        
                        # Confusion matrices (first 6 columns)
                        st.subheader("📊 Matrices de Confusion")
                        cols = list(results['summary']['results'].keys())[:6]
                        
                        for i in range(0, len(cols), 2):
                            col_charts = st.columns(2)
                            
                            with col_charts[0]:
                                col_name = cols[i]
                                confusion = results['summary']['results'][col_name]['confusion']
                                fig = plot_confusion_matrix(confusion, col_name)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            if i + 1 < len(cols):
                                with col_charts[1]:
                                    col_name = cols[i + 1]
                                    confusion = results['summary']['results'][col_name]['confusion']
                                    fig = plot_confusion_matrix(confusion, col_name)
                                    st.plotly_chart(fig, use_container_width=True)
                        
                        # Export button
                        st.subheader("💾 Export des Résultats")
                        excel_data = export_results_to_excel(results)
                        st.download_button(
                            label="📥 Télécharger les résultats (Excel)",
                            data=excel_data,
                            file_name=f"inference_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="download_results_1"
                        )
                    
                    else:
                        st.warning("⚠️ Aucun fichier n'a pu être traité")
                
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Erreur: {str(e)}")
    
    with tab3:
        st.subheader("Inférence sur un Ticker en temps réel")
        
        # Mode selection: Single ticker or Batch from file
        mode = st.radio(
            "Mode d'analyse",
            options=["Ticker unique", "Liste de tickers (Excel)"],
            horizontal=True
        )
        
        interval = st.selectbox(
            "Intervalle",
            options=['1min', '5min', '15min', '30min', '1hour', '4hour', '1day'],
            index=3,
            help="Intervalle de temps pour les données"
        )
        
        # Optional: API Key input
        with st.expander("🔑 Configuration API (optionnel)"):
            api_key = st.text_input(
                "FMP API Key",
                value="TxlfXZeoBpDIPHnFcoWD8pv3KQ3zuJ5V",
                type="password",
                help="Clé API pour Financial Modeling Prep"
            )
        
        if mode == "Ticker unique":
            ticker = st.text_input(
                "Symbole du Ticker",
                value="TSLA",
                help="Entrez le symbole du ticker (ex: TSLA, AAPL, MSFT)"
            ).upper()
        else:
            # File upload for batch processing
            uploaded_ticker_file = st.file_uploader(
                "Charger un fichier Excel avec liste de tickers",
                type=['xlsx'],
                help="Le fichier doit contenir une colonne 'ticker' ou 'Ticker'",
                key="ticker_file_upload"
            )
        
        # Single ticker mode
        if mode == "Ticker unique":
            if st.button("🚀 Récupérer et Analyser", key="ticker_fetch"):
                if not ticker:
                    st.error("❌ Veuillez entrer un symbole de ticker")
                else:
                    with st.spinner(f"Récupération des données pour {ticker}..."):
                        df_with_indicators, error = fetch_and_process_ticker_data(
                            ticker, 
                            interval=interval, 
                            api_key=api_key
                        )
                        
                        if error:
                            st.error(f"❌ {error}")
                        elif df_with_indicators is None:
                            st.error("❌ Impossible de récupérer les données")
                        else:
                            # Check if DataFrame has enough rows
                            required_rows = sequence_length + prediction_horizon
                            if len(df_with_indicators) < required_rows:
                                st.error(f"❌ Données insuffisantes: {len(df_with_indicators)} lignes < {required_rows} requis (sequence_length + prediction_horizon)")
                            else:
                                # Save to session state
                                st.session_state.ticker_data = df_with_indicators
                                st.session_state.ticker_symbol = ticker
                                st.session_state.ticker_interval = interval
                                st.session_state.ticker_mode = "single"
                                st.success(f"✅ Données récupérées pour {ticker} ({len(df_with_indicators)} lignes)")
        
        # Batch ticker mode
        else:
            if uploaded_ticker_file is not None:
                try:
                    # Read ticker list from Excel
                    ticker_df = pd.read_excel(uploaded_ticker_file)
                    
                    # Find ticker column (case insensitive)
                    ticker_col = None
                    for col in ticker_df.columns:
                        if col.lower() == 'ticker':
                            ticker_col = col
                            break
                    
                    if ticker_col is None:
                        st.error("❌ Aucune colonne 'ticker' ou 'Ticker' trouvée dans le fichier")
                    else:
                        ticker_list = ticker_df[ticker_col].dropna().unique().tolist()
                        st.info(f"📋 {len(ticker_list)} tickers trouvés: {', '.join(str(t) for t in ticker_list[:5])}{'...' if len(ticker_list) > 5 else ''}")
                        
                        if st.button("🚀 Analyser tous les tickers", key="batch_ticker_fetch"):
                            batch_results = []
                            progress_bar = st.progress(0, "Initialisation...")
                            status_text = st.empty()
                            
                            for idx, ticker_sym in enumerate(ticker_list):
                                ticker_sym = str(ticker_sym).upper().strip()
                                progress_bar.progress((idx + 1) / len(ticker_list), f"Traitement: {ticker_sym} ({idx + 1}/{len(ticker_list)})")
                                status_text.text(f"Ticker en cours: {ticker_sym}")
                                
                                # Fetch and process data
                                df_with_indicators, error = fetch_and_process_ticker_data(
                                    ticker_sym,
                                    interval=interval,
                                    api_key=api_key
                                )
                                
                                if error or df_with_indicators is None:
                                    batch_results.append({
                                        'ticker': ticker_sym,
                                        'status': 'error',
                                        'error': error or "Unknown error",
                                        'result': None
                                    })
                                    continue
                                
                                # Check if DataFrame has enough rows
                                required_rows = sequence_length + prediction_horizon
                                if len(df_with_indicators) < required_rows:
                                    batch_results.append({
                                        'ticker': ticker_sym,
                                        'status': 'error',
                                        'error': f"DataFrame too short: {len(df_with_indicators)} < {required_rows}",
                                        'result': None
                                    })
                                    continue
                                
                                # Save to temp file and run inference
                                try:
                                    temp_file = f"temp_batch_{ticker_sym}_{interval}.xlsx"
                                    df_with_indicators.to_excel(temp_file, index=False)
                                    
                                    result = inference.inference_excel_file(temp_file, n_dropout_samples)
                                    
                                    if result:
                                        batch_results.append({
                                            'ticker': ticker_sym,
                                            'status': 'success',
                                            'accuracy': result['overall_accuracy'],
                                            'result': result,
                                            'data': df_with_indicators
                                        })
                                    else:
                                        batch_results.append({
                                            'ticker': ticker_sym,
                                            'status': 'error',
                                            'error': "Inference failed",
                                            'result': None
                                        })
                                    
                                    # Clean up
                                    if os.path.exists(temp_file):
                                        os.remove(temp_file)
                                
                                except Exception as e:
                                    batch_results.append({
                                        'ticker': ticker_sym,
                                        'status': 'error',
                                        'error': str(e),
                                        'result': None
                                    })
                                    if os.path.exists(temp_file):
                                        os.remove(temp_file)
                            
                            progress_bar.empty()
                            status_text.empty()
                            
                            # Save batch results to session state
                            st.session_state.batch_ticker_results = batch_results
                            st.session_state.ticker_mode = "batch"
                            
                            # Display summary
                            success_count = sum(1 for r in batch_results if r['status'] == 'success')
                            error_count = len(batch_results) - success_count
                            
                            st.success(f"✅ Traitement terminé: {success_count} réussis, {error_count} échecs")
                
                except Exception as e:
                    st.error(f"❌ Erreur lors de la lecture du fichier: {str(e)}")
        
        # Display results based on mode
        if st.session_state.get('ticker_mode') == 'single' and 'ticker_data' in st.session_state:
            df_with_indicators = st.session_state.ticker_data
            ticker = st.session_state.ticker_symbol
            interval = st.session_state.ticker_interval
            
            # Display data preview
            with st.expander("📊 Aperçu des données"):
                # Show last (sequence_length + prediction_horizon) rows
                n_rows_to_show = sequence_length + prediction_horizon
                st.dataframe(df_with_indicators.tail(n_rows_to_show), use_container_width=True)
                
                # Display columns info
                col_info = st.columns(3)
                with col_info[0]:
                    st.metric("Nombre de lignes", len(df_with_indicators))
                with col_info[1]:
                    st.metric("Nombre de colonnes", len(df_with_indicators.columns))
                with col_info[2]:
                    st.metric("Dernière date", df_with_indicators['date'].iloc[-1] if 'date' in df_with_indicators.columns else "N/A")
            
            # Save to temporary file and run inference
            st.subheader("🔮 Lancer l'inférence")
            
            if st.button("🚀 Analyser avec le modèle", key="ticker_inference"):
                with st.spinner("Inférence en cours..."):
                    try:
                        # Save to temporary Excel file
                        temp_file = f"temp_ticker_{ticker}_{interval}.xlsx"
                        df_with_indicators.to_excel(temp_file, index=False)
                        
                        # Run inference
                        result = inference.inference_excel_file(temp_file, n_dropout_samples)
                        
                        if result:
                            st.success(f"✅ Inférence terminée pour {ticker}!")
                            
                            # Save result to session state
                            st.session_state.ticker_result = result
                            
                            # Display overall metrics
                            st.metric("Précision Globale", f"{result['overall_accuracy']*100:.2f}%")
                            
                            # Display results per indicator
                            st.subheader("📊 Prédictions par Indicateur")
                            
                            results_data = []
                            for col, col_result in result['results'].items():
                                results_data.append({
                                    'Indicateur': col,
                                    'Prédiction': '↗️ Hausse' if col_result['prediction'] == 1 else '↘️ Baisse',
                                    'Vérité': '↗️ Hausse' if col_result['true_label'] == 1 else '↘️ Baisse',
                                    'Correct': '✅' if col_result['correct'] else '❌',
                                    'Confiance': f"{col_result['confidence']:.4f}",
                                    'Prob. Baisse': f"{col_result['probability_class_0']:.4f}",
                                    'Prob. Hausse': f"{col_result['probability_class_1']:.4f}"
                                })
                            
                            results_df = pd.DataFrame(results_data)
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Statistics
                            correct_count = sum(1 for r in result['results'].values() if r['correct'])
                            total_count = len(result['results'])
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Prédictions Correctes", f"{correct_count}/{total_count}")
                            with col2:
                                avg_confidence = np.mean([r['confidence'] for r in result['results'].values()])
                                st.metric("Confiance Moyenne", f"{avg_confidence:.4f}")
                            with col3:
                                st.metric("Précision", f"{result['overall_accuracy']*100:.2f}%")
                            
                            # Export results
                            st.subheader("💾 Export des Résultats")
                            
                            # Create export dataframe
                            export_df = pd.DataFrame(results_data)
                            excel_buffer = BytesIO()
                            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                export_df.to_excel(writer, sheet_name='Predictions', index=False)
                                df_with_indicators.to_excel(writer, sheet_name='Data', index=False)
                            
                            excel_buffer.seek(0)
                            st.download_button(
                                label="📥 Télécharger les résultats (Excel)",
                                data=excel_buffer,
                                file_name=f"inference_{ticker}_{interval}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="download_ticker_results"
                            )
                        
                        else:
                            st.error("❌ Impossible de traiter les données")
                    
                    except Exception as e:
                        st.error(f"❌ Erreur lors de l'inférence: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
                    
                    finally:
                        # Clean up temp file
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
            
            # Display previous result if exists
            elif 'ticker_result' in st.session_state:
                result = st.session_state.ticker_result
                
                st.info("ℹ️ Résultat précédent disponible")
                
                # Display overall metrics
                st.metric("Précision Globale", f"{result['overall_accuracy']*100:.2f}%")
                
                # Display results per indicator
                st.subheader("📊 Prédictions par Indicateur")
                
                results_data = []
                for col, col_result in result['results'].items():
                    results_data.append({
                        'Indicateur': col,
                        'Prédiction': '↗️ Hausse' if col_result['prediction'] == 1 else '↘️ Baisse',
                        'Vérité': '↗️ Hausse' if col_result['true_label'] == 1 else '↘️ Baisse',
                        'Correct': '✅' if col_result['correct'] else '❌',
                        'Confiance': f"{col_result['confidence']:.4f}",
                        'Prob. Baisse': f"{col_result['probability_class_0']:.4f}",
                        'Prob. Hausse': f"{col_result['probability_class_1']:.4f}"
                    })
                
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True)
                
                # Statistics
                correct_count = sum(1 for r in result['results'].values() if r['correct'])
                total_count = len(result['results'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Prédictions Correctes", f"{correct_count}/{total_count}")
                with col2:
                    avg_confidence = np.mean([r['confidence'] for r in result['results'].values()])
                    st.metric("Confiance Moyenne", f"{avg_confidence:.4f}")
                with col3:
                    st.metric("Précision", f"{result['overall_accuracy']*100:.2f}%")
                
                # Export results
                st.subheader("💾 Export des Résultats")
                
                # Create export dataframe
                export_df = pd.DataFrame(results_data)
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    export_df.to_excel(writer, sheet_name='Predictions', index=False)
                    df_with_indicators.to_excel(writer, sheet_name='Data', index=False)
                
                excel_buffer.seek(0)
                st.download_button(
                    label="📥 Télécharger les résultats (Excel)",
                    data=excel_buffer,
                    file_name=f"inference_{ticker}_{interval}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_ticker_results_previous"
                )
        
        # Display batch results
        elif st.session_state.get('ticker_mode') == 'batch' and 'batch_ticker_results' in st.session_state:
            batch_results = st.session_state.batch_ticker_results
            
            st.subheader("📊 Résultats Batch des Tickers")
            
            # Summary metrics
            success_results = [r for r in batch_results if r['status'] == 'success']
            error_results = [r for r in batch_results if r['status'] == 'error']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tickers Réussis", len(success_results))
            with col2:
                st.metric("Tickers Échoués", len(error_results))
            with col3:
                if success_results:
                    avg_accuracy = np.mean([r['accuracy'] for r in success_results])
                    st.metric("Précision Moyenne", f"{avg_accuracy*100:.2f}%")
            
            # Calculate global statistics per transformer output (column/indicator)
            if success_results:
                st.subheader("🎯 Statistiques Globales par Sortie du Transformer")
                
                # Aggregate results across all tickers for each column/indicator
                column_stats = {}
                
                for ticker_result in success_results:
                    result = ticker_result['result']
                    for col, col_result in result['results'].items():
                        if col not in column_stats:
                            column_stats[col] = {
                                'predictions': [],
                                'true_labels': [],
                                'confidences': [],
                                'correct': []
                            }
                        
                        column_stats[col]['predictions'].append(col_result['prediction'])
                        column_stats[col]['true_labels'].append(col_result['true_label'])
                        column_stats[col]['confidences'].append(col_result['confidence'])
                        column_stats[col]['correct'].append(col_result['correct'])
                
                # Calculate metrics for each column
                global_column_metrics = {}
                for col, stats in column_stats.items():
                    predictions = np.array(stats['predictions'])
                    true_labels = np.array(stats['true_labels'])
                    correct = np.array(stats['correct'])
                    confidences = np.array(stats['confidences'])
                    
                    # Confusion matrix
                    tp = np.sum((predictions == 1) & (true_labels == 1))
                    tn = np.sum((predictions == 0) & (true_labels == 0))
                    fp = np.sum((predictions == 1) & (true_labels == 0))
                    fn = np.sum((predictions == 0) & (true_labels == 1))
                    
                    # Metrics
                    accuracy = np.mean(correct) * 100
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    mean_confidence = np.mean(confidences)
                    
                    global_column_metrics[col] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'mean_confidence': mean_confidence,
                        'confusion': {'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)},
                        'total_samples': len(predictions)
                    }
                
                # Display global precision per column
                st.subheader("🎯 Précision Globale par Sortie (Colonne/Indicateur)")
                precision_data = []
                for col, metrics in global_column_metrics.items():
                    precision_data.append({
                        'Sortie/Colonne': col,
                        'Précision Globale (%)': f"{metrics['accuracy']:.2f}",
                        'Precision': f"{metrics['precision']:.4f}",
                        'Recall': f"{metrics['recall']:.4f}",
                        'F1 Score': f"{metrics['f1']:.4f}",
                        'Confiance Moyenne': f"{metrics['mean_confidence']:.4f}",
                        'Échantillons': metrics['total_samples']
                    })
                
                precision_df = pd.DataFrame(precision_data)
                st.dataframe(precision_df, use_container_width=True)
                
                # Plot metrics per column
                st.subheader("📊 Métriques Visuelles par Sortie")
                fig = plot_metrics_per_column(global_column_metrics)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display detailed metrics table
                st.subheader("📋 Détail des Métriques par Sortie")
                detailed_metrics_data = []
                for col, metrics in global_column_metrics.items():
                    detailed_metrics_data.append({
                        'Sortie/Colonne': col,
                        'Accuracy (%)': f"{metrics['accuracy']:.2f}",
                        'Precision': f"{metrics['precision']:.4f}",
                        'Recall': f"{metrics['recall']:.4f}",
                        'F1 Score': f"{metrics['f1']:.4f}",
                        'Confiance Moy.': f"{metrics['mean_confidence']:.4f}",
                        'TP': metrics['confusion']['tp'],
                        'TN': metrics['confusion']['tn'],
                        'FP': metrics['confusion']['fp'],
                        'FN': metrics['confusion']['fn'],
                        'Total': metrics['total_samples']
                    })
                
                detailed_metrics_df = pd.DataFrame(detailed_metrics_data)
                st.dataframe(detailed_metrics_df, use_container_width=True)
                
                # Confusion matrices (first 6 columns)
                st.subheader("📊 Matrices de Confusion par Sortie")
                cols_to_plot = list(global_column_metrics.keys())[:6]
                
                for i in range(0, len(cols_to_plot), 2):
                    col_charts = st.columns(2)
                    
                    with col_charts[0]:
                        col_name = cols_to_plot[i]
                        confusion = global_column_metrics[col_name]['confusion']
                        fig = plot_confusion_matrix(confusion, col_name)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    if i + 1 < len(cols_to_plot):
                        with col_charts[1]:
                            col_name = cols_to_plot[i + 1]
                            confusion = global_column_metrics[col_name]['confusion']
                            fig = plot_confusion_matrix(confusion, col_name)
                            st.plotly_chart(fig, use_container_width=True)
                
                # Global average statistics across all columns
                st.subheader("📊 Statistiques Moyennes Globales")
                avg_accuracy = np.mean([m['accuracy'] for m in global_column_metrics.values()])
                avg_precision = np.mean([m['precision'] for m in global_column_metrics.values()])
                avg_recall = np.mean([m['recall'] for m in global_column_metrics.values()])
                avg_f1 = np.mean([m['f1'] for m in global_column_metrics.values()])
                avg_confidence = np.mean([m['mean_confidence'] for m in global_column_metrics.values()])
                
                total_tp = sum([m['confusion']['tp'] for m in global_column_metrics.values()])
                total_tn = sum([m['confusion']['tn'] for m in global_column_metrics.values()])
                total_fp = sum([m['confusion']['fp'] for m in global_column_metrics.values()])
                total_fn = sum([m['confusion']['fn'] for m in global_column_metrics.values()])
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy Moyenne", f"{avg_accuracy:.2f}%")
                with col2:
                    st.metric("Precision Moyenne", f"{avg_precision:.4f}")
                with col3:
                    st.metric("Recall Moyen", f"{avg_recall:.4f}")
                with col4:
                    st.metric("F1 Score Moyen", f"{avg_f1:.4f}")
                
                # Additional detailed stats
                with st.expander("📈 Statistiques Détaillées Complètes"):
                    global_stats_data = [{
                        'Métrique': 'Average Accuracy',
                        'Valeur': f"{avg_accuracy:.2f}%"
                    }, {
                        'Métrique': 'Average Precision',
                        'Valeur': f"{avg_precision:.4f}"
                    }, {
                        'Métrique': 'Average Recall',
                        'Valeur': f"{avg_recall:.4f}"
                    }, {
                        'Métrique': 'Average F1 Score',
                        'Valeur': f"{avg_f1:.4f}"
                    }, {
                        'Métrique': 'Average Confidence',
                        'Valeur': f"{avg_confidence:.4f}"
                    }, {
                        'Métrique': 'Total True Positives',
                        'Valeur': total_tp
                    }, {
                        'Métrique': 'Total True Negatives',
                        'Valeur': total_tn
                    }, {
                        'Métrique': 'Total False Positives',
                        'Valeur': total_fp
                    }, {
                        'Métrique': 'Total False Negatives',
                        'Valeur': total_fn
                    }, {
                        'Métrique': 'Total Predictions',
                        'Valeur': total_tp + total_tn + total_fp + total_fn
                    }]
                    
                    global_stats_df = pd.DataFrame(global_stats_data)
                    st.dataframe(global_stats_df, use_container_width=True)
            
            # Display summary table
            st.subheader("📋 Tableau Récapitulatif")
            summary_data = []
            for r in batch_results:
                if r['status'] == 'success':
                    summary_data.append({
                        'Ticker': r['ticker'],
                        'Status': '✅ Réussi',
                        'Précision': f"{r['accuracy']*100:.2f}%",
                        'Erreur': '-'
                    })
                else:
                    summary_data.append({
                        'Ticker': r['ticker'],
                        'Status': '❌ Échoué',
                        'Précision': '-',
                        'Erreur': r.get('error', 'Unknown error')
                    })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Detailed results per ticker
            if success_results:
                st.subheader("📈 Détails par Ticker")
                
                selected_ticker = st.selectbox(
                    "Sélectionner un ticker pour voir les détails",
                    options=[r['ticker'] for r in success_results]
                )
                
                # Find selected ticker result
                ticker_result = next((r for r in success_results if r['ticker'] == selected_ticker), None)
                
                if ticker_result:
                    result = ticker_result['result']
                    
                    st.metric("Précision Globale", f"{result['overall_accuracy']*100:.2f}%")
                    
                    # Display predictions
                    st.subheader(f"📊 Prédictions pour {selected_ticker}")
                    
                    results_data = []
                    for col, col_result in result['results'].items():
                        results_data.append({
                            'Indicateur': col,
                            'Prédiction': '↗️ Hausse' if col_result['prediction'] == 1 else '↘️ Baisse',
                            'Vérité': '↗️ Hausse' if col_result['true_label'] == 1 else '↘️ Baisse',
                            'Correct': '✅' if col_result['correct'] else '❌',
                            'Confiance': f"{col_result['confidence']:.4f}",
                            'Prob. Baisse': f"{col_result['probability_class_0']:.4f}",
                            'Prob. Hausse': f"{col_result['probability_class_1']:.4f}"
                        })
                    
                    results_df = pd.DataFrame(results_data)
                    st.dataframe(results_df, use_container_width=True)
            
            # Export all results
            st.subheader("💾 Export des Résultats")
            
            if success_results:
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    # Summary sheet
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
                    # Global statistics per transformer output (if calculated)
                    if 'global_column_metrics' in locals():
                        # Global precision per column sheet
                        global_precision_df = pd.DataFrame([{
                            'Sortie_Colonne': col,
                            'Precision_Globale_Pct': metrics['accuracy'],
                            'Precision_Score': metrics['precision'],
                            'Recall_Score': metrics['recall'],
                            'F1_Score': metrics['f1'],
                            'Confiance_Moyenne': metrics['mean_confidence'],
                            'Total_Samples': metrics['total_samples'],
                            'TP': metrics['confusion']['tp'],
                            'TN': metrics['confusion']['tn'],
                            'FP': metrics['confusion']['fp'],
                            'FN': metrics['confusion']['fn']
                        } for col, metrics in global_column_metrics.items()])
                        global_precision_df.to_excel(writer, sheet_name='Global Stats per Output', index=False)
                        
                        # Global average statistics sheet
                        avg_accuracy = np.mean([m['accuracy'] for m in global_column_metrics.values()])
                        avg_precision = np.mean([m['precision'] for m in global_column_metrics.values()])
                        avg_recall = np.mean([m['recall'] for m in global_column_metrics.values()])
                        avg_f1 = np.mean([m['f1'] for m in global_column_metrics.values()])
                        avg_confidence = np.mean([m['mean_confidence'] for m in global_column_metrics.values()])
                        
                        total_tp = sum([m['confusion']['tp'] for m in global_column_metrics.values()])
                        total_tn = sum([m['confusion']['tn'] for m in global_column_metrics.values()])
                        total_fp = sum([m['confusion']['fp'] for m in global_column_metrics.values()])
                        total_fn = sum([m['confusion']['fn'] for m in global_column_metrics.values()])
                        
                        global_avg_df = pd.DataFrame([{
                            'Metric': 'Average Accuracy (%)',
                            'Value': avg_accuracy
                        }, {
                            'Metric': 'Average Precision',
                            'Value': avg_precision
                        }, {
                            'Metric': 'Average Recall',
                            'Value': avg_recall
                        }, {
                            'Metric': 'Average F1 Score',
                            'Value': avg_f1
                        }, {
                            'Metric': 'Average Confidence',
                            'Value': avg_confidence
                        }, {
                            'Metric': 'Total True Positives',
                            'Value': total_tp
                        }, {
                            'Metric': 'Total True Negatives',
                            'Value': total_tn
                        }, {
                            'Metric': 'Total False Positives',
                            'Value': total_fp
                        }, {
                            'Metric': 'Total False Negatives',
                            'Value': total_fn
                        }, {
                            'Metric': 'Total Predictions',
                            'Value': total_tp + total_tn + total_fp + total_fn
                        }])
                        global_avg_df.to_excel(writer, sheet_name='Global Average Stats', index=False)
                    
                    # Detailed predictions for each ticker
                    for ticker_result in success_results:
                        result = ticker_result['result']
                        ticker_name = ticker_result['ticker'][:25]  # Excel sheet name limit
                        
                        detailed_data = []
                        for col, col_result in result['results'].items():
                            detailed_data.append({
                                'Indicateur': col,
                                'Prédiction': col_result['prediction'],
                                'Vérité': col_result['true_label'],
                                'Correct': col_result['correct'],
                                'Confiance': col_result['confidence'],
                                'Prob_Baisse': col_result['probability_class_0'],
                                'Prob_Hausse': col_result['probability_class_1']
                            })
                        
                        detailed_df = pd.DataFrame(detailed_data)
                        detailed_df.to_excel(writer, sheet_name=ticker_name, index=False)
                
                excel_buffer.seek(0)
                st.download_button(
                    label="📥 Télécharger tous les résultats (Excel)",
                    data=excel_buffer,
                    file_name=f"batch_inference_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_batch_results"
                )
    
    with tab4:
        st.subheader("📊 Résultats Précédents")
        
        if 'folder_results' in st.session_state:
            results = st.session_state.folder_results
            
            st.success("Résultats disponibles en session")
            
            # Display summary
            if results['summary']:
                st.metric("Précision Globale", f"{results['summary']['overall_accuracy']:.2f}%")
                
                # Export button
                excel_data = export_results_to_excel(results)
                st.download_button(
                    label="📥 Télécharger les résultats (Excel)",
                    data=excel_data,
                    file_name=f"inference_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_results_2"
                )
        else:
            st.info("ℹ️ Aucun résultat d'inférence disponible. Effectuez une inférence sur un dossier d'abord.")


if __name__ == "__main__":
    main()
