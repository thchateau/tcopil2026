#!/bin/bash
# Script pour lancer l'interface Streamlit d'inférence

echo "🚀 Lancement de l'interface Streamlit d'inférence Multi-Transformer"
echo ""

# Vérifier que streamlit est installé
if ! command -v streamlit &> /dev/null
then
    echo "❌ Streamlit n'est pas installé"
    echo "Installation avec: pip install streamlit"
    exit 1
fi

# Vérifier que le fichier existe
if [ ! -f "streamlit_inference_multi.py" ]; then
    echo "❌ Le fichier streamlit_inference_multi.py n'existe pas"
    exit 1
fi

# Lancer Streamlit
streamlit run streamlit_inference_multi.py
