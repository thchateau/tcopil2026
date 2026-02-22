# Inférence Multi-Input Transformer

Ce module permet d'effectuer des inférences avec le modèle Multi-Input Transformer entrainé.

## Fichiers

- **`inference_multi.py`** : Classe Python pour l'inférence
- **`streamlit_inference_multi.py`** : Interface Streamlit interactive
- **`test_inference_multi.py`** : Script de test

## Classe `MultiTransformerInference`

### Constructeur

```python
from inference_multi import MultiTransformerInference

inference = MultiTransformerInference(
    model_path="best_transformer_multi_trend_model.pth",
    sequence_length=150,
    prediction_horizon=15,
    d_model=128,
    nhead=8,
    num_layers=4,
    dim_feedforward=512,
    dropout=0.1,
    device=None  # Auto-detect par défaut
)
```

### Méthodes

#### 1. `inference_dataframe(df, n_dropout_samples=50)`

Effectue une inférence sur un DataFrame pandas.

**Arguments:**
- `df`: DataFrame avec les données numériques
- `n_dropout_samples`: Nombre d'échantillons pour l'estimation d'incertitude (default: 50)

**Retourne:**
- Dictionnaire avec les prédictions, labels, confidences et métriques
- `None` si l'inférence échoue

**Exemple:**
```python
import pandas as pd

df = pd.read_excel("data.xlsx")
result = inference.inference_dataframe(df, n_dropout_samples=50)

if result:
    print(f"Overall Accuracy: {result['overall_accuracy']*100:.2f}%")
    for col, col_result in result['results'].items():
        print(f"{col}: Prediction={col_result['prediction']}, Confidence={col_result['confidence']:.4f}")
```

#### 2. `inference_excel_file(excel_path, n_dropout_samples=50)`

Effectue une inférence sur un fichier Excel.

**Arguments:**
- `excel_path`: Chemin vers le fichier Excel
- `n_dropout_samples`: Nombre d'échantillons pour l'estimation d'incertitude (default: 50)

**Retourne:**
- Dictionnaire avec les résultats (même format que `inference_dataframe`)
- `None` si l'inférence échoue

**Exemple:**
```python
result = inference.inference_excel_file("data/AAPL_2023.xlsx", n_dropout_samples=50)

if result:
    print(f"File: {result['file_name']}")
    print(f"Overall Accuracy: {result['overall_accuracy']*100:.2f}%")
```

#### 3. `inference_folder(folder_path, n_dropout_samples=50, progress_callback=None)`

Effectue une inférence sur tous les fichiers Excel d'un dossier.

**Arguments:**
- `folder_path`: Chemin vers le dossier contenant les fichiers Excel
- `n_dropout_samples`: Nombre d'échantillons pour l'estimation d'incertitude (default: 50)
- `progress_callback`: Fonction de callback pour suivre la progression (optionnel)

**Retourne:**
- Dictionnaire avec:
  - `files`: Liste des résultats par fichier
  - `summary`: Métriques agrégées et matrice de confusion

**Exemple:**
```python
def progress_callback(current, total, filename):
    print(f"Processing {filename} ({current}/{total})")

results = inference.inference_folder(
    "datasindAV2025/",
    n_dropout_samples=50,
    progress_callback=progress_callback
)

if results['summary']:
    print(f"Overall Accuracy: {results['summary']['overall_accuracy']:.2f}%")
    print(f"Processed: {results['summary']['processed_files']}/{results['summary']['total_files']}")
    
    # Métriques par colonne
    for col, metrics in results['summary']['results'].items():
        print(f"{col}:")
        print(f"  Accuracy: {metrics['accuracy']:.2f}%")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
```

## Interface Streamlit

### Lancement

```bash
streamlit run streamlit_inference_multi.py
```

### Fonctionnalités

L'interface Streamlit propose 3 onglets :

#### 1. 📄 Fichier unique
- Charger un fichier Excel
- Lancer l'inférence
- Visualiser les résultats par colonne avec les prédictions, confidences et probabilités

#### 2. 📁 Dossier
- Spécifier un dossier contenant plusieurs fichiers Excel
- Lancer l'inférence sur tous les fichiers
- Visualiser les métriques globales (accuracy, precision, recall, F1)
- Graphiques interactifs des métriques par colonne
- Matrices de confusion pour les principales colonnes
- Export des résultats en Excel

#### 3. 📊 Résultats sauvegardés
- Accès aux résultats de la dernière inférence sur dossier
- Possibilité de télécharger les résultats en Excel

### Configuration

Dans la barre latérale, vous pouvez ajuster :

- **Chemin du modèle** : Fichier .pth du modèle entrainé
- **Sequence Length** : Longueur de la séquence (default: 150)
- **Prediction Horizon** : Horizon de prédiction (default: 15)
- **Bayesian Samples** : Nombre d'échantillons dropout (default: 50)
- **Architecture** : d_model, nhead, num_layers, dim_feedforward, dropout
- **Device** : auto, cuda, mps, ou cpu

## Script de Test

Pour tester rapidement la classe d'inférence :

```bash
python test_inference_multi.py
```

Ce script :
1. Initialise la classe d'inférence
2. Teste l'inférence sur un fichier unique
3. Teste l'inférence sur un dossier (3 premiers fichiers)
4. Affiche les résultats et métriques

## Format des Données

Les fichiers Excel doivent avoir le même format que ceux utilisés pour l'entraînement :
- Colonnes numériques représentant les indicateurs
- Au moins `sequence_length + prediction_horizon` lignes (default: 165 lignes)
- Les dernières lignes sont utilisées pour la prédiction

## Résultats d'Inférence

### Structure des résultats (fichier unique)

```python
{
    'results': {
        'column_name': {
            'prediction': 0 ou 1,  # 0=baisse, 1=hausse
            'true_label': 0 ou 1,
            'correct': True ou False,
            'confidence': 0.0 à 1.0,
            'probability_class_0': 0.0 à 1.0,
            'probability_class_1': 0.0 à 1.0
        },
        ...
    },
    'overall_accuracy': 0.0 à 1.0,
    'predictions': numpy array,
    'labels': numpy array,
    'confidences': numpy array
}
```

### Structure des résultats (dossier)

```python
{
    'files': [
        # Liste des résultats par fichier (même structure que ci-dessus)
    ],
    'summary': {
        'results': {
            'column_name': {
                'accuracy': 0.0 à 100.0,
                'precision': 0.0 à 1.0,
                'recall': 0.0 à 1.0,
                'f1': 0.0 à 1.0,
                'mean_confidence': 0.0 à 1.0,
                'confusion': {
                    'tp': int,
                    'tn': int,
                    'fp': int,
                    'fn': int
                }
            },
            ...
        },
        'overall_accuracy': 0.0 à 100.0,
        'total_files': int,
        'processed_files': int,
        'failed_files': int
    }
}
```

## Export Excel

L'interface Streamlit permet d'exporter les résultats dans un fichier Excel avec plusieurs feuilles :

1. **Files Summary** : Résumé par fichier
2. **Detailed Predictions** : Prédictions détaillées par fichier et colonne
3. **Summary Metrics** : Métriques agrégées par colonne
4. **Overall Statistics** : Statistiques globales
5. **Confusion Matrices** : Matrices de confusion par colonne

## Notes

- Le modèle doit être compatible avec l'architecture définie (même nombre de features et targets)
- Le premier fichier traité détermine les colonnes utilisées pour tous les autres
- Les valeurs manquantes (NaN) sont traitées par forward/backward fill
- L'estimation d'incertitude utilise le dropout bayésien (MC Dropout)

## Dépendances

```
torch
pandas
numpy
openpyxl
streamlit
plotly
```
