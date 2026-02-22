#!/usr/bin/env python3
"""
Script de test pour la classe d'inférence MultiTransformerInference.
"""

from inference_multi import MultiTransformerInference
import os

def test_inference():
    """Test the inference class with a simple example."""
    
    # Configuration
    model_path = "best_transformer_multi_trend_model.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file '{model_path}' not found!")
        print("Available .pth files:")
        pth_files = [f for f in os.listdir('.') if f.endswith('.pth')]
        for f in pth_files:
            print(f"  - {f}")
        return
    
    print("=" * 60)
    print("Test de la classe MultiTransformerInference")
    print("=" * 60)
    
    # Initialize inference
    print("\n1. Initialisation de la classe d'inférence...")
    inference = MultiTransformerInference(
        model_path=model_path,
        sequence_length=150,
        prediction_horizon=15
    )
    print("✅ Classe initialisée")
    
    # Test on a single file
    print("\n2. Test sur un fichier unique...")
    test_folder = "datasindAV2025"
    
    if not os.path.exists(test_folder):
        print(f"❌ Test folder '{test_folder}' not found!")
        return
    
    excel_files = [f for f in os.listdir(test_folder) if f.endswith('.xlsx')]
    
    if not excel_files:
        print(f"❌ No Excel files found in '{test_folder}'")
        return
    
    test_file = os.path.join(test_folder, excel_files[0])
    print(f"   Fichier test: {test_file}")
    
    result = inference.inference_excel_file(test_file, n_dropout_samples=30)
    
    if result:
        print("✅ Inférence réussie!")
        print(f"   Overall Accuracy: {result['overall_accuracy']*100:.2f}%")
        print(f"   Nombre de colonnes: {len(result['results'])}")
        
        # Show first 5 predictions
        print("\n   Premières prédictions:")
        for i, (col, res) in enumerate(list(result['results'].items())[:5]):
            pred_label = "↗️ Hausse" if res['prediction'] == 1 else "↘️ Baisse"
            true_label = "↗️ Hausse" if res['true_label'] == 1 else "↘️ Baisse"
            correct = "✅" if res['correct'] else "❌"
            print(f"      {col}: Pred={pred_label}, True={true_label} {correct} (conf={res['confidence']:.4f})")
    else:
        print("❌ Échec de l'inférence sur le fichier")
    
    # Test on folder (first 3 files only)
    print(f"\n3. Test sur un dossier (3 premiers fichiers)...")
    
    # Temporarily limit files for testing
    import glob
    all_files = glob.glob(os.path.join(test_folder, "*.xlsx"))
    test_files = all_files[:3]
    
    if len(all_files) > 3:
        # Create a temp folder with only 3 files (symlinks)
        temp_folder = "temp_test_folder"
        os.makedirs(temp_folder, exist_ok=True)
        
        import shutil
        for f in test_files:
            basename = os.path.basename(f)
            shutil.copy2(f, os.path.join(temp_folder, basename))
        
        results = inference.inference_folder(temp_folder, n_dropout_samples=30)
        
        # Clean up
        shutil.rmtree(temp_folder)
    else:
        results = inference.inference_folder(test_folder, n_dropout_samples=30)
    
    if results['summary']:
        print("✅ Inférence sur dossier réussie!")
        print(f"   Overall Accuracy: {results['summary']['overall_accuracy']:.2f}%")
        print(f"   Fichiers traités: {results['summary']['processed_files']}/{results['summary']['total_files']}")
        print(f"   Échecs: {results['summary']['failed_files']}")
        
        # Show metrics for first 3 columns
        print("\n   Métriques pour les 3 premières colonnes:")
        for i, (col, metrics) in enumerate(list(results['summary']['results'].items())[:3]):
            print(f"      {col}:")
            print(f"         Accuracy: {metrics['accuracy']:.2f}%")
            print(f"         Precision: {metrics['precision']:.4f}")
            print(f"         Recall: {metrics['recall']:.4f}")
            print(f"         F1: {metrics['f1']:.4f}")
    else:
        print("❌ Échec de l'inférence sur le dossier")
    
    print("\n" + "=" * 60)
    print("Test terminé!")
    print("=" * 60)


if __name__ == "__main__":
    test_inference()
