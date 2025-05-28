# ==========================================
# scripts/compare_models.py
# ==========================================
"""
Compare 1D CNN architectures on IASC dataset.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import load_iasc_data
from models.baseline_models import create_azimi_model, create_liu_model, create_rezende_model
from models.model_utils import evaluate_model_kfold, save_results


def compare_1d_models(epochs=30, num_folds=5, batch_size=64, models=None):
    """
    Compare 1D CNN models on IASC dataset.
    
    Args:
        epochs: Number of training epochs
        num_folds: Number of cross-validation folds  
        batch_size: Training batch size
        models: List of model names to evaluate (None = all)
    """
    # Load 1D data
    print("Loading IASC 1D dataset...")
    X, y = load_iasc_data(data_type="1D", channels=2)
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Define all available models
    all_models = {
        'azimi': lambda: create_azimi_model(X.shape[1:], 9),
        'liu': lambda: create_liu_model(X.shape[1:], 9),
        'rezende': lambda: create_rezende_model(X.shape[1:], 9)
    }
    
    # Select models to run
    if models is None:
        models_to_run = all_models
    else:
        models_to_run = {name: all_models[name] for name in models if name in all_models}
    
    # Evaluate each model
    all_results = {}
    for model_name, model_func in models_to_run.items():
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name.title()} model...")
        print(f"{'='*60}")
        
        results = evaluate_model_kfold(
            model_func, X, y,
            num_folds=num_folds,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1
        )
        
        all_results[model_name] = results
        save_results(results, model_name.title())
        
        print(f"\n{model_name.title()} Results:")
        print(f"Accuracy: {results['mean_accuracy']:.2f} (+- {results['std_accuracy']:.2f})")
        print(f"Loss: {results['mean_loss']:.4f}")
    
    # Print summary
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY OF ALL MODELS")
        print(f"{'='*60}")
        for model_name, results in all_results.items():
            print(f"{model_name.title():12} - Accuracy: {results['mean_accuracy']:6.2f} "
                  f"(+- {results['std_accuracy']:5.2f}) - Loss: {results['mean_loss']:.4f}")


if __name__ == '__main__':
    compare_1d_models()