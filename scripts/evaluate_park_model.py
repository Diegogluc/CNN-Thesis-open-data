# ==========================================
# scripts/evaluate_park_model.py
# ==========================================
"""
Evaluate Park et al. (2020) 2D CNN model on IASC dataset.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import load_iasc_data
from models.baseline_models import create_park_model
from models.model_utils import evaluate_model_kfold, save_results


def evaluate_park_model(epochs=30, num_folds=5, batch_size=64):
    """
    Evaluate Park 2D CNN model on IASC dataset.
    
    Args:
        epochs: Number of training epochs
        num_folds: Number of cross-validation folds
        batch_size: Training batch size
    """
    # Load 2D data for Park model
    print("Loading IASC 2D dataset for Park model...")
    X, y = load_iasc_data(data_type="park")
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Create Park model function
    park_model_func = lambda: create_park_model(X.shape[1:], 9)
    
    print(f"\n{'='*60}")
    print("Evaluating Park 2D CNN model...")
    print(f"{'='*60}")
    
    # Evaluate Park model
    results = evaluate_model_kfold(
        park_model_func, X, y,
        num_folds=num_folds,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1
    )
    
    save_results(results, "Park")
    
    print(f"\nPark Model Results:")
    print(f"Accuracy: {results['mean_accuracy']:.2f} (+- {results['std_accuracy']:.2f})")
    print(f"Loss: {results['mean_loss']:.4f}")


if __name__ == '__main__':
    evaluate_park_model()