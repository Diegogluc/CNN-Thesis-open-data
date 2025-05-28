# ==========================================
# models/model_utils.py
# ==========================================
"""
Utility functions for model training and evaluation.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf


def setup_gpu_memory(memory_limit=7168):
    """
    Configure GPU memory limit for TensorFlow.
    
    Args:
        memory_limit: Memory limit in MB (default: 7168 MB = 7GB)
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
            )
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")


def evaluate_model_kfold(model_func, X, y, num_folds=5, batch_size=64, 
                        epochs=30, verbose=1, random_state=42):
    """
    Evaluate model using k-fold cross validation.
    
    Args:
        model_func: Function that returns a compiled model
        X, y: Training data and labels
        num_folds: Number of folds for cross validation
        batch_size: Training batch size
        epochs: Number of training epochs
        verbose: Training verbosity
        random_state: Random seed for reproducibility
        
    Returns:
        dict: Dictionary with accuracy and loss statistics
    """
    acc_per_fold = []
    loss_per_fold = []
    
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    
    for fold_no, (train, test) in enumerate(skf.split(X, y.argmax(1))):
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no+1} ...')
        
        # Create new model for this fold
        model = model_func()
        
        # Train model
        history = model.fit(X[train], y[train], 
                          batch_size=batch_size, 
                          epochs=epochs, 
                          verbose=verbose)
        
        # Evaluate model
        scores = model.evaluate(X[test], y[test], verbose=0)
        print(f'Score for fold {fold_no+1}: {model.metrics_names[0]} of {scores[0]}; '
              f'{model.metrics_names[1]} of {scores[1]*100}%')
        
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])
        
        # Clean up
        del model, history
    
    # Calculate statistics
    results = {
        'accuracy_per_fold': acc_per_fold,
        'loss_per_fold': loss_per_fold,
        'mean_accuracy': np.mean(acc_per_fold),
        'std_accuracy': np.std(acc_per_fold),
        'mean_loss': np.mean(loss_per_fold)
    }
    
    return results


def save_results(results, model_name, output_path="results/"):
    """
    Save cross-validation results to file.
    
    Args:
        results: Results dictionary from evaluate_model_kfold
        model_name: Name of the model (e.g., 'Azimi', 'Liu', 'Rezende')
        output_path: Directory to save results
    """
    import os
    os.makedirs(output_path, exist_ok=True)
    
    filename = os.path.join(output_path, f'kfold_{model_name}.txt')
    
    with open(filename, "w") as f:
        f.write(f"Cross-validation results for {model_name} model\n")
        f.write("="*60 + "\n")
        
        for i, (loss, acc) in enumerate(zip(results['loss_per_fold'], 
                                          results['accuracy_per_fold'])):
            f.write(f"\nFold {i+1} - Loss: {loss:.4f} - Accuracy: {acc:.2f}%\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("Average scores for all folds:\n")
        f.write(f"Accuracy: {results['mean_accuracy']:.2f} "
                f"(+- {results['std_accuracy']:.2f})\n")
        f.write(f"Loss: {results['mean_loss']:.4f}\n")
        f.write("="*60 + "\n")
