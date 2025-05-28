# ==========================================
# hyperparameter_tuning/iasc_tuning.py
# ==========================================
"""
Hyperparameter tuning for 1D CNN models on IASC dataset.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv1D, MaxPooling1D, Flatten, Dropout
import keras_tuner as kt
from data.data_loader import load_iasc_data
from models.model_utils import setup_gpu_memory


def build_tunable_model(hp):
    """
    Build a tunable CNN model for hyperparameter optimization.
    
    Args:
        hp: HyperParameter object from Keras Tuner
        
    Returns:
        Compiled Keras model
    """
    model = Sequential()
    
    # First convolutional layer
    model.add(Conv1D(
        filters=hp.Int('conv1_filters', min_value=16, max_value=64, step=16),
        kernel_size=hp.Choice('conv1_kernel', values=[3, 5, 7]),
        input_shape=(None, 2)  # Will be set properly when data is loaded
    ))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    
    # Optional second convolutional layer
    if hp.Boolean('use_conv2'):
        model.add(Conv1D(
            filters=hp.Int('conv2_filters', min_value=16, max_value=64, step=16),
            kernel_size=hp.Choice('conv2_kernel', values=[3, 5])
        ))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=2))
    
    model.add(Flatten())
    
    # Dense layers
    model.add(Dense(
        units=hp.Int('dense1_units', min_value=32, max_value=128, step=32),
        activation='relu'
    ))
    
    # Dropout
    model.add(Dropout(hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)))
    
    # Output layer
    model.add(Dense(9, activation='softmax'))
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def main():
    """Main function for hyperparameter tuning."""
    # Setup GPU
    setup_gpu_memory()
    
    # Load data
    print("Loading IASC 1D dataset...")
    X, y = load_iasc_data(data_type="1D", channels=2)
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Update input shape in the build function
    def build_model_with_shape(hp):
        model = build_tunable_model(hp)
        # Rebuild with correct input shape
        model.build(input_shape=(None,) + X.shape[1:])
        return model
    
    # Create tuner
    tuner = kt.RandomSearch(
        build_model_with_shape,
        objective='val_accuracy',
        max_trials=20,
        directory='hyperparameter_tuning/results',
        project_name='iasc_1d_tuning'
    )
    
    # Search
    print("Starting hyperparameter search...")
    tuner.search(
        X, y,
        epochs=10,
        validation_split=0.2,
        batch_size=32,
        verbose=1
    )
    
    # Get best model
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print("\nBest hyperparameters:")
    for param, value in best_hyperparameters.values.items():
        print(f"  {param}: {value}")
    
    # Save results
    best_model.save('hyperparameter_tuning/results/best_iasc_1d_model.h5')
    
    print("\nHyperparameter tuning complete!")
    print("Best model saved to: hyperparameter_tuning/results/best_iasc_1d_model.h5")


if __name__ == '__main__':
    main()