# ==========================================
# hyperparameter_tuning/park_tuning.py
# ==========================================
"""
Hyperparameter tuning for Park 2D CNN model on IASC dataset.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
import keras_tuner as kt
from data.data_loader import load_iasc_data
from models.model_utils import setup_gpu_memory


def build_tunable_park_model(hp):
    """
    Build a tunable 2D CNN model for Park architecture optimization.
    
    Args:
        hp: HyperParameter object from Keras Tuner
        
    Returns:
        Compiled Keras model
    """
    model = Sequential()
    
    # First convolutional block
    model.add(Conv2D(
        filters=hp.Int('conv1_filters', min_value=16, max_value=64, step=16),
        kernel_size=hp.Choice('conv1_kernel', values=[(3,3), (4,4), (5,5)]),
        strides=(1, 1),
        input_shape=(None, None, None)  # Will be set properly when data is loaded
    ))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D((3, 3), strides=(1, 1)))
    
    # Second convolutional block
    model.add(Conv2D(
        filters=hp.Int('conv2_filters', min_value=32, max_value=64, step=16),
        kernel_size=hp.Choice('conv2_kernel', values=[(3,3), (4,4)]),
        strides=(1, 1)
    ))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D((3, 3), strides=(1, 1)))
    
    # Third convolutional block
    model.add(Conv2D(
        filters=hp.Int('conv3_filters', min_value=1, max_value=8, step=1),
        kernel_size=(4, 4),
        strides=(1, 1)
    ))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D((3, 3), strides=(1, 1)))
    
    # Dense layers
    model.add(Flatten())
    
    # Optional dropout
    if hp.Boolean('use_dropout'):
        model.add(Dropout(hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)))
    
    # Output layer
    model.add(Dense(9, activation='softmax'))
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG')
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def main():
    """Main function for Park model hyperparameter tuning."""
    # Setup GPU
    setup_gpu_memory()
    
    # Load 2D data
    print("Loading IASC 2D dataset for Park model tuning...")
    X, y = load_iasc_data(data_type="park")
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Update input shape in the build function
    def build_model_with_shape(hp):
        model = build_tunable_park_model(hp)
        # Rebuild with correct input shape
        model.build(input_shape=(None,) + X.shape[1:])
        return model
    
    # Create tuner
    tuner = kt.RandomSearch(
        build_model_with_shape,
        objective='val_accuracy',
        max_trials=15,
        directory='hyperparameter_tuning/results',
        project_name='park_2d_tuning'
    )
    
    # Search
    print("Starting hyperparameter search for Park model...")
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
    
    print("\nBest hyperparameters for Park model:")
    for param, value in best_hyperparameters.values.items():
        print(f"  {param}: {value}")
    
    # Save results
    best_model.save('hyperparameter_tuning/results/best_park_2d_model.h5')
    
    print("\nPark model hyperparameter tuning complete!")
    print("Best model saved to: hyperparameter_tuning/results/best_park_2d_model.h5")


if __name__ == '__main__':
    main()