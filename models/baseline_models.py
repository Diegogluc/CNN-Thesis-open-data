# ==========================================
# models/baseline_models.py
# ==========================================
"""
CNN Architecture Implementations from Literature Review

This module implements CNN architectures from:
- Liu et al. (2020): Compact 1D CNN with tanh activation and minimal layers.
- Rezende et al. (2020): 1D CNN with dropout regularization and dense layers.  
- Azimi et al. (2020): 1D Multi-block CNN with batch normalization and LeakyReLU activation.
- Park et al. (2020): 2D CNN for spectral data classification with multiple convolutional blocks.

Used for comparative analysis in 1D and 2D signal classification tasks.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Activation, Conv1D, MaxPooling1D, 
                                   Conv2D, MaxPooling2D, Flatten, BatchNormalization, 
                                   LeakyReLU, Dropout)


def create_azimi_model(input_shape, num_classes=9):
    """
    Create Azimi et al. (2020) CNN architecture.
    
    1D Multi-block CNN with batch normalization and LeakyReLU activation.
    
    Args:
        input_shape: Shape of input data (sequence_length, features)
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    model = Sequential()
    
    # First convolutional block
    model.add(Conv1D(64, 3, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv1D(64, 3))
    model.add(Activation('relu'))
    model.add(Conv1D(64, 3))
    model.add(Activation('relu'))
    model.add(BatchNormalization(-1, 0.99, 0.001))
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling1D(pool_size=2, padding='valid'))
    
    # Second convolutional block
    model.add(Conv1D(17, 3))
    model.add(Activation('relu'))
    model.add(Conv1D(17, 3))
    model.add(Activation('relu'))
    model.add(Conv1D(17, 3))
    model.add(Activation('relu'))
    model.add(BatchNormalization(-1, 0.99, 0.001))
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling1D(pool_size=2, padding='valid'))
    
    # Third convolutional block
    model.add(Conv1D(65, 3))
    model.add(Activation('relu'))
    model.add(Conv1D(65, 3))
    model.add(Activation('relu'))
    model.add(Conv1D(65, 3))
    model.add(Activation('relu'))
    model.add(BatchNormalization(-1, 0.99, 0.001))
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling1D(pool_size=2, padding='valid'))
    
    # Dense layers
    model.add(Flatten())
    model.add(BatchNormalization(-1, 0.99, 0.001))
    model.add(Dense(80, activation='relu'))
    model.add(Dropout(0.35))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.35))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1.951E-03),
                  metrics=['accuracy', 'Precision'])
    
    return model


def create_park_model(input_shape, num_classes=9):
    """
    Create Park et al. (2020) 2D CNN architecture.
    
    2D CNN for spectral data classification with multiple convolutional blocks.
    
    Args:
        input_shape: Shape of input data (height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
        
    Note:
        This model expects 2D input data (e.g., spectrograms, time-frequency representations)
    """
    model = Sequential()
    
    # First convolutional block
    model.add(Conv2D(32, (4, 4), strides=(1, 1), input_shape=input_shape))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D((5, 5), strides=(1, 1)))
    
    # Second convolutional block
    model.add(Conv2D(49, (4, 4), strides=(1, 1)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D((5, 5), strides=(1, 1)))
    
    # Third convolutional block
    model.add(Conv2D(1, (4, 4), strides=(1, 1)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D((5, 5), strides=(1, 1)))
    
    # Dense layers
    model.add(Flatten())
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=2.380E-3),
                  metrics=['accuracy', 'Precision'])
    
    return model


def create_liu_model(input_shape, num_classes=9):
    """
    Create Liu et al. (2020) CNN architecture.
    
    Compact 1D CNN with tanh activation and minimal layers.
    
    Args:
        input_shape: Shape of input data (sequence_length, features)
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    model = Sequential()
    
    # First convolutional layer
    model.add(Conv1D(32, 5, input_shape=input_shape))
    model.add(Activation('tanh'))
    model.add(MaxPooling1D(pool_size=5))
    
    # Second convolutional layer
    model.add(Conv1D(48, 5, activation='tanh'))
    model.add(Flatten())
    model.add(Activation('tanh'))
    
    # Dense layers
    model.add(Dense(82))
    model.add(Activation('tanh'))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=6.654E-04),
                  metrics=['accuracy', 'Precision'])
    
    return model


def create_rezende_model(input_shape, num_classes=9):
    """
    Create Rezende et al. (2020) CNN architecture.
    
    1D CNN with dropout regularization and dense layers.
    
    Args:
        input_shape: Shape of input data (sequence_length, features)
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    model = Sequential()
    
    # Convolutional layer
    model.add(Conv1D(48, 5, input_shape=input_shape))
    model.add(Activation('tanh'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    
    # Dense layers
    model.add(Dense(100))
    model.add(Activation('tanh'))
    model.add(Dense(100))
    model.add(Activation('tanh'))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=2.015E-04),
                  metrics=['accuracy', 'Precision'])
    
    return model