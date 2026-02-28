"""
CNN Model Architecture for CIFAR-10 Classification

This module defines the CNN architecture used for image classification
on the CIFAR-10 dataset (32x32x3 RGB images, 10 classes).
"""

from typing import Tuple, Optional
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def create_cnn_model(
    input_shape: Tuple[int, int, int] = (32, 32, 3),
    num_classes: int = 10,
    dropout_rate: float = 0.5,
    l2_reg: float = 0.001,
) -> tf.keras.Model:
    """
    Create a CNN model for CIFAR-10 classification.

    Architecture:
    - 3 Convolutional blocks with BatchNorm and MaxPooling
    - 2 Dense layers with Dropout
    - Softmax output layer

    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        l2_reg: L2 regularization factor

    Returns:
        Compiled Keras model
    """
    model = models.Sequential(
        [
            # Input layer
            layers.InputLayer(input_shape=input_shape),
            # First Convolutional Block
            layers.Conv2D(
                32, (3, 3), padding="same", kernel_regularizer=regularizers.l2(l2_reg)
            ),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Conv2D(
                32, (3, 3), padding="same", kernel_regularizer=regularizers.l2(l2_reg)
            ),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Second Convolutional Block
            layers.Conv2D(
                64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(l2_reg)
            ),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Conv2D(
                64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(l2_reg)
            ),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Third Convolutional Block
            layers.Conv2D(
                128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(l2_reg)
            ),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Conv2D(
                128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(l2_reg)
            ),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Dense Layers
            layers.Flatten(),
            layers.Dense(256, kernel_regularizer=regularizers.l2(l2_reg)),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Dropout(dropout_rate),
            layers.Dense(128, kernel_regularizer=regularizers.l2(l2_reg)),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Dropout(dropout_rate),
            # Output Layer
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    return model


def compile_model(
    model: tf.keras.Model, learning_rate: float = 0.001, optimizer: str = "adam"
) -> tf.keras.Model:
    """
    Compile the model with optimizer, loss, and metrics.

    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer
        optimizer: Optimizer name ('adam', 'sgd', 'rmsprop')

    Returns:
        Compiled model
    """
    if optimizer == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "sgd":
        opt = tf.keras.optimizers.SGD(
            learning_rate=learning_rate, momentum=0.9, nesterov=True
        )
    elif optimizer == "rmsprop":
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    model.compile(
        optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def get_model_summary(model: tf.keras.Model) -> str:
    """Get model summary as string."""
    string_list = []
    model.summary(print_fn=lambda x: string_list.append(x))
    return "\n".join(string_list)


def create_simple_cnn(
    input_shape: Tuple[int, int, int] = (32, 32, 3), num_classes: int = 10
) -> tf.keras.Model:
    """
    Create a simpler CNN for faster training (useful for testing).

    Args:
        input_shape: Shape of input images
        num_classes: Number of output classes

    Returns:
        Compiled Keras model
    """
    model = models.Sequential(
        [
            layers.InputLayer(input_shape=input_shape),
            # Conv Block 1
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            # Conv Block 2
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            # Conv Block 3
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            # Dense layers
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    return model
