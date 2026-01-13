# -*- coding: utf-8 -*-
"""
CNN model architectures for the TP.
- Basic CNN
- ResNet

@author: Tchassi Daniel
@matricule: 21P073
"""
from tensorflow import keras

def build_basic_cnn(input_shape, num_classes):
    """
    Builds a classic CNN architecture as described in the TP.
    Conv -> Pool -> Conv -> Pool -> Flatten -> Dense -> Dense
    """
    model = keras.Sequential([
        # Convolutional Layer 1: 32 filters, 3x3, ReLU
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        # Max Pooling Layer 1: 2x2
        keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Convolutional Layer 2: 64 filters, 3x3, ReLU
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        # Max Pooling Layer 2: 2x2
        keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten for transition to Dense layers
        keras.layers.Flatten(),

        # Dense Layer 1: 512 units, ReLU
        keras.layers.Dense(512, activation='relu'),
        # Output Layer: num_classes units, Softmax
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def residual_block(x, filters, kernel_size=(3, 3), stride=1):
    """
    Simplified residual block with a skip connection.
    """
    # Main path
    y = keras.layers.Conv2D(filters, kernel_size, strides=stride, padding='same', activation='relu')(x)
    y = keras.layers.Conv2D(filters, kernel_size, padding='same')(y)

    # Skip connection path
    if stride > 1 or x.shape[-1] != filters:
        # Adapt dimensions if necessary (e.g., with a 1x1 conv)
        x = keras.layers.Conv2D(filters, (1, 1), strides=stride, padding='same')(x)

    # Add skip connection to main path
    z = keras.layers.Add()([x, y])
    z = keras.layers.Activation('relu')(z)
    return z

def build_resnet(input_shape, num_classes):
    """
    Builds a small ResNet architecture using residual blocks.
    """
    input_layer = keras.Input(shape=input_shape)

    # Initial Conv layer
    x = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)

    # Residual blocks as per TP suggestion (3 consecutive blocks)
    x = residual_block(x, 32)
    x = residual_block(x, 64, stride=2)  # Dimension reduction
    x = residual_block(x, 64)

    # Global Average Pooling to reduce dimensions before Dense layers
    x = keras.layers.GlobalAveragePooling2D()(x)

    # Classifier
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    output = keras.layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output)
    return model
