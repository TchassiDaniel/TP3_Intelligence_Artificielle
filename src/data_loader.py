# -*- coding: utf-8 -*-
"""
Data loading and preprocessing for CIFAR-10.

@author: Tchassi Daniel
@matricule: 21P073
"""
import tensorflow as tf
from tensorflow import keras

def load_cifar10_data():
    """
    Loads and preprocesses the CIFAR-10 dataset.

    Returns:
        tuple: A tuple containing (x_train, y_train), (x_test, y_test),
               input_shape, and num_classes.
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    num_classes = 10
    input_shape = x_train.shape[1:]

    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Convert labels to One-Hot Encoding format
    y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)

    print("CIFAR-10 data loaded and preprocessed:")
    print(f" - Input data shape: {input_shape}")
    print(f" - Number of classes: {num_classes}")

    return (x_train, y_train), (x_test, y_test), input_shape, num_classes
