# model.py
import tensorflow as tf
from keras import layers, models
from keras.metrics import MeanSquaredError

def build_model(input_dim):
    """
    Small MLP that takes flattened board tensor.
    For better results you could use CNNs over 8x8x12, but MLP is simpler to show.
    """
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # predicts chance to win for side-to-move
    ])
    model.compile(optimizer='adam',
    loss='mean_squared_error',  # or use the class: MeanSquaredError()
    metrics=['mean_squared_error'])
    return model

def save_model(model, path):
    model.save(path)

def load_model(path):
    return tf.keras.models.load_model(path)
