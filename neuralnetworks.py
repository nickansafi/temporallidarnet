import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv1D, BatchNormalization, TimeDistributed, LSTM, Reshape

def original(num_lidar_range_values):
    return tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=24, kernel_size=10, strides=4, activation='relu', input_shape=(num_lidar_range_values, 1)),
    tf.keras.layers.Conv1D(filters=36, kernel_size=8, strides=4, activation='relu'),
    tf.keras.layers.Conv1D(filters=48, kernel_size=4, strides=2, activation='relu'),
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(2, activation='tanh')
])

def pilotnet_x3_time_distributed(img_shape):
    return tf.keras.Sequential([
    tf.keras.layers.BatchNormalization(epsilon=0.001, axis=-1, input_shape=(3, img_shape, 1)),
    tf.keras.layers.TimeDistributed(Conv1D(filters=24, kernel_size=5, strides=2, activation="relu", padding='same')),
    tf.keras.layers.TimeDistributed(Conv1D(filters=36, kernel_size=5, strides=2, activation="relu", padding='same')),
    tf.keras.layers.TimeDistributed(Conv1D(filters=48, kernel_size=5, strides=2, activation="relu", padding='same')),
    tf.keras.layers.TimeDistributed(Conv1D(filters=64, kernel_size=3, strides=1, activation="relu", padding='same')),
    tf.keras.layers.TimeDistributed(Conv1D(filters=64, kernel_size=3, strides=1, activation="relu", padding='same')),
    tf.keras.layers.TimeDistributed(Flatten()),
    # tf.keras.layers.Reshape((-1, 64)),
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(2)
    ])