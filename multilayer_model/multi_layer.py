import tensorflow as tf
from tensorflow.keras import layers
from utils import data_process

train_df = data_process.load_train_dataset("../data/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv")

def build_mlp(input_dim=784, n_classes=25, hidden_units=(128, 64, 24), activation="relu", **kwargs):
    """Build a sequential MLP with configurable hidden layers and activation."""
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_dim,)))
    for units in hidden_units:
        model.add(tf.keras.layers.Dense(units, activation=activation, **kwargs))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(n_classes, activation="softmax"))
    
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    return model