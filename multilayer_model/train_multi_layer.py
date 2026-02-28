from utils import load_train_dataset
from multi_layer import build_mlp
import tensorflow as tf

# config gpu
tf.config.list_physical_devices('GPU')

# load data
x_train, y_train = load_train_dataset("data/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv")

# build the model
model = build_mlp()

# train the model
model.fit(x_train, y_train, epochs=20, batch_size=32)

# save the model
model.save("multilayer_model.h5")

print("Model saved")