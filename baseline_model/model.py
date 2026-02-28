import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
import tensorflow as tf

from utils import load_train_dataset

import pickle

# config gpu
tf.config.list_physical_devices('GPU')

# load data
x_train, y_train = load_train_dataset("data/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv")

np.random.seed(42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)


#parameters for RandomForestClassifier
rrf_params = {
    'n_estimators': [200, 300, 500],
    'max_depth': [20, 30, 40, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

# train the model
model = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=rrf_params,
    n_iter=50,
    cv=5,
    verbose=True,
)
model.fit(X_train_scaled, y_train)

#saving the baseline model
MODEL_PATH = "baseline_model.pkl"
with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
print(f"Model is saved as {MODEL_PATH}")

