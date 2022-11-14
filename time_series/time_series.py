#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
import time
import re

# %%
df = pd.read_csv("inputs/database.csv")
# %%
df.columns.tolist()
df = df[["Date", "Time", "Latitude", "Longitude", "Depth", "Magnitude"]]
df.head()
# %%
df["Timestamp"] = df["Date"] + " " + df["Time"]
tmp = df.Timestamp.loc[df["Timestamp"].str.contains(r"T|Z")]
tmp = tmp.apply(lambda x: " ".join(re.split("T|Z", x)[:2]))
df.loc[df["Timestamp"].str.contains(r"T|Z"), "Timestamp"] = tmp.values
df["Timestamp"] = pd.to_datetime(
    df["Timestamp"], format="%m/%d/%Y %H:%M:%S", errors="coerce"
)

df["Timestamp"] = (df["Timestamp"] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
# %%
X = df[["Timestamp", "Latitude", "Longitude"]]
y = df[["Magnitude", "Depth"]]
import sklearn
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(X_train.shape, X_test.shape, y_train.shape, X_test.shape)
# %%
from keras.models import Sequential
from keras.layers import Dense


def create_model(neurons, activation, optimizer, loss):
    model = Sequential()
    model.add(Dense(neurons, activation=activation, input_shape=(3,)))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(2, activation="softmax"))

    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    return model


# %%
from keras.wrappers.scikit_learn import KerasClassifier

model = KerasClassifier(build_fn=create_model, verbose=0)

neurons = [16, 64, 128, 256]
# neurons = [16]
batch_size = [10, 20, 50, 100]
# batch_size = [10]
epochs = [10]
activation = ["relu", "tanh", "sigmoid", "hard_sigmoid", "linear", "exponential"]
# activation = ['sigmoid', 'relu']
optimizer = ["SGD", "RMSprop", "Adagrad", "Adadelta", "Adam", "Adamax", "Nadam"]
# optimizer = ['SGD', 'Adadelta']
loss = ["squared_hinge"]

param_grid = dict(
    neurons=neurons,
    batch_size=batch_size,
    epochs=epochs,
    activation=activation,
    optimizer=optimizer,
    loss=loss,
)

from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_["mean_test_score"]
stds = grid_result.cv_results_["std_test_score"]
params = grid_result.cv_results_["params"]
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# %%
model = Sequential()
model.add(Dense(16, activation="relu", input_shape=(3,)))
model.add(Dense(16, activation="relu"))
model.add(Dense(2, activation="softmax"))

model.compile(optimizer="SGD", loss="squared_hinge", metrics=["accuracy"])
model.fit(
    X_train,
    y_train,
    batch_size=10,
    epochs=20,
    verbose=1,
    validation_data=(X_test, y_test),
)

[test_loss, test_acc] = model.evaluate(X_test, y_test)
print(
    "Evaluation result on Test Data : Loss = {}, accuracy = {}".format(
        test_loss, test_acc
    )
)
