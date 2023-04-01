#%%
import pandas as pd
import re
import tensorflow as tf
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV

# tf.test.is_built_with_cuda()
# tf.test.is_gpu_available(cuda_only=True)
# %%
df = pd.read_csv("inputs/database.csv")
df.columns.tolist()
df = df[["Date", "Time", "Latitude", "Longitude", "Depth", "Magnitude"]]
df.head()
df["Timestamp"] = df["Date"] + " " + df["Time"]
tmp = df.Timestamp.loc[df["Timestamp"].str.contains(r"T|Z")]
tmp = tmp.apply(lambda x: " ".join(re.split("T|Z", x)[:2]))
tmp = tmp.apply(lambda x: x[:-4])
tmp = tmp.apply(lambda x: re.split("\s|-", x))
tmp = tmp.apply(lambda x: "/".join((x[1], x[2], x[0])) + " " + x[3])
df.loc[df["Timestamp"].str.contains(r"T|Z"), "Timestamp"] = tmp.values
df["Timestamp"] = pd.to_datetime(
    df["Timestamp"], format="%m/%d/%Y %H:%M:%S", errors="coerce"
)

df["Timestamp"] = (df["Timestamp"] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
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
from keras.layers import Dense, Activation


def create_model(neurons, activation, optimizer, loss):
    model = Sequential()
    model.add(Dense(neurons, activation=activation, input_shape=(3,)))
    model.add(Dense(2, activation="softmax"))

    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    return model


# %%
with tf.device("/GPU:0"):
    model = KerasRegressor(
        build_fn=create_model,
        verbose=0,
        neurons=16,  # [16]
        batch_size=[10],
        epochs=[10],
        activation="sigmoid",  # ["sigmoid", "relu"],
        optimizer=["SGD", "Adadelta"],
        loss=["squared_hinge"],
    )

    # neurons = [16, 64, 128, 256]
    neurons = [16]
    # batch_size = [10, 20, 50, 100]
    batch_size = [10]
    epochs = [10]
    # activation = ["relu", "tanh", "sigmoid", "hard_sigmoid", "linear", "exponential"]
    activation = ["sigmoid", "relu"]

    # optimizer = ["SGD", "RMSprop", "Adagrad", "Adadelta", "Adam", "Adamax", "Nadam"]
    optimizer = ["SGD", "Adadelta"]
    loss = ["squared_hinge"]

    param_grid = dict(
        neurons=neurons,
        batch_size=batch_size,
        epochs=epochs,
        activation=activation,
        optimizer=optimizer,
        loss=loss,
    )

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
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


# #%%
# import os

# # import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.datasets import cifar10

# # physical_devices = tf.config.list_physical_devices("GPU")
# # tf.config.experimental.set_memory_growth(physical_devices[0], True)

# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x_train = x_train.astype("float32") / 255.0
# x_test = x_test.astype("float32") / 255.0

# model = keras.Sequential(
#     [
#         keras.Input(shape=(32, 32, 3)),
#         layers.Conv2D(32, 3, padding="valid", activation="relu"),
#         layers.MaxPooling2D(),
#         layers.Conv2D(64, 3, activation="relu"),
#         layers.MaxPooling2D(),
#         layers.Conv2D(128, 3, activation="relu"),
#         layers.Flatten(),
#         layers.Dense(64, activation="relu"),
#         layers.Dense(10),
#     ]
# )


# def my_model():
#     inputs = keras.Input(shape=(32, 32, 3))
#     x = layers.Conv2D(32, 3)(inputs)
#     x = layers.BatchNormalization()(x)
#     x = keras.activations.relu(x)
#     x = layers.MaxPooling2D()(x)
#     x = layers.Conv2D(64, 3)(x)
#     x = layers.BatchNormalization()(x)
#     x = keras.activations.relu(x)
#     x = layers.MaxPooling2D()(x)
#     x = layers.Conv2D(128, 3)(x)
#     x = layers.BatchNormalization()(x)
#     x = keras.activations.relu(x)
#     x = layers.Flatten()(x)
#     x = layers.Dense(64, activation="relu")(x)
#     outputs = layers.Dense(10)(x)
#     model = keras.Model(inputs=inputs, outputs=outputs)
#     return model


# model = my_model()
# model.compile(
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     optimizer=keras.optimizers.Adam(learning_rate=3e-4),
#     metrics=["accuracy"],
# )

# model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)
# model.evaluate(x_test, y_test, batch_size=64, verbose=2)

# #%%
# import numpy
# import matplotlib.pyplot as plt
# import pandas
# import math
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error

# # fix random seed for reproducibility
# numpy.random.seed(7)

# # load the dataset
# dataframe = pandas.read_csv(
#     "inputs/airline-passengers.csv", usecols=[1], engine="python"
# )
# dataset = dataframe.values
# dataset = dataset.astype("float32")
# # normalize the dataset
# scaler = MinMaxScaler(feature_range=(0, 1))
# dataset = scaler.fit_transform(dataset)
# #%%

# # split into train and test sets
# train_size = int(len(dataset) * 0.67)
# test_size = len(dataset) - train_size
# train, test = dataset[0:train_size, :], dataset[train_size : len(dataset), :]
# print(len(train), len(test))
# #%%


# def create_dataset(dataset, look_back=1):
#     dataX, dataY = [], []
#     for i in range(len(dataset) - look_back - 1):
#         a = dataset[i : (i + look_back), 0]
#         dataX.append(a)
#         dataY.append(dataset[i + look_back, 0])
#     return numpy.array(dataX), numpy.array(dataY)


# #%%
# look_back = 1
# trainX, trainY = create_dataset(train, look_back)
# testX, testY = create_dataset(test, look_back)
# # reshape input to be [samples, time steps, features]
# trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
# testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# with tf.device("/GPU:0"):
#     model = Sequential()
#     model.add(LSTM(4, input_shape=(1, look_back)))
#     model.add(Dense(1))
#     model.compile(loss="mean_squared_error", optimizer="adam")
#     model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
# #%%
# # make predictions
# trainPredict = model.predict(trainX)
# testPredict = model.predict(testX)
# # invert predictions
# trainPredict = scaler.inverse_transform(trainPredict)
# trainY = scaler.inverse_transform([trainY])
# testPredict = scaler.inverse_transform(testPredict)
# testY = scaler.inverse_transform([testY])
# # calculate root mean squared error
# trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
# testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))

# # shift train predictions for plotting
# trainPredictPlot = numpy.empty_like(dataset)
# trainPredictPlot[:, :] = numpy.nan
# trainPredictPlot[look_back : len(trainPredict) + look_back, :] = trainPredict
# # shift test predictions for plotting
# testPredictPlot = numpy.empty_like(dataset)
# testPredictPlot[:, :] = numpy.nan
# testPredictPlot[
#     len(trainPredict) + (look_back * 2) + 1 : len(dataset) - 1, :
# ] = testPredict
# # plot baseline and predictions
# plt.plot(scaler.inverse_transform(dataset))
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
# plt.show()

# %%
def principal_payment(loan_amount, monthly_payment, interest_rate):
    # Calculate the monthly interest rate
    monthly_interest_rate = interest_rate / 100 / 12

    # Calculate the principal payment
    principal_payment = monthly_payment - (monthly_interest_rate * loan_amount)

    return principal_payment


principal_payment(810_000, 3700, 5.40)
