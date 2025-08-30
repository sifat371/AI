from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt


def build_model():
    inputs = Input((1,))
    h0 = Dense(8, activation='relu')(inputs)
    h1 = Dense(16, activation='relu')(h0) 
    h2 = Dense(32, activation='relu')(h1)
    h3 = Dense(8, activation='relu')(h2)
    outputs = Dense(1)(h3)

    model = Model(inputs, outputs)
    model.summary(show_trainable=True)

    return model


def my_polynomial(x):
    y = 5 * x**2 + 10 * x - 2
    return y


def data_process(n=100000):
    x = np.random.uniform(low=-10, high=10, size=n) 
    y = my_polynomial(x)
    return x, y


def prepare_train_val_test():
    x, y = data_process()
    total_n = len(x)
    print(x.shape, total_n)

    indices = np.random.permutation(total_n)
    x, y = x[indices], y[indices]

    train_n = int(total_n * 0.7)
    val_n = int(total_n * 0.1)
    test_n = total_n - train_n - val_n

    trainX = x[: train_n]
    trainY = y[: train_n]
    valX = x[train_n: train_n + val_n]
    valY = y[train_n: train_n + val_n]
    testX = x[train_n + val_n:]
    testY = y[train_n + val_n:]
    print('total_n: {}, train_n: {}, val_n: {}, test_n: {}'.format(len(x), len(trainX), len(valX), len(testX)))

    return (trainX, trainY), (valX, valY), (testX, testY)


def main():
    #--- Build model
    model = build_model()
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    #--- Prepare data
    (trainX, trainY), (valX, valY), (testX, testY) = prepare_train_val_test()

    #--- Train model
    history = model.fit(trainX, trainY, validation_data=(valX, valY), epochs=10)

    #--- Predict
    y_pred = model.predict(testX)

    #--- Plot
    plt.scatter(testX, testY, label="True", alpha=0.6)
    plt.scatter(testX, y_pred, label="Predicted", alpha=0.6)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
