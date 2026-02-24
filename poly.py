from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt


def polimonial(x):
    return 7*x**4 - 4*x**3 - x + 6

def data_processing():
    n = 10000

    x = np.random.uniform(-15, 15, n)
    y = polimonial(x)

    x_min, x_max = x.min(), x.max()
    #x_scaled = (x - x_min)/(x_max - x_min)

    y_min, y_max = y.min(), y.max()
    #y_scaled = (y - y_min)/(y_max - y_min)

    x_scaled = 2 * (x - x_min) / (x_max - x_min) - 1
    y_scaled = 2 * (y - y_min) / (y_max - y_min) - 1

    return x_scaled, y_scaled, x_max, y_max, x_min, y_min

def data_prepatation():
    x_scaled, y_scaled, x_max, y_max, x_min, y_min = data_processing()
    total_n = len(x_scaled)
    train_n = int(total_n * 0.8)
    val_n = int(total_n* 0.1)

    train_x = x_scaled[:train_n].reshape(-1, 1)
    train_y = y_scaled[:train_n]

    val_x = x_scaled[train_n: train_n+val_n].reshape(-1,1)
    val_y = y_scaled[train_n: train_n+val_n]

    test_x = x_scaled[train_n+val_n:].reshape(-1,1)
    test_y = y_scaled[train_n+val_n:]

    return (train_x, train_y), (val_x, val_y), (test_x, test_y)

def build_model():
    inputs = Input((1,))
    h1 = Dense(32, activation="relu")(inputs)
    h2 = Dense(64, activation="relu")(h1)
    h3 = Dense(128, activation="relu")(h2)
    outputs = Dense(1)(h3)
    model = Model(inputs, outputs)
    model.summary(show_trainable = True)
    return model

def main():
    model = build_model()
    model.compile(loss = 'mse', optimizer = Adam(learning_rate = 1e-4))

    (train_x, train_y), (val_x, val_y), (test_x, test_y) = data_prepatation()
    
    history = model.fit(
        train_x,
        train_y,
        validation_data = (val_x, val_y),
        epochs = 10,
        verbose = 1,
        batch_size = 32
    )
    y_test_prediction = model.predict(test_x)

    # # plt.plot(test_x, y_test_prediction, color = 'red', label = 'predict')
    # # plt.plot(test_y, test_y, color = 'blue', label = 'ideal')
    plt.scatter(test_x, test_y, label="True")
    plt.scatter(test_x, y_test_prediction, label="Predicted")

    # model.evaluate(test_x, test_y)
    plt.grid()
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='Train MSE')
    plt.plot(history.history['val_loss'], label='Val MSE')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
