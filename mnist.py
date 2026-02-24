from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.datasets import fashion_mnist
import numpy as np

(trainX, trainY), (testX, testY) = fashion_mnist.load_data()

trainX = trainX.astype('float32') / 255.0
testX = testX.astype('float32') / 255.0

trainX = trainX.reshape(-1, 28, 28, 1)
testX = testX.reshape(-1, 28, 28, 1)

trainY = to_categorical(trainY, 10)
testY = to_categorical(testY, 10)

total = len(trainX)
train_size = int(0.85 * total)

x_train = trainX[:train_size]
y_train = trainY[:train_size]

x_val = trainX[train_size:]
y_val = trainY[train_size:]


inputs = Input((28, 28, 1))

x = Conv2D(16, (3,3), activation='relu', name='conv1')(inputs)
x = Conv2D(32, (3,3), activation='relu', name='conv2')(x)
x = Conv2D(64, (3,3), activation='relu', name='conv3')(x)

x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dense(32, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)   # ⭐ 10 classes

model = Model(inputs, outputs)
model.summary(show_trainable=True)

checkpoint = ModelCheckpoint(
    'best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)


model.compile(optimizer=Adagrad(learning_rate=0.003),
              loss="categorical_crossentropy",
              metrics=['accuracy'])

history1 = model.fit(x_train, y_train,
          validation_data=(x_val, y_val),
          epochs=2,
          batch_size=32,
          callbacks=[checkpoint])

for layer in model.layers:
    if 'conv' in layer.name:
        layer.trainable = False

model.compile(optimizer=Adagrad(learning_rate=0.003),
              loss="categorical_crossentropy",
              metrics=['accuracy'])

history2 = model.fit(x_train, y_train,
          validation_data=(x_val, y_val),
          epochs=3,
          batch_size=32,
          callbacks=[checkpoint])

test_loss, test_acc = model.evaluate(testX, testY)
print("Test Accuracy :", test_acc)
print("Test Loss :", test_loss)


loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']

plt.plot(loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.legend()
plt.grid()
plt.show()
