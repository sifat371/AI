import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt

# ===============================
# Paths (CHANGE THIS)
# ===============================
DATA_PATH = "/content/mnist_png"   # <- your pendrive path

# ===============================
# Load datasets
# ===============================
train_ds = image_dataset_from_directory(
    DATA_PATH + "/train",
    image_size=(28,28),
    batch_size=32,
    color_mode="grayscale",
    label_mode="categorical",
    validation_split=0.15,
    subset="training",
    seed=42
)

val_ds = image_dataset_from_directory(
    DATA_PATH + "/train",
    image_size=(28,28),
    batch_size=32,
    color_mode="grayscale",
    label_mode="categorical",
    validation_split=0.15,
    subset="validation",
    seed=42
)

test_ds = image_dataset_from_directory(
    DATA_PATH + "/test",
    image_size=(28,28),
    batch_size=32,
    color_mode="grayscale",
    label_mode="categorical"
)

# Normalize
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x,y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x,y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x,y: (normalization_layer(x), y))

# ===============================
# Build CNN
# ===============================
inputs = Input((28,28,1))

x = Conv2D(16,(3,3),activation='relu')(inputs)
x = MaxPooling2D()(x)

x = Conv2D(32,(3,3),activation='relu')(x)
x = MaxPooling2D()(x)

x = Flatten()(x)
x = Dense(128,activation='relu')(x)
outputs = Dense(10,activation='softmax')(x)

model = Model(inputs, outputs)
model.summary()

# ===============================
# Checkpoint
# ===============================
checkpoint = ModelCheckpoint(
    "best_model.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

# ===============================
# Compile
# ===============================
model.compile(
    optimizer=Adagrad(0.003),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ===============================
# Train
# ===============================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=8,
    callbacks=[checkpoint]
)

# ===============================
# Test
# ===============================
test_loss, test_acc = model.evaluate(test_ds)
print("Test accuracy:", test_acc)

# ===============================
# Plot
# ===============================
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.grid()
plt.show()
