from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.models import Model

inputs=Input((2,))
h1=Dense(4,activation='relu')(inputs)
outputs=Dense(1,activation='softmax')(h1)
model=Model(inputs,outputs)
model.summary(show_trainable=True)
