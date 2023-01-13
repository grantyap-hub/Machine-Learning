import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("healthcare-dataset-stroke-data.csv")
data = data.drop(['id'], axis=1)
x_vals = pd.get_dummies(data, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])
y_vals = data['stroke']

model = keras.Sequential(
    [
        layers.Normalization(axis=-1, mean=0, variance=1, invert=False),
        layers.Dense(50, input_shape=(13,), activation='relu', name="hidden_layer"),
        layers.Dense(1, activation='sigmoid', name="output_layer")
    ]
)


model.compile(

    optimizer=keras.optimizers.SGD(),
    loss='binary_crossentropy',
    metrics='accuracy',
)

result = model.fit(
    x_vals,
    y_vals,
    batch_size=64,
    epochs=100,
    validation_split=0.2
)

testing = model.evaluate(x_vals, y_vals)
print(testing)
