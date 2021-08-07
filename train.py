"""
Note: do not tweak this code unless you know what you're doing.'
"""


from os import sep
import tensorflow as tf
from sklearn.model_selection import train_test_split
from pandas import read_csv
import numpy as np
import json

with open('config.json') as f:
   config = json.load(f)
labels = config['labels']
data_version = config['DataVersion']
model_version = config['ModelVersion']

data = read_csv(f'data/data-v{round(data_version-.1, 1)}.csv', sep=",")
y = np.array(data.iloc[:, -1])
X = np.array(data.iloc[:, 0:-1])
# print(np.where(X == 'thumbs-up'))
X = X.astype(np.float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(X_train.shape[0]))
model.add(tf.keras.layers.Dense(128, activation='relu', name='middlelayer'))
model.add(tf.keras.layers.Dense(len(labels), activation='softmax', name='outputlayer'))


model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=110)
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Loss: {test_loss}\naccuracy: {test_acc}")



model.save(f'models/model-v{model_version}.h5')
with open('config.json', 'w') as f:
   json.dump(config, f, indent=2)