"""
Note: do not tweak this code unless you know what you're doing.'
"""

import tensorflow as tf
import json
from init import main


with open('config.json') as f:
	_d = json.load(f)
	labels = _d['labels']
	model = _d['CurrentModel']
model = tf.keras.models.load_model(f'models/{model}')
if __name__ == '__main__':
	main('neuralModel', labels, model=model)
