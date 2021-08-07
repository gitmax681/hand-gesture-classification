import cv2
import mediapipe as mp
import math
import os
import json
import tensorflow as tf
import numpy as np
from utlis import scaleValue, returnDistance
from returnvideo import main


with open('config.json') as file:
    config = json.load(file)
    labels = config['labels']
if __name__ == '__main__':
    name = input('Enter the Name of Gesture: ')
    with open('config.json', 'w') as f:
        labels.append(name)
        json.dump(config, f, indent=2)
    main('Generate', '',model=name)