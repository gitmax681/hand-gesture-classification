"""
Note: do not tweak this code unless you know what you're doing.'
"""


import tensorflow as tf
import numpy as np
import math
import mediapipe as mp
import json

_drawing = mp.solutions.drawing_utils
_hand = mp.solutions.hands
_max_samples = 0
_first_time = True

def returnDistance(co):  # euclidean distance
    return math.sqrt((co[0]-co[1])**2 + (co[2]-co[3])**2)

def scaleValue(X):
    X = tf.math.divide(
        tf.math.subtract(
            X,
            tf.math.reduce_min(X)
        ),
        tf.subtract(
            tf.math.reduce_max(X),
            tf.math.reduce_min(X)
        )
    )
    return X


def createVector(image, results):
    for hand_landmarks in results.multi_hand_landmarks:
        _drawing.draw_landmarks(
            image, hand_landmarks, _hand.HAND_CONNECTIONS)

        dd = []
        prev = 0
        for hand in range(21):
            normalizedLandmark = hand_landmarks.landmark[hand]
            pixelCoordinatesLandmark = _drawing._normalized_to_pixel_coordinates(
                normalizedLandmark.x, normalizedLandmark.y, 900, 900)
            if type(prev) != tuple and prev == 0 and type(pixelCoordinatesLandmark) != None:
                prev = (pixelCoordinatesLandmark, hand)
                continue
            else:
                if pixelCoordinatesLandmark != None:
                    for co in range(21):
                        if prev[1] != co:
                            try:
                                comark = hand_landmarks.landmark[co]
                                pixelcomark = _drawing._normalized_to_pixel_coordinates(
                                    comark.x, comark.y, 900, 900)
                                dd.append(returnDistance(
                                    (prev[0][0], pixelcomark[0], prev[0][1], pixelcomark[1])))
                            except TypeError:
                                continue

                    prev = (pixelCoordinatesLandmark, hand)
        return np.array(scaleValue(dd))


def knnModel(model, X, labels):
    try: 
        return labels[model.predict([X])[0]]
    except ValueError:
        return ""

def neuralModel(model, X, labels):
    if len(X) <= 400:
        try:
            X = np.array(X).astype(np.float)
            X = X.reshape(1, 1, 400)
            prediction = np.argmax(model.predict(X)[0])
            return labels[prediction]
        except ValueError:
            return ''
        
def Generate(name, vectors, _):
    global _first_time, _max_samples
    with open('config.json') as f:
        _info = json.load(f)
        data_version= _info['DataVersion']
        labels = _info['labels']
        max_samples = _info['MaxSamples']
        
    with open(f'data/data-v{round(data_version, 1)}.csv', 'a+') as f:
        bt = '\n'
        if _first_time:
            try:
                _max_samples = max_samples
                with open(f'data/data-v{round(data_version-.1, 1)}.csv', 'r') as prior:
                    bt+=prior.read() + "\n"
                    print(f"Prior Version data-v{round(data_version-.1, 1)}.csv was found")
                    _first_time = False
            except FileNotFoundError:
                print("PriorVersionNotFound: previous version of data was not found\n")        
                _first_time = False                            
        for _ in vectors:
            bt += f"{_},"
        bt += f"{name},"
        bt += str(labels.index(name))
        f.write(bt)
        _max_samples -= 1
        if _max_samples <=0:
            print('New Version Of data Created')
            _info['CurrentData'] = f"data-v{round(_info['DataVersion'],1)}.csv"
            _info['DataVersion'] += .1
            _info["CurrentIndex"] += 1
            with open('config.json', 'w') as file:
                json.dump(_info, file, indent=2)
            exit()
    return f"{max_samples-_max_samples}/{max_samples}"