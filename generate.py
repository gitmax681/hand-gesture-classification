import cv2
import mediapipe as mp
import math
import os
import json
import tensorflow as tf
import numpy as np


with open('config.json') as file:
    config = json.load(file)
labels = config['labels']
data_version = config['DataVersion']
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def returnDistance(co):  # euclidean distance
    return math.sqrt((co[0]-co[1])**2 + (co[2]-co[3])**2)


def scaleValue(X):
    X = np.array(tf.math.divide(
        tf.math.subtract(
            X,
            tf.math.reduce_min(X)
        ),
        tf.subtract(
            tf.math.reduce_max(X),
            tf.math.reduce_min(X)
        )
    ))
    return list(X)


def run(name):
    firstTime = True
    cap = cv2.VideoCapture(0)
    samples = config['MaxSamples']  # maximum samples for a class
    with mp_hands.Hands(
            min_detection_confidence=0.65,
            min_tracking_confidence=0.8) as hands:
        while cap.isOpened() and samples >= 0:
            success, image = cap.read()
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    dictt = []
                    prev = 0
                    for hand in range(21):
                        normalizedLandmark = hand_landmarks.landmark[hand]
                        pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(
                            normalizedLandmark.x, normalizedLandmark.y, 1000, 1000)
                        if type(prev) != tuple and prev == 0 and type(pixelCoordinatesLandmark) != None:
                            prev = (pixelCoordinatesLandmark, hand)
                            continue
                        else:
                            if pixelCoordinatesLandmark != None:
                                for co in range(21):
                                    try:
                                        if prev[1] != co:
                                            comark = hand_landmarks.landmark[co]
                                            pixelcomark = mp_drawing._normalized_to_pixel_coordinates(
                                                comark.x, comark.y, 1000, 1000)

                                            coordinates = (
                                                prev[0][0], pixelcomark[0], prev[0][1], pixelcomark[1])
                                            dictt.append(
                                                returnDistance(coordinates))

                                    except TypeError as e:
                                        # print(f"Landmark singles not found.: {e}")
                                        pass
                                prev = (pixelCoordinatesLandmark, hand)
                    print(
                        f"{config['MaxSamples']-samples}/{config['MaxSamples']}", end='\r')
                    samples -= 1
                    with open(f'data/data-v{round(data_version, 1)}.csv', 'a+') as f:
                        bt = '\n'
                        if firstTime:
                            try:
                                with open(f'data/data-v{round(data_version-.1, 1)}.csv', 'r') as prior:
                                    bt+=prior.read() + "\n"
                                    firstTime = False
                            except FileNotFoundError:
                                print("PriorVersionNotFound: previous version of data was not found")        
                                firstTime = False                            
                        dictt = scaleValue(dictt)
                        for _ in dictt:
                            bt += f"{_},"
                        bt += f"{name},"
                        bt += str(labels.index(name))
                        f.write(bt)
            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()


if __name__ == '__main__':
    name = input('Enter the Name of Gesture: ')
    config['labels'].append(name)
    config['DataVersion'] += .1
    config["CurrentIndex"] += 1

    with open('config.json', 'w') as f:
        json.dump(config, f, indent=2)
    run(name)
