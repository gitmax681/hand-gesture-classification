from os import sep
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from pandas import read_csv
import numpy
import mediapipe as mp
import cv2
import math

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


data = read_csv('data/data.csv', sep=',')
y = data.iloc[:, -1]
X = numpy.array(data.iloc[:, 0:-1])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))



def returnDistance(x1, x2, y1, y2):
    d = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return d


cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                dd = []
                prev = 0
                for hand in range(21):
                    normalizedLandmark = hand_landmarks.landmark[hand]
                    pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(
                        normalizedLandmark.x, normalizedLandmark.y, 900, 900)
                    if type(prev) != tuple and prev == 0 and type(pixelCoordinatesLandmark) != None:
                        prev = (pixelCoordinatesLandmark, hand)
                        continue
                    else:
                        if pixelCoordinatesLandmark != None:
                            for co in range(21):
                                if prev[1] != co:
                                    comark = hand_landmarks.landmark[co]
                                    pixelcomark = mp_drawing._normalized_to_pixel_coordinates(
                                        comark.x, comark.y, 900, 900)
                                    dd.append(returnDistance(
                                        prev[0][0], pixelcomark[0], prev[0][1], pixelcomark[1]))

                            prev = (pixelCoordinatesLandmark, hand)

                print(model.predict([numpy.array(dd)]))

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
