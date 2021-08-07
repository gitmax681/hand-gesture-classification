"""
Note: do not tweak this code unless you know what you're doing.'
"""


import cv2
import mediapipe as mp
from utlis import createVector, knnModel, neuralModel, Generate

mp_hands = mp.solutions.hands
drawing = mp.solutions.drawing_utils

def main(z, labels, model=None):
    maps = {
        'knnModel': knnModel,
        'neuralModel': neuralModel,
        'Generate': Generate
    }
    
    cap = cv2.VideoCapture(0)
    try:
        with mp_hands.Hands(
                min_detection_confidence=0.65,
                min_tracking_confidence=0.8) as hands:
            while cap.isOpened():
                success, image = cap.read()
                image.flags.writeable = False
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                results = hands.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    dd = createVector(image, results)
                    try:
                        prediction = maps[z](model, dd, labels)
                    except KeyError:
                        print("ProgrammingError: No such action found")
                    cv2.putText(image, prediction, (25, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                cv2.imshow('MediaPipe Hands', image)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
        cap.release()
    except KeyboardInterrupt:
        print("\nTerminating....")


