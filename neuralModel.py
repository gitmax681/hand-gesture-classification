import tensorflow as tf
import numpy as np
import mediapipe as mp
import cv2
import math
import json

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
model = tf.keras.models.load_model('models/model-v1.1.h5')


def returnDistance(x1, x2, y1, y2):
	d = math.sqrt((x2-x1)**2 + (y2-y1)**2)
	return d

with open('config.json') as f:
	labels = json.load(f)['labels']
try: 
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
									try:
										if prev[1] != co:
											comark = hand_landmarks.landmark[co]
											pixelcomark = mp_drawing._normalized_to_pixel_coordinates(
												comark.x, comark.y, 900, 900)
											dd.append(returnDistance(
												prev[0][0], pixelcomark[0], prev[0][1], pixelcomark[1]))
									except TypeError:
										continue
								prev = (pixelCoordinatesLandmark, hand)
					if len(dd) < 400:
						continue
					X = np.array(dd).astype(np.float)
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
					X = X.reshape(1, 1, 400)
					prediction = model.predict(X)
					cv2.putText(image, labels[np.argmax(
						prediction)], (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

			cv2.imshow('MediaPipe Hands', image)
			if cv2.waitKey(5) & 0xFF == 27:
				break
	cap.release()
except KeyboardInterrupt:
    print('\nExiting.....')