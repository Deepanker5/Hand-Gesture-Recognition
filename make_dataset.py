import os
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import pickle

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = "data"

data = [] #pictures of hand
letters = [] #category of sign

for directory in os.listdir(DATA_DIR):
    directory_path = os.path.join(DATA_DIR, directory)

    # Check if it's a directory before processing
    if os.path.isdir(directory_path):
        for image_path in os.listdir(directory_path):


            image = cv2.imread(os.path.join(directory_path, image_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            res = hands.process(image_rgb)
            data_auxilary = []
            if res.multi_hand_landmarks:
                for hnd_lndmrks in res.multi_hand_landmarks:
                    for i in range(len(hnd_lndmrks.landmark)):
                        x_hnd = hnd_lndmrks.landmark[i].x
                        y_hnd = hnd_lndmrks.landmark[i].y
                        data_auxilary.append(x_hnd)
                        data_auxilary.append(y_hnd)

                data.append(data_auxilary)
                letters.append(directory)

file = open('training_data.pickle', 'wb')
pickle.dump({'data': data, 'letters': letters}, file)
file.close



plt.show()