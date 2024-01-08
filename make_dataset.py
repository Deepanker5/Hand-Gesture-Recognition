import os
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = "data"

for directory in os.listdir(DATA_DIR):
    directory_path = os.path.join(DATA_DIR, directory)

    # Check if it's a directory before processing
    if os.path.isdir(directory_path):
        for image_path in os.listdir(directory_path)[:1]:
            image = cv2.imread(os.path.join(directory_path, image_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            res = hands.process(image_rgb)

            if res.multi_hand_landmarks:
                for hnd_lndmrks in res.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image_rgb, hnd_lndmrks, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing_styles.get_default_hand_landmarks_style(),
                                              mp_drawing_styles.get_default_hand_connections_style())

            plt.figure()
            plt.imshow(image_rgb)

plt.show()