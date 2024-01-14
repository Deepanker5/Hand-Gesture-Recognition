import cv2
import mediapipe as mp
import pickle
import numpy as np
import pygame

rnd_forest_model = pickle.load(open('./chosen_model.p', 'rb'))
model = rnd_forest_model['chosen_model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

letter_translation = {0: 'A', 1: 'B', 2: 'L'}

capture = cv2.VideoCapture(0)

pygame.init()

width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Hand Tracking")

font = pygame.font.SysFont(None, 55)
sentence_font = pygame.font.SysFont(None, 30)

sentence = ""
last_predicted_letter = None

while True:
    ret, frame = capture.read()
    data_auxilary = []

    x_ = []
    y_ = []
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(frame_rgb)

    if res.multi_hand_landmarks:
        for hnd_lndmarks in res.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hnd_lndmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style())

        for hnd_lndmrks in res.multi_hand_landmarks:
            for i in range(len(hnd_lndmrks.landmark)):
                x_hnd = hnd_lndmrks.landmark[i].x
                y_hnd = hnd_lndmrks.landmark[i].y
                data_auxilary.append(x_hnd)
                data_auxilary.append(y_hnd)
                x_.append(x_hnd)
                y_.append(y_hnd)
        letter_prediction = model.predict([np.asarray(data_auxilary)])
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        predicted_letter = letter_translation[int(letter_prediction[0])]

        if predicted_letter != last_predicted_letter:
            sentence += predicted_letter
            last_predicted_letter = predicted_letter
            if len(sentence) > 30:
                sentence = sentence[1:]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_letter, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        frame = pygame.surfarray.make_surface(frame)
        screen.blit(frame, (0, 0))

        text = font.render("Predicted Letter: " + predicted_letter, True, (255, 255, 255))
        screen.blit(text, (20, 20))

        sentence_text = sentence_font.render("Sentence: " + sentence, True, (255, 255, 255))
        screen.blit(sentence_text, (20, height - 50))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                cv2.destroyAllWindows()
                exit()
    pygame.display.flip()
    
    

