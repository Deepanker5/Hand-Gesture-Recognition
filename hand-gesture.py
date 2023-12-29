import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#get webcam to work
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence = 0.9, min_tracking_confidence = 0.9) as pose:

    while cap.isOpened():
        ret , frame = cap.read()

        #different color
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #render detection 
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)


        cv2.imshow("mediapipe feed", image)
        if cv2.waitKey(10) & 0xFF ==ord('q'):
            break
        

cap.release()
cv2.destroyAllWindows()