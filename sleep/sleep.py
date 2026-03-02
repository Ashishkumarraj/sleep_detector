import cv2
import numpy as np
import time
import pygame
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================
# AUDIO INIT
# ==========================
pygame.mixer.init()
pygame.mixer.music.load("alert.mp3")

# ==========================
# LOAD MEDIAPIPE MODEL
# ==========================
base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)

# ==========================
# EAR FUNCTION
# ==========================
def calculate_EAR(eye_landmarks):
    A = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
    B = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
    C = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
    return (A + B) / (2.0 * C)

# Eye landmark indices (MediaPipe 468 model)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.23
CLOSED_TIME = 1.0

cap = cv2.VideoCapture(0)

closed_start = None
is_playing = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_image)

    if result.face_landmarks:
        landmarks = result.face_landmarks[0]
        h, w, _ = frame.shape

        left_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE_IDX]
        right_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE_IDX]

        left_ear = calculate_EAR(left_eye)
        right_ear = calculate_EAR(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Draw eye landmarks
        for (x, y) in left_eye + right_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Drowsiness logic
        if ear < EAR_THRESHOLD:
            if closed_start is None:
                closed_start = time.time()
            elif time.time() - closed_start >= CLOSED_TIME:
                if not is_playing:
                    pygame.mixer.music.play(-1)
                    is_playing = True
        else:
            closed_start = None
            if is_playing:
                pygame.mixer.music.stop()
                is_playing = False

        cv2.putText(frame, f"EAR: {ear:.2f}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Real Driver Drowsiness Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
pygame.mixer.music.stop()
cv2.destroyAllWindows()