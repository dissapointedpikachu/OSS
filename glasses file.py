import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

def apply_glasses_filter(frame, landmarks, glasses_filter):
    left_eye = landmarks[33]
    right_eye = landmarks[263]

    eye_center_x = int((left_eye.x + right_eye.x) * frame.shape[1] / 2)
    eye_center_y = int((left_eye.y + right_eye.y) * frame.shape[0] / 2)

    glasses_width = 200
    glasses_height = 100 

    glasses_resized = cv2.resize(glasses_filter, (glasses_width, glasses_height))

    glasses_x = int(eye_center_x - glasses_width / 2)
    glasses_y = int(eye_center_y - glasses_height / 2 + 30)

    glasses_x = max(0, min(glasses_x, frame.shape[1] - glasses_width))
    glasses_y = max(0, min(glasses_y, frame.shape[0] - glasses_height))

    roi = frame[glasses_y:glasses_y+glasses_height, glasses_x:glasses_x+glasses_width]
    alpha_glasses = glasses_resized[:, :, 3] / 255.0
    alpha_frame = 1.0 - alpha_glasses

    for c in range(3):
        roi[:, :, c] = (alpha_glasses * glasses_resized[:, :, c] +
                         alpha_frame * roi[:, :, c])

    frame[glasses_y:glasses_y+glasses_height, glasses_x:glasses_x+glasses_width] = roi
    
    return frame

cap = cv2.VideoCapture(0)
glasses_filter = cv2.imread('glasses_filter.png', cv2.IMREAD_UNCHANGED)

if glasses_filter is None:
    print("Can't find glasses_filter.png.")
    exit()

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't open webcam.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                frame = apply_glasses_filter(frame, landmarks.landmark, glasses_filter)

        cv2.imshow('Glasses Filter', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
