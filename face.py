import cv2
import numpy as np
import mediapipe as mp

# Mediapipe Face Detection 초기화
mp_face_detection = mp.solutions.face_detection

# 놀란 표정 감지 및 강아지 필터 적용
def apply_dog_filter(frame, faces, dog_filter):
    for face in faces:
        bbox = face.location_data.relative_bounding_box
        ih, iw, _ = frame.shape
        x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)

        # 강아지 필터 크기 조정
        resized_filter = cv2.resize(dog_filter, (w, h))

        # 필터를 적용할 ROI
        roi = frame[y:y+h, x:x+w]

        # 필터 적용 (투명도 유지)
        alpha_filter = resized_filter[:, :, 3] / 255.0
        alpha_frame = 1.0 - alpha_filter

        for c in range(3):  # RGB 채널 적용
            roi[:, :, c] = (alpha_filter * resized_filter[:, :, c] +
                            alpha_frame * roi[:, :, c])

        frame[y:y+h, x:x+w] = roi
    return frame

# 웹캠 열기
cap = cv2.VideoCapture(0)  # 로컬 환경에서 웹캠 사용
dog_filter = cv2.imread('dog_filter.png', cv2.IMREAD_UNCHANGED)  # 투명 채널 포함된 필터

if dog_filter is None:
    print("dog_filter.png 파일을 찾을 수 없습니다. 필터 이미지를 같은 디렉토리에 두세요.")
    exit()

# Mediapipe Face Detection 초기화
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("카메라에서 영상을 가져올 수 없습니다.")
            break

        frame = cv2.flip(frame, 1)  # 좌우 반전
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            frame = apply_dog_filter(frame, results.detections, dog_filter)

        # 로컬 환경에서 실시간 이미지 표시
        cv2.imshow('Dog Filter', frame)

        # ESC 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == 27:  # ESC 키의 ASCII 값은 27
            break

cap.release()
cv2.destroyAllWindows()

#