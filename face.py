import cv2
import numpy as np
import mediapipe as mp

# Mediapipe Face Mesh 초기화
mp_face_mesh = mp.solutions.face_mesh

# 강아지 필터 적용 함수
def apply_dog_filter(frame, landmarks, dog_filter):
    # 코 끝 위치 (랜드마크 번호 1)
    nose_tip = landmarks[1]  # 코 끝 (랜드마크 번호 1)

    # 필터 크기 설정 (기준: 필터 크기 고정)
    filter_height = 100  # 필터 높이 고정
    filter_width = 100   # 필터 너비 고정
    filter_resized = cv2.resize(dog_filter, (filter_width, filter_height))

    # 필터의 하단이 코 끝 위치에 맞도록 필터 위치 조정
    filter_x = int(nose_tip.x * frame.shape[1] - filter_width / 2)  # 코 X 위치에 맞추어 중앙 배치
    filter_y = int(nose_tip.y * frame.shape[0] - filter_height)    # 코 Y 위치에서 필터 높이만큼 빼기 (하단 맞추기)

    # 필터가 화면을 벗어나지 않도록 조
    filter_x = max(0, min(filter_x, frame.shape[1] - filter_width))
    filter_y = max(0, min(filter_y, frame.shape[0] - filter_height))

    # 필터 적용 (하단이 코 위치에 맞춰짐)
    roi = frame[filter_y:filter_y+filter_height, filter_x:filter_x+filter_width]
    alpha_filter = filter_resized[:, :, 3] / 255.0  # alpha 채널 (투명도)
    alpha_frame = 1.0 - alpha_filter

    for c in range(3):  # RGB 채널 처리
        roi[:, :, c] = (alpha_filter * filter_resized[:, :, c] +
                         alpha_frame * roi[:, :, c])

    frame[filter_y:filter_y+filter_height, filter_x:filter_x+filter_width] = roi
    
    return frame

# 웹캠 열기
cap = cv2.VideoCapture(0)
dog_filter = cv2.imread('dog_filter2.png', cv2.IMREAD_UNCHANGED)

if dog_filter is None:
    print("dog_filter.png 파일을 찾을 수 없습니다. 필터 이미지를 같은 디렉토리에 두세요.")
    exit()

# Mediapipe Face Mesh 초기화
with mp.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("카메라에서 영상을 가져올 수 없습니다.")
            break

        frame = cv2.flip(frame, 1)  # 좌우 반전
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                frame = apply_dog_filter(frame, landmarks.landmark, dog_filter)

        # 실시간 이미지 표시
        cv2.imshow('Dog Filter', frame)

        # ESC 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
