import cv2
import numpy as np
import mediapipe as mp

# Mediapipe Face Mesh 초기화
mp_face_mesh = mp.solutions.face_mesh

# 고정된 안경 필터 적용 함수
def apply_glasses_filter(frame, landmarks, glasses_filter):
    # 왼쪽 눈과 오른쪽 눈의 위치 (랜드마크 번호 33, 263)
    left_eye = landmarks[33]
    right_eye = landmarks[263]

    # 두 눈 사이의 중앙 계산
    eye_center_x = int((left_eye.x + right_eye.x) * frame.shape[1] / 2)
    eye_center_y = int((left_eye.y + right_eye.y) * frame.shape[0] / 2)

    # 안경 크기 고정 (예시로 고정된 크기 설정)
    glasses_width = 200  # 고정된 안경 너비
    glasses_height = 100  # 고정된 안경 높이

    # 안경 이미지 크기 조정
    glasses_resized = cv2.resize(glasses_filter, (glasses_width, glasses_height))

    # 안경의 위치 설정 (두 눈의 중앙을 기준으로 배치)
    glasses_x = int(eye_center_x - glasses_width / 2)
    glasses_y = int(eye_center_y - glasses_height / 2 + 30)

    # 안경이 화면을 벗어나지 않도록 조정
    glasses_x = max(0, min(glasses_x, frame.shape[1] - glasses_width))
    glasses_y = max(0, min(glasses_y, frame.shape[0] - glasses_height))

    # 안경 적용
    roi = frame[glasses_y:glasses_y+glasses_height, glasses_x:glasses_x+glasses_width]
    alpha_glasses = glasses_resized[:, :, 3] / 255.0  # alpha 채널 (투명도)
    alpha_frame = 1.0 - alpha_glasses

    for c in range(3):  # RGB 채널 처리
        roi[:, :, c] = (alpha_glasses * glasses_resized[:, :, c] +
                         alpha_frame * roi[:, :, c])

    frame[glasses_y:glasses_y+glasses_height, glasses_x:glasses_x+glasses_width] = roi
    
    return frame

# 웹캠 열기
cap = cv2.VideoCapture(0)
glasses_filter = cv2.imread('glasses_filter.png', cv2.IMREAD_UNCHANGED)

if glasses_filter is None:
    print("glasses_filter.png 파일을 찾을 수 없습니다. 필터 이미지를 같은 디렉토리에 두세요.")
    exit()

# Mediapipe Face Mesh 초기화
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
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
                frame = apply_glasses_filter(frame, landmarks.landmark, glasses_filter)

        # 실시간 이미지 표시
        cv2.imshow('Glasses Filter', frame)

        # ESC 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
