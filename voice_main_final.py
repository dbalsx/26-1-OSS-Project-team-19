import cv2
from ultralytics import YOLO
import pyttsx3
import pygame
import time

# ======================================
# A-eye Voice Guidance System
# ======================================

model = YOLO("best.pt")

# 경고음 초기화
pygame.mixer.init()
warning_sound = pygame.mixer.Sound("beep.wav")

# TTS 함수
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 160)

    engine.say(text)
    engine.runAndWait()

    engine.stop()

# 클래스 이름 한글 변환
korean_name = {
    "person": "사람",
    "car": "자동차",
    "moto cycle": "오토바이",
    "obs": "장애물"
}

# 중복 음성 방지
last_spoken = {}
COOLDOWN = 3

# ======================================
# 웹캠 / 영상 선택
# ======================================

USE_WEBCAM = True

if USE_WEBCAM:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture("test_video.mp4")

# ======================================
# 메인 루프
# ======================================

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break

    height, width, _ = frame.shape

    results = model(frame)

    annotated_frame = results[0].plot()

    for box in results[0].boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        x_center = (x1 + x2) / 2

        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]

        conf = float(box.conf[0])

        # 신뢰도 기준
        if conf < 0.25:
            continue

        # 방향 계산
        if x_center < width * 0.33:
            direction = "11시 방향"
        elif x_center < width * 0.66:
            direction = "정면"
        else:
            direction = "1시 방향"

        object_name = korean_name.get(
            class_name,
            class_name
        )

        key = f"{direction}_{object_name}"

        current_time = time.time()

        # ======================================
        # Zone1 (주의)
        # ======================================

        if y2 <= (height * 0.8):

            if (
                key not in last_spoken
                or current_time - last_spoken[key] > COOLDOWN
            ):

                message = (
                    f"{direction}에 "
                    f"{object_name}이 있습니다."
                )

                print("🟡", message)

                speak(message)

                last_spoken[key] = current_time

        # ======================================
        # Zone2 (위험)
        # ======================================

        else:

            if (
                key not in last_spoken
                or current_time - last_spoken[key] > COOLDOWN
            ):

                message = (
                    f"위험. "
                    f"{direction} "
                    f"{object_name}"
                )

                print("🔴", message)

                warning_sound.play()

                speak(message)

                last_spoken[key] = current_time

    cv2.imshow(
        "A-eye Voice Guidance System",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()