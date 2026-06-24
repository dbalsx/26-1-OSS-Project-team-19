import cv2
import time
import pyttsx3
import pygame
import threading
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="A-eye Voice Guidance", layout="wide")
st.title("🚶‍♂️ A-eye Voice Guidance System 👀")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

pygame.mixer.init()
try:
    warning_sound = pygame.mixer.Sound("beep.wav")
except:
    warning_sound = None

audio_state = {"is_speaking": False}

def speak_async(text, play_warning=False):
    def task():
        audio_state["is_speaking"] = True
        
        if play_warning and warning_sound:
            warning_sound.play()
            time.sleep(0.5)
            
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 160)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            pass
        finally:
            audio_state["is_speaking"] = False

    if not audio_state["is_speaking"]:
        threading.Thread(target=task, daemon=True).start()

korean_name = {
    "person": "사람", "car": "자동차", 
    "moto cycle": "오토바이", "obs": "장애물"
}

col1, col2 = st.columns([3, 1])

with col1:
    stframe = st.empty()

with col2:
   
    video_options = {
        "학교 주차장 시연": "test_video1.mp4",
        "명동 거리 시연": "test_video2.mp4"
    }
    
    
    selected_video_name = st.selectbox("📁 시연 영상 선택", list(video_options.keys()))
    selected_video_path = video_options[selected_video_name]
    
    start_btn = st.button("🚀 시연 영상 시작", use_container_width=True)
    stop_btn = st.button("🛑 중지", use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 실시간 안내 로그")
    log_text = st.empty()

if start_btn:
   
    cap = cv2.VideoCapture(selected_video_path)
    
    global_last_spoken = 0
    COOLDOWN = 4.0
    
    frame_count = 0
    frame_skip = 3 
    
    log_text.info(f"'{selected_video_name}' 영상을 분석 중입니다...")

    while cap.isOpened():
        if stop_btn: 
            st.warning("분석이 중지되었습니다.")
            break
            
        success, frame = cap.read()
        if not success:
            st.success("영상 재생이 완료되었습니다.")
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        height, width, _ = frame.shape
        results = model(frame, conf=0.5)
        annotated_frame = results[0].plot()

        most_danger_box = None
        max_y2 = 0 

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if y2 > max_y2:
                max_y2 = y2
                most_danger_box = box

        current_time = time.time()

        if (current_time - global_last_spoken > COOLDOWN) and not audio_state["is_speaking"]:
            if most_danger_box is not None:
                x1, y1, x2, y2 = map(int, most_danger_box.xyxy[0])
                x_center = (x1 + x2) / 2
                
                cls_id = int(most_danger_box.cls[0])
                class_name = model.names[cls_id]
                object_name = korean_name.get(class_name, class_name)

                if x_center < width * 0.33: direction = "11시 방향"
                elif x_center < width * 0.66: direction = "정면"
                else: direction = "1시 방향"

                if y2 > (height * 0.7):
                    msg = f"위험. {direction} {object_name}."
                    log_text.error(f"🔴 [위험] {msg}")
                    speak_async(msg, play_warning=True)
                    global_last_spoken = current_time
                    
                else:
                    msg = f"{direction}에 {object_name}."
                    log_text.warning(f"🟡 [주의] {msg}")
                    speak_async(msg, play_warning=False)
                    global_last_spoken = current_time

        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        stframe.image(rgb_frame, channels="RGB", use_container_width=True)

    cap.release()
