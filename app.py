import cv2
import mediapipe as mp
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import torchvision.transforms as transforms
import os
import tkinter as tk
from tkinter import simpledialog
from enum import Enum
import torch
import numpy as np
import sounddevice as sd
from collections import deque
import time
import soundfile as sf


class VADRecorder:
    def __init__(self):
        # Load model
        self.model, self.utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                # force_reload=True,
                trust_repo=True
        )
        self.vad_iterator = self.utils[3](self.model)  # VADIterator is the 4th element

        # Settings
        self.SAMPLE_RATE = 16000
        self.BUFFER_SIZE = self.SAMPLE_RATE * 60  # 1 minute buffer
        self.THRESHOLD = 0.65
        self.MIN_DURATION = 0.5
        self.MARGIN = 1
        self.SILENCE_TIME = 0.6

        # State
        self.reset_state()

    def reset_state(self):
        self.audio_buffer = deque(maxlen=self.BUFFER_SIZE)
        self.is_speaking = False
        self.speech_start_sample = None
        self.sample_counter = 0
        self.silence_counter = 0
        self.ema_speech_prob = 0
        self.saved_filename = None

    def _save_audio_segment(self, start_sample, end_sample):
        audio_array = np.array(list(self.audio_buffer), dtype=np.int16)
        start = max(0, start_sample - int(self.MARGIN * self.SAMPLE_RATE))
        end = min(len(audio_array), end_sample + int(self.MARGIN * self.SAMPLE_RATE))
        segment = audio_array[start:end]

        if len(segment) / self.SAMPLE_RATE < self.MIN_DURATION:
            print(f"Segment too short, skipping save.")
            return

        filename = f"speech_{time.strftime('%Y%m%d_%H%M%S')}.wav"
        sf.write(filename, segment, self.SAMPLE_RATE)
        print(f"Audio saved: {filename}")
        self.saved_filename = filename

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(status)

        if self.saved_filename:  # Stop if we already have a file
            return

        audio_int16 = (indata * 32768).astype(np.int16).flatten()
        self.audio_buffer.extend(audio_int16)

        if len(audio_int16) < 512:
            return

        audio_tensor = torch.from_numpy(audio_int16).float()
        speech_prob = self.vad_iterator.model(audio_tensor, self.SAMPLE_RATE).item()
        self.ema_speech_prob = 0.9 * self.ema_speech_prob + 0.1 * speech_prob

        if self.ema_speech_prob > self.THRESHOLD:
            if not self.is_speaking:
                self.is_speaking = True
                self.speech_start_sample = self.sample_counter
            self.silence_counter = 0
        else:
            if self.is_speaking:
                self.silence_counter += frames / self.SAMPLE_RATE
                if self.silence_counter >= self.SILENCE_TIME:
                    self.is_speaking = False
                    speech_end_sample = self.sample_counter
                    duration = (speech_end_sample - self.speech_start_sample) / self.SAMPLE_RATE
                    if duration >= self.MIN_DURATION:
                        self._save_audio_segment(self.speech_start_sample, speech_end_sample)

        self.sample_counter += frames

    def record(self, timeout=10):
        self.reset_state()
        stream = sd.InputStream(callback=self._callback, channels=1, samplerate=self.SAMPLE_RATE, blocksize=512)
        with stream:
            print("Listening for speech...")
            start_time = time.time()
            while time.time() - start_time < timeout:
                if self.saved_filename:
                    break
                sd.sleep(100)

        print("Finished listening.")
        return self.saved_filename


# For use in other scripts
def listen_and_record_speech(timeout=10):
    """
    Creates a VADRecorder instance and records one speech segment.
    Returns the filename or None.
    """
    # This is a workaround for a potential issue where the VAD model
    # needs to be reloaded in certain execution contexts.
    recorder = VADRecorder()
    filename = recorder.record(timeout=timeout)
    return filename


# ====== 1. 상태(State) 정의 ======
class State(Enum):
    IDLE = 0
    USER_CHECK = 1
    ENROLL = 2
    WELCOME = 3
    ASR = 4
    BYE = 5


# ====== 2. 전역 변수 및 플래그 ======
# 상태 전이를 위한 플래그
FACE_DETECTED = False
USER_EXIST = False
ENROLL_SUCCESS = False
VAD = False
BYE_EXIST = False
TIMER_EXPIRED = False  # WELCOME, BYE 상태의 타이머

# 상태 간 데이터 공유를 위한 변수
sh_face_crop = None
sh_bbox = None
sh_embedding = None
sh_current_user = None
sh_audio_file = None  # 음성 파일 경로 (모의)
sh_message = "Initializing..."
sh_color = (255, 255, 0)
sh_timer_end = 0
sh_prev_unkonw = None

# ====== 3. 유틸리티 함수 ======
DB_PATH = "faces_db.npy"
SIM_THRESHOLD = 0.65


def load_db():
    if os.path.exists(DB_PATH):
        data = np.load(DB_PATH, allow_pickle=True).item()
        return data["name_list"], data["group_list"], data["embeddings"]
    else:
        return [], [], np.empty((0, 512))


def save_db(name_list, group_list, embeddings):
    np.save(DB_PATH, {"name_list": name_list, "group_list": group_list, "embeddings": embeddings})


def get_name_group_popup():
    root = tk.Tk()
    root.withdraw()

    name = simpledialog.askstring("Face Registration", "Enter your name:")
    if not name:
        root.destroy()
        return None, None

    group = simpledialog.askstring("Face Registration", "Enter your group:")
    root.destroy()
    return name, group


def find_match(embedding, name_list, embeddings):
    if len(embeddings) == 0: return None, 0
    sims = [np.dot(embedding, emb) / (np.linalg.norm(embedding) * np.linalg.norm(emb)) for emb in embeddings]
    max_idx = np.argmax(sims)
    if sims[max_idx] >= SIM_THRESHOLD:
        return name_list[max_idx], sims[max_idx]
    else:
        return None, sims[max_idx]


# --- 얼굴 검출 및 bbox 갱신 함수 ---
def update_face_detection():
    global FACE_DETECTED, sh_face_crop, sh_bbox, sh_frame

    image = sh_frame.copy()
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_image)

    if results.detections and len(results.detections) == 1:
        FACE_DETECTED = True
        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = image.shape
        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
        x, y = max(0, x), max(0, y)
        sh_bbox = (x, y, w, h)
        sh_face_crop = image[y:y + h, x:x + w]
        if sh_face_crop.size == 0:
            FACE_DETECTED = False
    else:
        FACE_DETECTED = False
        sh_bbox = None
        sh_face_crop = None

    return results


# --- 음성 처리 모의(Mock) 함수 ---
# TODO: 이 함수들을 실제 음성 처리 모듈로 교체하세요.

import whisper

whisper_model = whisper.load_model("base")


def asr_from_wav(file_path: str) -> str:
    print(f"C:\\Users\\Qualcomm\\workspace\\famigo\\famigo\\{file_path}",
          os.path.exists(f"C:\\Users\\Qualcomm\\workspace\\famigo\\famigo\\{file_path}"))
    result = whisper_model.transcribe(f"C:\\Users\\Qualcomm\\workspace\\famigo\\famigo\\{file_path}")
    print(result)
    return result['text']


# ====== 4. 각 상태의 행동(Action) 함수 정의 ======

def enter_idle():
    global sh_message, sh_color

    results = update_face_detection()  # 얼굴 검출 및 bbox 갱신

    if not FACE_DETECTED:
        if results.detections and len(results.detections) > 1:
            sh_message = f"{len(results.detections)} faces detected. Only one please."
            sh_color = (0, 0, 255)
        else:
            sh_message = "Waiting for user..."
            sh_color = (255, 255, 0)


def enter_user_check():
    global USER_EXIST, sh_embedding, sh_current_user, sh_message, sh_color

    update_face_detection()  # 얼굴 검출 및 bbox 갱신

    if sh_face_crop is None: return

    face_pil = Image.fromarray(cv2.cvtColor(sh_face_crop, cv2.COLOR_BGR2RGB))
    face_tensor = preprocess(face_pil).unsqueeze(0)
    with torch.no_grad():
        embedding = resnet(face_tensor)[0].cpu().numpy()
    sh_embedding = embedding / np.linalg.norm(embedding)

    match_name, sim = find_match(sh_embedding, name_list, embeddings)
    if match_name:
        USER_EXIST = True
        sh_current_user = match_name
        sh_message = f"Identifying... {match_name} ({sim:.2f})"
        sh_color = (0, 255, 0)
    else:
        USER_EXIST = False
        sh_message = f"Unknown user. Press 'y' to enroll."
        sh_color = (0, 255, 255)


def enter_enroll(key):
    global ENROLL_SUCCESS, name_list, group_list, embeddings, sh_current_user, sh_message, sh_color

    ENROLL_SUCCESS = False
    results = update_face_detection()  # 얼굴 검출 및 bbox 갱신

    if not FACE_DETECTED:
        if results.detections and len(results.detections) > 1:
            sh_message = f"{len(results.detections)} faces detected. Only one please."
            sh_color = (0, 0, 255)
        else:
            sh_message = "Waiting for user..."
            sh_color = (255, 255, 0)
    else:
        sh_message = f"Unknown user. Press 'y' to enroll."
        sh_color = (0, 255, 255)

    if key == ord('y'):
        new_name, new_group = get_name_group_popup()
        if new_name and sh_embedding is not None:
            name_list.append(new_name)
            group_list.append(new_group)
            if embeddings.size:
                embeddings = np.vstack([embeddings, sh_embedding])
            else:
                embeddings = np.array([sh_embedding])
            save_db(name_list, group_list, embeddings)
            sh_current_user = new_name
            ENROLL_SUCCESS = True
            print(f"Enrollment successful for {sh_current_user} from group {new_group}")
        else:
            print("Enrollment cancelled.")


def enter_welcome():
    global VAD, sh_audio_file, TIMER_EXPIRED, sh_message, sh_color

    update_face_detection()  # 얼굴 검출 및 bbox 갱신

    TIMER_EXPIRED = False
    VAD = False
    sh_message = f"Hi, {sh_current_user}!"
    sh_color = (0, 255, 0)

    if time.time() > sh_timer_end:
        TIMER_EXPIRED = True
        cv2.destroyAllWindows()
        print(1)

        sh_audio_file = listen_and_record_speech(timeout=5)
        if sh_audio_file:
            VAD = True


def enter_asr():
    global BYE_EXIST

    update_face_detection()  # 얼굴 검출 및 bbox 갱신

    text = asr_from_wav(sh_audio_file)
    text = "".join(text.split())
    if "잘가" in text or "bye" in text.lower():
        BYE_EXIST = True
    else:
        BYE_EXIST = False


def enter_bye():
    global TIMER_EXPIRED, sh_message, sh_color

    update_face_detection()  # 얼굴 검출 및 bbox 갱신

    TIMER_EXPIRED = False
    sh_message = f"Bye, {sh_current_user}!"
    sh_color = (255, 0, 255)

    if time.time() > sh_timer_end:
        TIMER_EXPIRED = True


# ====== 5. 상태 전이 및 호출 함수 ======
def state_transition(current_state: State) -> State:
    global sh_prev_unkonw, sh_embedding
    if current_state == State.IDLE:
        return State.USER_CHECK if FACE_DETECTED else State.IDLE
    elif current_state == State.USER_CHECK:
        return State.WELCOME if USER_EXIST else State.ENROLL

    elif current_state == State.ENROLL:
        if not ENROLL_SUCCESS and (
                sh_embedding is not None and len(sh_embedding) != 0 and sh_prev_unkonw is not None and len(
                sh_prev_unkonw) != 0) and ((np.dot(sh_prev_unkonw, sh_embedding) / (
                np.linalg.norm(sh_prev_unkonw) * np.linalg.norm(sh_embedding))) >= SIM_THRESHOLD):
            return State.ENROLL
        sh_prev_unkonw = sh_embedding
        return State.WELCOME if ENROLL_SUCCESS else State.IDLE

    elif current_state == State.WELCOME:
        if TIMER_EXPIRED:
            return State.ASR if VAD else State.IDLE
        return State.WELCOME
    elif current_state == State.ASR:
        return State.BYE if BYE_EXIST else State.IDLE
    elif current_state == State.BYE:
        return State.IDLE if TIMER_EXPIRED else State.BYE
    return current_state


def call_state_fn(state: State, key):
    if state == State.IDLE:
        enter_idle()
    elif state == State.USER_CHECK:
        enter_user_check()
    elif state == State.ENROLL:
        enter_enroll(key)
    elif state == State.WELCOME:
        enter_welcome()
    elif state == State.ASR:
        enter_asr()
    elif state == State.BYE:
        enter_bye()


# ====== 6. 모델 초기화 및 메인 루프 ======
print("Loading models...")
resnet = InceptionResnetV1(pretrained='vggface2').eval()
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
preprocess = transforms.Compose([
        transforms.Resize((160, 160)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
name_list, group_list, embeddings = load_db()
print("Models loaded.")

import time
import cv2
import streamlit as st

st.set_page_config(page_title="OpenCV Camera on Streamlit", layout="centered")

st.title("📷 OpenCV Camera on Streamlit")

# 카메라 인덱스 선택 (0이 기본 내장/첫 번째 웹캠)
cam_index = st.number_input("Camera index", min_value=0, max_value=10, value=0, step=1)

# 좌우반전/해상도 옵션
flip = st.checkbox("Flip horizontally", value=True)
width = st.slider("Frame width", 320, 1920, 640, step=10)
run = st.toggle("Run camera", value=False)

# 영상 표시용 placeholder
frame_slot = st.empty()

# 상태 유지용
if "cap" not in st.session_state:
    st.session_state.cap = None

def open_camera(index: int):
    cap = cv2.VideoCapture(index)
    # 해상도 설정 (가능한 경우)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    return cap

# cap = cv2.VideoCapture(0)
state = State.IDLE
print("Starting state machine...")
if run:
    # 카메라 열기
    if st.session_state.cap is None or not st.session_state.cap.isOpened():
        st.session_state.cap = open_camera(int(cam_index))
        if not st.session_state.cap.isOpened():
            st.error("카메라를 열 수 없습니다. 인덱스를 바꾸거나 다른 앱을 종료해보세요.")
            st.stop()

    while run:
        success, sh_frame = st.session_state.cap.read()
        if not success: break
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break

        call_state_fn(state, key)
        new_state = state_transition(state)

        if new_state != state:
            print(f"State Change: {state.name} -> {new_state.name}")
            state = new_state
            # 상태 진입 시 초기화 로직
            if state == State.WELCOME:
                sh_timer_end = time.time() + 2.0  # 2초간 인사
            elif state == State.BYE:
                sh_timer_end = time.time() + 2.0  # 2초간 작별인사

        # 화면 그리기
        display_frame = sh_frame.copy()

        if sh_bbox:
            x, y, w, h = sh_bbox
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), sh_color, 2)
            cv2.putText(display_frame, sh_message, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, sh_color, 2)
        else:
            cv2.putText(display_frame, sh_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, sh_color, 2)
        # cv2.imshow('State Machine', display_frame)
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

        # 사이즈 맞추기 (가로 기준)
        h, w, _ = frame_rgb.shape
        new_h = int(h * (width / w))
        frame_rgb = cv2.resize(frame_rgb, (int(width), new_h))

        # 화면에 출력
        frame_slot.image(frame_rgb, channels="RGB", caption="Live", use_container_width=False)

        # CPU 사용률/지연 줄이기
        time.sleep(0.01)

        # 토글 상태 갱신
        run = st.session_state.get("_toggle_run", True)  # 내부 보호

    # 루프 종료 시 자원 해제
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
        frame_slot.empty()
        st.info("카메라를 종료했습니다.")

    # cap.release()
    # cv2.destroyAllWindows()
    # face_detection.close()