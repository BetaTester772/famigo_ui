import os
import time
import ssl
import threading
from enum import Enum
from collections import deque

import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import torchvision.transforms as transforms
from PIL import Image
import mediapipe as mp
import streamlit as st
from facenet_pytorch import InceptionResnetV1

ssl._create_default_https_context = ssl._create_unverified_context  # (torch.hub SSL 회피용)


# =========================
# 캐시된 싱글톤 리소스 (중복 로딩 방지)
# =========================

@st.cache_resource
def get_whisper_model(model_name="large-v3", device=None):
    import whisper
    return whisper.load_model(model_name) if device is None else whisper.load_model(model_name, device=device)


@st.cache_resource
def get_facenet_model():
    return InceptionResnetV1(pretrained='vggface2').eval()


@st.cache_resource
def get_silero_vad_bundle():
    model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            trust_repo=True
    )
    return model, utils


@st.cache_resource
def get_face_detector():
    return mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)


# =========================
# VAD Recorder (공유 모델 사용)
# =========================

class VADRecorder:
    def __init__(self, model=None, utils=None):
        if model is None or utils is None:
            model, utils = get_silero_vad_bundle()
        self.model = model
        self.utils = utils
        self.vad_iterator = self.utils[3](self.model)  # VADIterator

        self.SAMPLE_RATE = 16000
        self.BUFFER_SIZE = self.SAMPLE_RATE * 60  # 1 minute buffer
        self.THRESHOLD = 0.65
        self.MIN_DURATION = 0.5
        self.MARGIN = 1
        self.SILENCE_TIME = 0.6

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
            print("Segment too short, skipping save.")
            return
        filename = f"audio/speech_{time.strftime('%Y%m%d_%H%M%S')}.wav"
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


def listen_and_record_speech(timeout=10):
    model, utils = get_silero_vad_bundle()
    recorder = VADRecorder(model=model, utils=utils)
    filename = recorder.record(timeout=timeout)
    return filename


# =========================
# State Definition
# =========================

class State(Enum):
    IDLE = 0
    USER_CHECK = 1
    ENROLL = 2
    WELCOME = 3
    ASR = 4
    BYE = 5


# =========================
# Globals & Flags
# =========================

# 상태 플래그
FACE_DETECTED = False
USER_EXIST = False
ENROLL_SUCCESS = False
VAD = False
BYE_EXIST = False
TIMER_EXPIRED = False  # WELCOME, BYE state's timer

# 공유 데이터
sh_face_crop = None
sh_bbox = None
sh_embedding = None
sh_current_user = None
sh_audio_file = None
sh_message = "Initializing..."
sh_color = (255, 255, 0)
sh_timer_end = 0
sh_prev_unkonw = None

# 비동기 작업 플래그
VAD_TASK_STARTED = False
VAD_TASK_RUNNING = False
ASR_TASK_STARTED = False
ASR_TASK_RUNNING = False
ASR_TEXT = None

# DB/Threshold
DB_PATH = "faces_db.npy"
SIM_THRESHOLD = 0.65


# =========================
# Utils
# =========================

def load_db():
    if os.path.exists(DB_PATH):
        data = np.load(DB_PATH, allow_pickle=True).item()
        return data["name_list"], data["group_list"], data["embeddings"]
    else:
        return [], [], np.empty((0, 512))


def save_db(name_list, group_list, embeddings):
    np.save(DB_PATH, {"name_list": name_list, "group_list": group_list, "embeddings": embeddings})


def find_match(embedding, name_list, embeddings):
    if len(embeddings) == 0:
        return None, 0
    sims = [np.dot(embedding, emb) / (np.linalg.norm(embedding) * np.linalg.norm(emb)) for emb in embeddings]
    max_idx = np.argmax(sims)
    if sims[max_idx] >= SIM_THRESHOLD:
        return name_list[max_idx], sims[max_idx]
    else:
        return None, sims[max_idx]


# Face detection & bbox
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


# =========================
# Whisper ASR (캐시된 모델 사용)
# =========================

def asr_from_wav(file_path: str) -> str:
    print(f"./{file_path}", os.path.exists(f"./{file_path}"))
    result = whisper_model.transcribe(f"./{file_path}")
    return result['text']


# =========================
# State Action Functions
# =========================

def enter_idle():
    global sh_message, sh_color
    results = update_face_detection()
    if not FACE_DETECTED:
        if results.detections and len(results.detections) > 1:
            sh_message = f"{len(results.detections)} faces detected. Only one please."
            sh_color = (0, 0, 255)
        else:
            sh_message = "Waiting for user..."
            sh_color = (255, 255, 0)


def enter_user_check():
    global USER_EXIST, sh_embedding, sh_current_user, sh_message, sh_color

    update_face_detection()
    if sh_face_crop is None:
        return

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
        sh_message = "Unknown user. Use the right panel to enroll."
        sh_color = (0, 255, 255)


def enter_enroll(key=None):
    # 키보드 입력은 사용하지 않음 (UI에서 처리)
    global sh_message, sh_color

    results = update_face_detection()
    if not FACE_DETECTED:
        if results.detections and len(results.detections) > 1:
            sh_message = f"{len(results.detections)} faces detected. Only one please."
            sh_color = (0, 0, 255)
        else:
            sh_message = "등록을 위해 얼굴을 카메라에 비춰주세요."
            sh_color = (255, 255, 0)
    else:
        sh_message = "알 수 없는 사용자입니다. 오른쪽 패널의 폼으로 등록하세요."
        sh_color = (0, 255, 255)


def enter_welcome():
    global VAD, sh_audio_file, TIMER_EXPIRED, sh_message, sh_color

    update_face_detection()
    sh_message = f"Hi, {sh_current_user}!"
    sh_color = (0, 255, 0)

    TIMER_EXPIRED = (time.time() > sh_timer_end)
    if TIMER_EXPIRED and not VAD_TASK_STARTED:
        start_vad_async(timeout=5)


def enter_asr():
    update_face_detection()
    if sh_audio_file and not ASR_TASK_STARTED:
        start_asr_async(sh_audio_file)


def enter_bye():
    global TIMER_EXPIRED, sh_message, sh_color

    update_face_detection()
    sh_message = f"Bye, {sh_current_user}!"
    sh_color = (255, 0, 255)
    TIMER_EXPIRED = (time.time() > sh_timer_end)


# =========================
# Async Workers
# =========================

def start_vad_async(timeout=5):
    """녹음을 비동기로 시작."""
    global VAD_TASK_STARTED, VAD_TASK_RUNNING
    if VAD_TASK_RUNNING:
        return
    VAD_TASK_STARTED = True
    VAD_TASK_RUNNING = True

    def _worker():
        global sh_audio_file, VAD, VAD_TASK_RUNNING
        try:
            filename = listen_and_record_speech(timeout=timeout)
            if filename:
                sh_audio_file = filename
                VAD = True
            else:
                VAD = False
        finally:
            VAD_TASK_RUNNING = False

    threading.Thread(target=_worker, daemon=True).start()


def start_asr_async(file_path: str):
    """Whisper를 비동기로 실행."""
    global ASR_TASK_STARTED, ASR_TASK_RUNNING
    if ASR_TASK_RUNNING:
        return
    ASR_TASK_STARTED = True
    ASR_TASK_RUNNING = True

    def _worker():
        global ASR_TEXT, BYE_EXIST, ASR_TASK_RUNNING
        try:
            text = asr_from_wav(file_path)
            ASR_TEXT = text
            print(f"ASR result: {ASR_TEXT}")
            t = "".join(text.split())
            BYE_EXIST = ("잘가" in t) or ("bye" in t.lower())
        finally:
            ASR_TASK_RUNNING = False

    threading.Thread(target=_worker, daemon=True).start()


# =========================
# Transitions & Dispatcher
# =========================

def state_transition(current_state: State) -> State:
    global sh_prev_unkonw, sh_embedding, name_list, group_list, embeddings

    if current_state == State.IDLE:
        return State.USER_CHECK if FACE_DETECTED else State.IDLE

    elif current_state == State.USER_CHECK:
        return State.WELCOME if USER_EXIST else State.ENROLL

    elif current_state == State.ENROLL:
        if ENROLL_SUCCESS:
            name_list, group_list, embeddings = load_db()
            return State.WELCOME
        sh_prev_unkonw = sh_embedding
        return State.IDLE if not FACE_DETECTED else State.ENROLL

    elif current_state == State.WELCOME:
        if not (time.time() > sh_timer_end):
            return State.WELCOME
        if VAD:
            return State.ASR
        if VAD_TASK_RUNNING:
            return State.WELCOME
        return State.IDLE

    elif current_state == State.ASR:
        if ASR_TASK_RUNNING:
            return State.ASR
        if ASR_TASK_STARTED and not ASR_TASK_RUNNING:
            return State.BYE if BYE_EXIST else State.IDLE
        return State.ASR

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


# =========================
# Model Init (캐시 사용)
# =========================

print("Loading models (cached)...")
resnet = get_facenet_model()
face_detection = get_face_detector()
whisper_model = get_whisper_model("base.en")
preprocess = transforms.Compose([
        transforms.Resize((160, 160)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
name_list, group_list, embeddings = load_db()
print("Models loaded (using cache).")

# =========================
# Streamlit UI & Main Loop
# =========================

st.set_page_config(page_title="Face Kiosk", layout="wide")
st.title("👤 Face Kiosk with State UI")

# Layout
col_video, col_ui = st.columns([3, 2], vertical_alignment="top")

# Camera / Options
with col_video:
    st.subheader("📷 Camera")
    cam_index = st.number_input("Camera index", min_value=0, max_value=10, value=0, step=1)
    width = st.slider("Frame width", 320, 1920, 640, step=10)
    run = st.toggle("Run camera", value=False)
    frame_slot = st.empty()

# UI placeholders
with col_ui:
    st.subheader("🧭 State Panel")
    state_badge = st.empty()
    message_slot = st.empty()
    enroll_slot = st.empty()
    welcome_slot = st.empty()
    asr_slot = st.empty()
    bye_slot = st.empty()
    audio_slot = st.empty()
    debug_slot = st.expander("Debug", expanded=False)

# Keep only camera handle in session_state
if "cap" not in st.session_state:
    st.session_state.cap = None


def open_camera(index: int, target_w: int):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
    return cap


# ENROLL UI lifecycle flags & unique keys
ENROLL_UI_BUILT = False
enroll_face_ph = None
enroll_form_counter = 0
current_enroll_form_key = None
current_enroll_name_key = None
current_enroll_group_key = None

# Initial state
state = State.IDLE
st.caption("Starting state machine...")


# ENROLL submit helper
def ui_enroll_submit(new_name: str, new_group: str):
    global ENROLL_SUCCESS, USER_EXIST, name_list, group_list, embeddings, sh_current_user

    if not new_name or new_name.strip() == "":
        st.warning("이름은 필수입니다.")
        return
    if sh_embedding is None or len(sh_embedding) == 0:
        st.error("얼굴 임베딩이 준비되지 않았습니다. 카메라에 얼굴을 똑바로 비춰주세요.")
        return
    if any(n == new_name for n in name_list):
        st.warning("이미 존재하는 이름입니다. 다른 이름을 입력하세요.")
        return

    name_list.append(new_name)
    group_list.append(new_group)
    if embeddings.size:
        embeddings = np.vstack([embeddings, sh_embedding])
    else:
        embeddings = np.array([sh_embedding])

    save_db(name_list, group_list, embeddings)

    sh_current_user = new_name
    ENROLL_SUCCESS = True
    USER_EXIST = True

    st.success(f"등록 완료: {new_name} ({new_group if new_group else 'group 미지정'})")
    print("[DB Updated] ", name_list, group_list, embeddings.shape)


if run:
    # Open camera once
    if st.session_state.cap is None or not st.session_state.cap.isOpened():
        st.session_state.cap = open_camera(int(cam_index), int(width))
        if not st.session_state.cap.isOpened():
            st.error("카메라를 열 수 없습니다. 인덱스를 바꾸거나 다른 앱을 종료해보세요.")
            st.stop()


    # UI helper: render state panel
    def render_state_panel(current_state: State):
        global ENROLL_UI_BUILT, enroll_face_ph
        global current_enroll_form_key, current_enroll_name_key, current_enroll_group_key

        # Badge
        state_badge.markdown(f"**Current State:** :blue[{current_state.name}]")

        # ENROLL 외 상태 슬롯 정리
        if current_state != State.ENROLL:
            enroll_slot.empty()
        if current_state != State.WELCOME:
            welcome_slot.empty()
        if current_state != State.ASR:
            asr_slot.empty()
        if current_state != State.BYE:
            bye_slot.empty()

        # Message
        with message_slot.container():
            st.markdown(f"**Message:** {sh_message}")

        # ENROLL UI (form created once per entry)
        if current_state == State.ENROLL:
            # 안전장치: 혹시 키가 None이면 즉석 생성
            if current_enroll_form_key is None or current_enroll_name_key is None or current_enroll_group_key is None:
                ts = int(time.time() * 1000)
                current_enroll_form_key = f"form_enroll_{ts}"
                current_enroll_name_key = f"enroll_name_{ts}"
                current_enroll_group_key = f"enroll_group_{ts}"

            if not ENROLL_UI_BUILT:
                ENROLL_UI_BUILT = True
                with enroll_slot.container():
                    st.info("알 수 없는 사용자입니다. 아래 폼으로 등록을 진행하세요.")
                    enroll_face_ph = st.empty()

                    with st.form(key=current_enroll_form_key, clear_on_submit=False):
                        new_name = st.text_input("이름", key=current_enroll_name_key)
                        new_group = st.text_input("그룹(선택)", key=current_enroll_group_key)
                        submitted = st.form_submit_button("등록하기", use_container_width=True)
                    if submitted:
                        ui_enroll_submit(new_name, new_group)

            # 얼굴 미리보기 갱신
            if enroll_face_ph is not None:
                if sh_face_crop is not None and sh_face_crop.size != 0:
                    face_rgb = cv2.cvtColor(sh_face_crop, cv2.COLOR_BGR2RGB)
                    enroll_face_ph.image(face_rgb, caption="등록할 얼굴", use_container_width=True)
                else:
                    enroll_face_ph.warning("얼굴이 감지되지 않았습니다. 카메라를 향해 한 명만 비춰주세요.")

        else:
            # Leaving ENROLL -> flag reset
            if ENROLL_UI_BUILT:
                ENROLL_UI_BUILT = False
                enroll_face_ph = None

        # WELCOME UI (녹음 진행상태)
        if current_state == State.WELCOME:
            with welcome_slot.container():
                if not (time.time() > sh_timer_end):
                    remain = max(0.0, sh_timer_end - time.time())
                    st.success(f"Hi, **{sh_current_user}**! 곧 녹음을 시작합니다.")
                    st.progress(min(max(1.0 - (remain / 2.0), 0.0), 1.0), text="Greeting...")
                else:
                    if VAD_TASK_RUNNING:
                        st.info("🎙️ 음성 녹음 중...")
                    elif VAD:
                        st.success("🎧 음성 캡처 완료! ASR로 이동합니다.")
                    else:
                        st.warning("녹음을 시작하지 못했습니다. 돌아갑니다.")

        # ASR UI (진행상태)
        if current_state == State.ASR:
            with asr_slot.container():
                if ASR_TASK_RUNNING:
                    st.info("🧠 Whisper로 음성을 변환 중...")
                elif ASR_TEXT is not None:
                    st.write("**ASR 결과:** ", ASR_TEXT)
                    st.write(f"**BYE detected:** {'Yes' if BYE_EXIST else 'No'}")
                else:
                    st.write("대기 중...")

        # BYE UI
        if current_state == State.BYE:
            with bye_slot.container():
                st.warning(f"Bye, **{sh_current_user}**!")
                remain = max(0.0, sh_timer_end - time.time())
                pct = min(max(1.0 - (remain / 2.0), 0.0), 1.0)
                st.progress(pct, text="Ending...")


    # Main loop
    while run:
        success, sh_frame = st.session_state.cap.read()
        if not success:
            st.error("프레임을 읽지 못했습니다.")
            break

        key = cv2.waitKey(1) & 0xFF

        # Call & transition
        previous_state = state
        call_state_fn(state, key)
        new_state = state_transition(state)

        if new_state != state:
            print(f"State Change: {state.name} -> {new_state.name}")

            # ENROLL로 진입 시: 고유 키 생성 + 초기화
            if new_state == State.ENROLL and state != State.ENROLL:
                ENROLL_SUCCESS = False
                USER_EXIST = False
                ENROLL_UI_BUILT = False
                enroll_form_counter += 1
                current_enroll_form_key = f"form_enroll_{enroll_form_counter}"
                current_enroll_name_key = f"enroll_name_{enroll_form_counter}"
                current_enroll_group_key = f"enroll_group_{enroll_form_counter}"

            # WELCOME로 진입 시: 타이머/녹음 플래그 초기화
            if new_state == State.WELCOME:
                sh_timer_end = time.time() + 2.0
                VAD = False
                VAD_TASK_STARTED = False
                VAD_TASK_RUNNING = False
                sh_audio_file = None

            # ASR로 진입 시: ASR 비동기 초기화
            if new_state == State.ASR:
                ASR_TEXT = None
                BYE_EXIST = False
                ASR_TASK_STARTED = False
                ASR_TASK_RUNNING = False

            # BYE로 진입 시: 타이머
            if new_state == State.BYE:
                sh_timer_end = time.time() + 2.0

            state = new_state

        # Update state panel
        render_state_panel(state)

        # Draw overlays
        display_frame = sh_frame.copy()
        if sh_bbox:
            x, y, w, h = sh_bbox
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), sh_color, 2)
            cv2.putText(display_frame, sh_message, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, sh_color, 2)
        else:
            cv2.putText(display_frame, sh_message, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, sh_color, 2)

        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h0, w0, _ = frame_rgb.shape
        new_h = int(h0 * (width / w0))
        frame_rgb = cv2.resize(frame_rgb, (int(width), new_h))
        frame_slot.image(frame_rgb, channels="RGB", caption="Live", use_container_width=True)

        # Debug info
        with debug_slot:
            st.write({
                    "FACE_DETECTED"     : FACE_DETECTED,
                    "USER_EXIST"        : USER_EXIST,
                    "ENROLL_SUCCESS"    : ENROLL_SUCCESS,
                    "VAD"               : VAD,
                    "BYE_EXIST"         : BYE_EXIST,
                    "TIMER_EXPIRED"     : TIMER_EXPIRED,
                    "current_user"      : sh_current_user,
                    "audio_file"        : sh_audio_file,
                    "id(whisper_model)" : id(whisper_model),
                    "id(resnet)"        : id(resnet),
                    "id(face_detection)": id(face_detection),
            })

        time.sleep(0.01)
        run = st.session_state.get("_toggle_run", True)

    # Cleanup
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
        frame_slot.empty()
        st.info("카메라를 종료했습니다.")
