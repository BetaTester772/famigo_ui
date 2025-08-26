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
from playsound3 import playsound
import gc

cv2.setNumThreads(1)  # ← 선택: OpenCV 내부 스레드 과다 사용 억제

# ===== paths / dirs =====
ssl._create_default_https_context = ssl._create_unverified_context  # torch.hub SSL 회피
TEMP_AUDIO_DIR = "audio"
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

# ===== feature toggles =====
ENABLE_TTS = True  # Kokoro가 설치돼 있지 않으면 자동으로 skip
DEFAULT_TTS_LANG = "a"  # Kokoro lang_code
DEFAULT_TTS_VOICE = "af_heart"  # Kokoro voice (예: 'af_heart')
DEFAULT_TTS_SR = 24000  # Kokoro sample rate


# =========================
# 캐시된 싱글톤 리소스 (중복 로딩 방지)
# =========================
@st.cache_resource
def get_whisper_model(model_name="medium.en", device=None):
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


@st.cache_resource
def get_kokoro_pipeline():
    if not ENABLE_TTS:
        return None
    try:
        from kokoro import KPipeline
        pipeline = KPipeline(lang_code=DEFAULT_TTS_LANG)
        return pipeline
    except Exception as e:
        print(f"[TTS] Kokoro pipeline init failed: {e}")
        return None


# =========================
# VAD Recorder (shared model)
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
        filename = f"{TEMP_AUDIO_DIR}/speech_{time.strftime('%Y%m%d_%H%M%S')}.wav"
        sf.write(filename, segment, self.SAMPLE_RATE)
        print(f"Audio saved: {filename}")
        self.saved_filename = filename

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(status)
        if self.saved_filename:
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


def listen_and_record_speech(timeout=10, model=None, utils=None):
    if model is None or utils is None:
        raise RuntimeError("VAD model/utils must be provided from main thread")
    recorder = VADRecorder(model=model, utils=utils)
    return recorder.record(timeout=timeout)


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
FACE_DETECTED = False
USER_EXIST = False
ENROLL_SUCCESS = False
VAD = False
BYE_EXIST = False
TIMER_EXPIRED = False  # WELCOME/BYE timer

# shared data
sh_face_crop = None
sh_bbox = None
sh_embedding = None
sh_current_user = None
sh_audio_file = None
sh_tts_file = None
sh_message = "Initializing..."
sh_color = (255, 255, 0)
sh_timer_end = 0
# sh_prev_unkown = None

SESSION_USER = None
USER_SWITCHED = False

# ❗ 세션 한정 그룹명 (WELCOME~BYE 사이 메모리 보관)
sh_session_group = None

# async flags
VAD_TASK_STARTED = False
VAD_TASK_RUNNING = False
ASR_TASK_STARTED = False
ASR_TASK_RUNNING = False
ASR_TEXT = None

# DB / threshold
DB_PATH = "faces_db.npy"
SIM_THRESHOLD = 0.5

# BBOX smoothing
BBOX_AVG_N = 5
_bbox_history = deque(maxlen=BBOX_AVG_N)


# =========================
# Utils (DB: name_list + embeddings 만 사용)
# =========================
def load_db():
    if os.path.exists(DB_PATH):
        data = np.load(DB_PATH, allow_pickle=True).item()
        name = data["name_list"]
        embs = data["embeddings"]
        return name, embs
    else:
        return [], np.empty((0, 512))


def save_db(name_list, embeddings):
    np.save(DB_PATH, {"name_list": name_list, "embeddings": embeddings})


def find_match(embedding, name_list, embeddings):
    if len(embeddings) == 0:
        return None, 0
    sims = [np.dot(embedding, emb) / (np.linalg.norm(embedding) * np.linalg.norm(emb)) for emb in embeddings]
    max_idx = np.argmax(sims)
    if sims[max_idx] >= SIM_THRESHOLD:
        return name_list[max_idx], sims[max_idx]
    else:
        return None, sims[max_idx]


def detect_user_change():
    """세션 중 사용자 변경 감지: SESSION_USER와 현재 프레임의 매칭 결과가 다르면 True"""
    global USER_SWITCHED, sh_message, sh_color

    if SESSION_USER is None:
        return  # 아직 세션 시작 전

    if sh_face_crop is None or sh_face_crop.size == 0:
        return  # 얼굴 미검출 상태는 변경으로 보지 않음

    face_pil = Image.fromarray(cv2.cvtColor(sh_face_crop, cv2.COLOR_BGR2RGB))
    face_tensor = preprocess(face_pil).unsqueeze(0)
    with torch.no_grad():
        emb = resnet(face_tensor)[0].cpu().numpy()
    emb = emb / np.linalg.norm(emb)

    match_name, sim = find_match(emb, name_list, embeddings)

    # 다른 "알려진" 사용자가 잡히면 변경으로 판단
    if match_name and match_name != SESSION_USER:
        USER_SWITCHED = True
        sh_message = f"User changed → {match_name}. Ending session..."
        sh_color = (255, 0, 0)


def _clip_bbox(x, y, w, h, iw, ih):
    x = max(0, min(x, iw - 1))
    y = max(0, min(y, ih - 1))
    w = max(1, min(w, iw - x))
    h = max(1, min(h, ih - y))
    return x, y, w, h


# Face detection + averaged bbox over last N frames
def update_face_detection():
    global FACE_DETECTED, sh_face_crop, sh_bbox, sh_frame, _bbox_history, BBOX_AVG_N

    # deque maxlen 동기화
    if _bbox_history.maxlen != BBOX_AVG_N:
        _bbox_history = deque(list(_bbox_history), maxlen=BBOX_AVG_N)

    image = sh_frame  # 불필요한 copy() 삭제
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_image)

    ih, iw, _ = image.shape
    det_count = len(results.detections) if (results and results.detections) else 0

    if det_count == 1:
        FACE_DETECTED = True
        det = results.detections[0]
        bboxC = det.location_data.relative_bounding_box
        x = int(bboxC.xmin * iw);
        y = int(bboxC.ymin * ih)
        w = int(bboxC.width * iw);
        h = int(bboxC.height * ih)
        x, y, w, h = _clip_bbox(x, y, w, h, iw, ih)

        _bbox_history.append((x, y, w, h))
        xs = np.mean([b[0] for b in _bbox_history])
        ys = np.mean([b[1] for b in _bbox_history])
        ws = np.mean([b[2] for b in _bbox_history])
        hs = np.mean([b[3] for b in _bbox_history])
        xa, ya, wa, ha = _clip_bbox(int(xs), int(ys), int(ws), int(hs), iw, ih)

        sh_bbox = (xa, ya, wa, ha)
        # ROI는 copy()로 분리해 상위 프레임 버퍼 참조 끊기
        sh_face_crop = image[ya:ya + ha, xa:xa + wa].copy()
        if sh_face_crop.size == 0:
            FACE_DETECTED = False
            _bbox_history.clear()
            sh_bbox = None
            sh_face_crop = None
    else:
        FACE_DETECTED = False
        _bbox_history.clear()
        sh_bbox = None
        sh_face_crop = None

    # 무거운 중간 객체 즉시 해제 힌트
    del rgb_image, results
    return det_count


# =========================
# Whisper ASR (cached)
# =========================
def asr_from_wav(file_path: str) -> str:
    result = whisper_model.transcribe(file_path, language='en', fp16=False)
    return result['text']


# =========================
# TTS helpers (Kokoro)
# =========================
def build_tts_reply_text(asr_text: str, user: str | None) -> str:
    t = "".join(asr_text.split()).lower()
    if "잘가" in t or "bye" in t:
        return f"Good bye, {user if user else ''}."
    # 간단 에코 응답
    return f"{user if user else ''} said that {asr_text}"


def synthesize_tts_kokoro(text: str) -> str | None:
    # 메인에서 만든 kokoro_pipeline 전역을 그대로 사용 (스레드에서 cache 호출 금지)
    if kokoro_pipeline is None or not ENABLE_TTS:
        return None
    try:
        chunks = []
        for _, _, audio in kokoro_pipeline(text, voice=DEFAULT_TTS_VOICE):
            chunks.append(audio)
        if not chunks:
            return None
        audio = chunks[0]
        out_path = f"{TEMP_AUDIO_DIR}/tts_{int(time.time())}.wav"
        sf.write(out_path, audio, DEFAULT_TTS_SR)
        playsound(out_path)
        return out_path
    except Exception as e:
        print(f"[TTS] synthesis failed: {e}")
        return None


# =========================
# State Action Functions
# =========================
def enter_idle():
    global sh_message, sh_color
    det_count = update_face_detection()
    if not FACE_DETECTED:
        if det_count > 1:
            sh_message = f"{det_count} faces detected. Only one please."
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
    gc.collect()


def enter_enroll(key=None):
    global sh_message, sh_color
    det_count = update_face_detection()
    if not FACE_DETECTED:
        if det_count > 1:
            sh_message = f"{det_count} faces detected. Only one please."
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
    detect_user_change()

    sh_message = f"Hi, {sh_current_user}!"
    sh_color = (0, 255, 0)

    TIMER_EXPIRED = (time.time() > sh_timer_end)
    if TIMER_EXPIRED and not VAD_TASK_STARTED:
        start_vad_async(timeout=5)


def enter_asr():
    update_face_detection()
    detect_user_change()

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
            filename = listen_and_record_speech(timeout=timeout, model=vad_model, utils=vad_utils)
            if filename:
                sh_audio_file = filename
                VAD = True
            else:
                VAD = False
        finally:
            VAD_TASK_RUNNING = False

    threading.Thread(target=_worker, daemon=True).start()


def start_asr_async(file_path: str):
    """Whisper를 비동기로 실행 + Kokoro TTS 생성."""
    global ASR_TASK_STARTED, ASR_TASK_RUNNING
    if ASR_TASK_RUNNING:
        return
    ASR_TASK_STARTED = True
    ASR_TASK_RUNNING = True

    def _worker():
        global ASR_TEXT, BYE_EXIST, ASR_TASK_RUNNING, sh_tts_file
        try:
            text = asr_from_wav(file_path)
            ASR_TEXT = text
            print("[ASR] ", text)
            t = "".join(text.split()).lower()
            BYE_EXIST = ("잘가" in t) or ("bye" in t)

            # TTS 생성
            tts_text = build_tts_reply_text(ASR_TEXT, sh_current_user)
            sh_tts_file = synthesize_tts_kokoro(tts_text)
        finally:
            ASR_TASK_RUNNING = False

    threading.Thread(target=_worker, daemon=True).start()


# =========================
# Transitions & Dispatcher
# =========================
def state_transition(current_state: State) -> State:
    global sh_embedding, name_list, embeddings  # ,sh_prev_unkown

    if current_state == State.IDLE:
        return State.USER_CHECK if FACE_DETECTED else State.IDLE

    elif current_state == State.USER_CHECK:
        if USER_EXIST:
            return State.WELCOME if sh_session_group else State.USER_CHECK
        else:
            return State.ENROLL

    elif current_state == State.ENROLL:
        if ENROLL_SUCCESS:
            name_list, embeddings = load_db()
            # USER_CHECK로 돌아가 group 입력
            return State.USER_CHECK
        # sh_prev_unkown = sh_embedding
        return State.IDLE if not FACE_DETECTED else State.ENROLL

    elif current_state == State.WELCOME:
        if USER_SWITCHED:
            return State.BYE

        if not (time.time() > sh_timer_end):
            return State.WELCOME
        if VAD:
            return State.ASR
        if VAD_TASK_RUNNING:
            return State.WELCOME
        return State.IDLE

    elif current_state == State.ASR:
        if USER_SWITCHED:
            return State.BYE

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
# Model Init (cached)
# =========================
print("Loading models (cached)...")
resnet = get_facenet_model()
face_detection = get_face_detector()
whisper_model = get_whisper_model("medium.en")

# 여기서 미리 로드하고, 전역으로 들고만 있음 (스레드에서 새로 부르지 않음)
vad_model, vad_utils = get_silero_vad_bundle()
kokoro_pipeline = get_kokoro_pipeline()

preprocess = transforms.Compose([
        transforms.Resize((160, 160)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
name_list, embeddings = load_db()
print("Models loaded (using cache).")

# =========================
# Streamlit UI & Main Loop
# =========================
st.set_page_config(page_title="Face Kiosk", layout="wide")
st.title("👤 Face Kiosk with State UI")

col_video, col_ui = st.columns([3, 2], vertical_alignment="top")

# Camera / Options
with col_video:
    st.subheader("📷 Camera")
    cam_index = st.number_input("Camera index", min_value=0, max_value=10, value=0, step=1)
    width = st.slider("Frame width", 320, 1920, 640, step=10)
    bbox_avg_n_ui = st.slider("BBOX smoothing (frames)", 1, 30, 5, help="Average the face bbox over N frames.")
    run = st.toggle("Run camera", value=True)
    frame_slot = st.empty()

# UI placeholders
with col_ui:
    st.subheader("🔖 Group")
    group_ui = st.empty()  # ← 그룹 입력 전용 placeholder
    st.subheader("🧭 State Panel")
    state_badge = st.empty()
    message_slot = st.empty()
    usercheck_slot = st.empty()
    enroll_slot = st.empty()
    welcome_slot = st.empty()
    asr_slot = st.empty()
    bye_slot = st.empty()
    audio_slot = st.empty()
    debug_slot = st.expander("Debug", expanded=False)
    with debug_slot:
        debug_ph = st.empty()  # ← 한 칸 확보

if "group_key_counter" not in st.session_state:
    st.session_state.group_key_counter = 1

if "current_group_key" not in st.session_state:
    st.session_state.current_group_key = f"usercheck_group_input_{st.session_state.group_key_counter}"

# 👇 그룹 입력창: 루프 바깥에서 단 1회만 생성
with group_ui.container():
    _gkey = st.session_state.current_group_key
    _gval = (st.session_state.get(_gkey, "") or "").strip()

    st.text_input(
            "그룹명 (BYE 후에만 다시 입력)",
            key=_gkey,  # ← 현재 키만 사용
            placeholder="예: slpr",
            disabled=bool(_gval),  # 값이 있으면 잠금
    )
    st.caption(f"현재 그룹: {_gval or '-'}")

GROUP_INPUT_COUNTER = 0  # 고정: 카운터
CURRENT_GROUP_KEY = None  # 현재 프레임에서 사용할 user_key

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

# WELCOME 그룹 입력용 고유 키
WELCOME_KEY = None
WELCOME_KEY_COUNTER = 0

# WELCOME UI 1회 생성 가드 & 플레이스홀더
WELCOME_UI_BUILT = False
welcome_group_ph = None
welcome_status_ph = None
welcome_progress_ph = None

# Initial state
state = State.IDLE
st.caption("Starting state machine...")


# ENROLL submit helper (이름만 저장)
def ui_enroll_submit(new_name: str):
    global ENROLL_SUCCESS, USER_EXIST, name_list, embeddings, sh_current_user

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
    if embeddings.size:
        embeddings = np.vstack([embeddings, sh_embedding])
    else:
        embeddings = np.array([sh_embedding])

    save_db(name_list, embeddings)

    sh_current_user = new_name
    ENROLL_SUCCESS = True
    USER_EXIST = True

    st.success(f"등록 완료: {new_name}")
    print("[DB Updated] ", name_list, embeddings.shape)


# UI render helper
def render_state_panel(current_state: State):
    global ENROLL_UI_BUILT, enroll_face_ph, current_enroll_name_key, current_enroll_form_key
    global current_usercheck_key, usercheck_form_counter, sh_session_group  # ← 정리

    state_badge.markdown(f"**Current State:** :blue[{current_state.name}]")

    # 상태별 슬롯 정리
    if current_state != State.USER_CHECK:
        if usercheck_slot is not None:
            usercheck_slot.empty()

    if current_state != State.ENROLL:
        enroll_slot.empty()
    if current_state != State.WELCOME:
        welcome_slot.empty()
    if current_state != State.ASR:
        asr_slot.empty()
    if current_state != State.BYE:
        bye_slot.empty()

    message_slot.markdown(f"**Message:** {sh_message}")

    # ---------- USER_CHECK ----------
    if current_state == State.USER_CHECK:
        usercheck_slot.empty()
        with usercheck_slot.container():
            st.info("사용자 확인 중입니다. 위의 ‘그룹명’ 입력창에 입력하세요. (BYE 전까지 유지)")

    # ---------- ENROLL ----------
    if current_state == State.ENROLL:
        if current_enroll_form_key is None or current_enroll_name_key is None:
            ts = int(time.time() * 1000)
            current_enroll_form_key = f"form_enroll_{ts}"
            current_enroll_name_key = f"enroll_name_{ts}"

        if not ENROLL_UI_BUILT:
            ENROLL_UI_BUILT = True
            with enroll_slot.container():
                st.info("알 수 없는 사용자입니다. 아래 폼으로 등록을 진행하세요.")
                globals()['enroll_face_ph'] = st.empty()
                with st.form(key=current_enroll_form_key, clear_on_submit=False):
                    new_name = st.text_input("이름", key=current_enroll_name_key)
                    submitted = st.form_submit_button("등록하기", use_container_width=True)
                if submitted:
                    ui_enroll_submit(new_name)

        if enroll_face_ph is not None:
            if sh_face_crop is not None and sh_face_crop.size != 0:
                face_rgb = cv2.cvtColor(sh_face_crop, cv2.COLOR_BGR2RGB)
                enroll_face_ph.image(face_rgb, caption="등록할 얼굴", use_container_width=True)
            else:
                enroll_face_ph.warning("얼굴이 감지되지 않았습니다. 카메라를 향해 한 명만 비춰주세요.")
    else:
        if ENROLL_UI_BUILT:
            ENROLL_UI_BUILT = False
            globals()['enroll_face_ph'] = None

    # ---------- WELCOME ----------
    if current_state == State.WELCOME:
        welcome_slot.empty()
        with welcome_slot.container():
            if sh_session_group:
                st.caption(f"(세션 그룹: {sh_session_group})")
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

    # ---------- ASR ----------
    if current_state == State.ASR:
        asr_slot.empty()
        with asr_slot.container():
            if ASR_TASK_RUNNING:
                st.info("🧠 Whisper로 음성을 변환 중...")
            elif ASR_TEXT is not None:
                st.write("**ASR 결과:** ", ASR_TEXT)
                st.write(f"**BYE detected:** {'Yes' if BYE_EXIST else 'No'}")
                if sh_tts_file:
                    audio_slot.audio(sh_tts_file)
            else:
                st.write("대기 중...")

    # ---------- BYE ----------
    if current_state == State.BYE:
        bye_slot.empty()
        with bye_slot.container():
            st.warning(f"Bye, **{sh_current_user}**!")
            if sh_session_group:
                st.caption(f"(세션 그룹: {sh_session_group})")
            remain = max(0.0, sh_timer_end - time.time())
            pct = min(max(1.0 - (remain / 2.0), 0.0), 1.0)
            st.progress(pct, text="Ending...")


# ========= run =========
if run:
    # camera open
    if st.session_state.cap is None or not st.session_state.cap.isOpened():
        st.session_state.cap = open_camera(int(cam_index), int(width))
        if not st.session_state.cap.isOpened():
            st.error("카메라를 열 수 없습니다. 인덱스를 바꾸거나 다른 앱을 종료해보세요.")
            st.stop()

    # initial state
    state = State.IDLE

    frame_i = 0
    # main loop
    while run:
        t0 = time.time()
        frame_i += 1

        # update bbox smoothing window
        BBOX_AVG_N = int(bbox_avg_n_ui)  # globally referenced in update_face_detection()

        success, sh_frame = st.session_state.cap.read()
        if not success:
            st.error("프레임을 읽지 못했습니다.")
            break

        # key = cv2.waitKey(1) & 0xFF
        key = None

        _gkey = st.session_state.current_group_key
        sh_session_group = (st.session_state.get(_gkey, "") or "").strip() or None

        # state call + transition
        call_state_fn(state, key)
        new_state = state_transition(state)

        if new_state != state:
            print(f"State Change: {state.name} -> {new_state.name}")

            # ENROLL로 진입 시: 초기화 및 폼 키 설정
            if new_state == State.ENROLL and state != State.ENROLL:
                ENROLL_SUCCESS = False
                USER_EXIST = False
                ENROLL_UI_BUILT = False
                enroll_form_counter += 1
                current_enroll_form_key = f"form_enroll_{enroll_form_counter}"
                current_enroll_name_key = f"enroll_name_{enroll_form_counter}"  # ← 추가

            # WELCOME로 진입 시: 타이머/녹음 플래그 초기화 + 세션 그룹 초기화
            if new_state == State.WELCOME:
                sh_timer_end = time.time() + 2.0
                VAD = False
                VAD_TASK_STARTED = False
                VAD_TASK_RUNNING = False
                sh_audio_file = None
                sh_tts_file = None

                SESSION_USER = sh_current_user
                USER_SWITCHED = False

                # sh_session_group = None

                # # 🔑 이번 방문용 고유 key 생성
                # WELCOME_KEY_COUNTER += 1
                # WELCOME_KEY = f"welcome_group_input_{WELCOME_KEY_COUNTER}"
                #
                # # 예전 welcome_group_input_* 키 제거
                # for k in list(st.session_state.keys()):
                #     if k.startswith("welcome_group_input_") and k != WELCOME_KEY:
                #         st.session_state.pop(k, None)
                #
                # # 다음 줄은 새로 추가: 이번 방문 UI를 다시 만들 수 있게 리셋
                # WELCOME_UI_BUILT = False

            # ASR로 진입 시: ASR 비동기 초기화
            if new_state == State.ASR:
                ASR_TEXT = None
                BYE_EXIST = False
                ASR_TASK_STARTED = False
                ASR_TASK_RUNNING = False
                # 그룹은 유지 (BYE까지)

            # BYE로 진입 시: 타이머
            if new_state == State.BYE:
                sh_timer_end = time.time() + 2.0

            # BYE -> IDLE로 떠날 때: 세션 그룹/키 정리
            if state == State.BYE and new_state == State.IDLE:
                sh_session_group = None

                SESSION_USER = None
                USER_SWITCHED = False

                # 새 키 미리 만들기
                new_key = f"usercheck_group_input_{st.session_state.group_key_counter + 1}"

                # 예전 그룹 위젯 키들 정리 (새 키는 유지)
                for k in list(st.session_state.keys()):
                    if k.startswith("usercheck_group_input_") and k != new_key:
                        st.session_state.pop(k, None)

                # 카운터+현재키 갱신
                st.session_state.group_key_counter += 1
                st.session_state.current_group_key = new_key

                # 새 키로 입력창을 다시 만들기 위해 rerun
                st.rerun()

            state = new_state

        # UI panel
        render_state_panel(state)

        # draw overlays
        display_frame = sh_frame
        if sh_bbox:
            x, y, w, h = sh_bbox
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), sh_color, 2)
            cv2.putText(display_frame, sh_message, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, sh_color, 2)
        else:
            cv2.putText(display_frame, sh_message, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, sh_color, 2)

        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h0, w0, _ = display_frame.shape
        new_h = int(h0 * (width / w0))
        resized = cv2.resize(display_frame, (int(width), new_h))
        frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        frame_slot.image(
                frame_rgb,
                channels="RGB",
                caption="Live",
                use_container_width=True,
                output_format="JPEG",
        )
        del resized, frame_rgb, display_frame

        # debug info
        debug_ph.json({
                "FACE_DETECTED"     : FACE_DETECTED,
                "USER_EXIST"        : USER_EXIST,
                "ENROLL_SUCCESS"    : ENROLL_SUCCESS,
                "VAD"               : VAD,
                "VAD_TASK_RUNNING"  : VAD_TASK_RUNNING,
                "ASR_TASK_RUNNING"  : ASR_TASK_RUNNING,
                "BYE_EXIST"         : BYE_EXIST,
                "TIMER_EXPIRED"     : TIMER_EXPIRED,
                "current_user"      : sh_current_user,
                "session_group"     : sh_session_group,  # ← 세션 한정 그룹 표시 (DB 저장 안 함)
                "audio_file"        : sh_audio_file,
                "tts_file"          : sh_tts_file,
                "bbox_avg_n"        : BBOX_AVG_N,
                "len(_bbox_history)": len(_bbox_history),
                "id(whisper_model)" : id(whisper_model),
                "id(resnet)"        : id(resnet),
                "id(face_detection)": id(face_detection),
        })

        if frame_i % 30 == 0:
            gc.collect()
            frame_i = 0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # FPS 제한 (예: 30fps)
        elapsed = time.time() - t0
        sleep_for = max(0.0, (1 / 30) - elapsed)
        time.sleep(sleep_for)
        run = st.session_state.get("_toggle_run", True)

    # cleanup
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
    cv2.destroyAllWindows()
    frame_slot.empty()
    st.info("카메라를 종료했습니다.")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
