import cv2
import mediapipe as mp
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import torchvision.transforms as transforms
import os
from enum import Enum
import torch
import numpy as np
import sounddevice as sd
from collections import deque
import time
import soundfile as sf
import uuid
import threading

# 전역(Globals & Flags 섹션) 어딘가에 추가
REC_THREAD = None
REC_DONE = False

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


# =========================
# VAD Recorder
# =========================

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
        self.THRESHOLD = 0.5  # TODO: fit to environment
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
    recorder = VADRecorder()
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

FACE_DETECTED = False
USER_EXIST = False
ENROLL_SUCCESS = False
VAD = False
BYE_EXIST = False
TIMER_EXPIRED = False  # WELCOME, BYE state's timer
ASR_TEXT = ""

# Shared between states
sh_face_crop = None
sh_bbox = None
sh_embedding = None
sh_current_user = None
sh_audio_file = None
sh_message = "Initializing..."
sh_color = (255, 255, 0)
sh_timer_end = 0
sh_prev_unkonw = None

# =========================
# Utils
# =========================

DB_PATH = "faces_db.npy"
SIM_THRESHOLD = 0.65


def _record_worker(timeout: int):
    """
    WELCOME 단계에서 비동기로 한 번만 녹음.
    완료되면 sh_audio_file, VAD, REC_DONE 갱신.
    """
    global sh_audio_file, VAD, REC_DONE
    filename = listen_and_record_speech(timeout=timeout)  # 내부에서 VADRecorder 실행
    sh_audio_file = filename
    VAD = bool(filename)
    REC_DONE = True


def _asr_worker(path: str):
    """
    ASR를 백그라운드에서 실행하고 결과를 플래그에 기록.
    """
    global ASR_TEXT, BYE_EXIST, ASR_DONE
    txt = asr_from_wav(path) if path else ""
    ASR_TEXT = txt or ""
    t = "".join(ASR_TEXT.split())
    BYE_EXIST = ("잘가" in t) or ("bye" in t.lower())
    ASR_DONE = True


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
# ASR (Whisper)
# =========================

import whisper

whisper_model = whisper.load_model("large-v3")


def asr_from_wav(file_path: str) -> str:
    print(f"./{file_path}",
          os.path.exists(f"./{file_path}"))
    result = whisper_model.transcribe(f"./{file_path}")
    print(result)
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
    # key kept for signature compatibility; not used
    global ENROLL_SUCCESS, sh_message, sh_color

    results = update_face_detection()

    if not FACE_DETECTED:
        if results.detections and len(results.detections) > 1:
            sh_message = f"{len(results.detections)} faces detected. Only one please."
            sh_color = (0, 0, 255)
        else:
            sh_message = "Please show your face to the camera for enrollment."
            sh_color = (255, 255, 0)
    else:
        sh_message = "Unknown user. Use the right panel to enroll."
        sh_color = (0, 255, 255)


def enter_welcome():
    global VAD, TIMER_EXPIRED, sh_message, sh_color, REC_THREAD, REC_DONE

    update_face_detection()  # 카메라는 계속 읽고 있어도 됨

    sh_message = f"Hi, {sh_current_user}!"
    sh_color = (0, 255, 0)

    # 인사 타이머가 끝나면 녹음을 시작(단, 한 번만)
    if time.time() > sh_timer_end:
        TIMER_EXPIRED = True

        # 녹음 스레드가 아직 없고, 완료도 안됐으면 시작
        if (REC_THREAD is None or not REC_THREAD.is_alive()) and not REC_DONE:
            REC_THREAD = threading.Thread(target=_record_worker, args=(5,), daemon=True)
            REC_THREAD.start()

        # UI 메시지: 녹음 중 안내 (루프 매 프레임 갱신)
        if not REC_DONE:
            sh_message = f"Listening... (up to 5s)"
        else:
            # 녹음 완료됨: 메시지만 바꿔두고, 전이는 state_transition에서 처리
            sh_message = "Audio captured!" if VAD else "No speech detected."


def enter_asr():
    global ASR_THREAD, ASR_DONE, sh_message, sh_color

    # 카메라/얼굴 검출은 계속 돈다 (UI 오버레이를 위해 필요)
    update_face_detection()

    # 최초 진입 시 스레드 시작
    if (ASR_THREAD is None or not ASR_THREAD.is_alive()) and not ASR_DONE and sh_audio_file:
        ASR_THREAD = threading.Thread(target=_asr_worker, args=(sh_audio_file,), daemon=True)
        ASR_THREAD.start()

    # UI 메시지
    if not ASR_DONE:
        sh_message = "Transcribing..."
        sh_color = (0, 255, 255)
    else:
        sh_message = "ASR done"
        sh_color = (0, 200, 0)


def enter_bye():
    global TIMER_EXPIRED, sh_message, sh_color

    update_face_detection()

    TIMER_EXPIRED = False
    sh_message = f"Bye, {sh_current_user}!"
    sh_color = (255, 0, 255)

    if time.time() > sh_timer_end:
        TIMER_EXPIRED = True


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
        # Stay in ENROLL until success; go IDLE only if face lost
        if ENROLL_SUCCESS:
            print("[Enroll Success] Reloading DB...")
            name_list, group_list, embeddings = load_db()
            return State.WELCOME
        sh_prev_unkonw = sh_embedding
        return State.IDLE if not FACE_DETECTED else State.ENROLL

    elif current_state == State.WELCOME:
        if TIMER_EXPIRED and REC_DONE:
            return State.ASR if VAD else State.IDLE
        return State.WELCOME

    elif current_state == State.ASR:
        if ASR_DONE:
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
# Model Init
# =========================

print("Loading models...")
resnet = InceptionResnetV1(pretrained='vggface2').eval()
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
preprocess = transforms.Compose([
        transforms.Resize((160, 160)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
name_list, group_list, embeddings = load_db()
print("Models loaded.")

# =========================
# Streamlit UI & Main Loop
# =========================

import streamlit as st

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

if "enroll_form_key" not in st.session_state:
    st.session_state.enroll_form_key = None


def open_camera(index: int, target_w: int):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
    return cap


# ENROLL UI lifecycle flags
ENROLL_UI_BUILT = False
enroll_face_ph = None

# Initial state
state = State.IDLE
st.caption("Starting state machine...")


# ENROLL submit helper (uses Streamlit; defined after importing st)
def ui_enroll_submit(new_name: str, new_group: str):
    global ENROLL_SUCCESS, name_list, group_list, embeddings, sh_current_user

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

        # Badge
        state_badge.markdown(f"**Current State:** :blue[{current_state.name}]")

        # --- 중요: 현재 상태가 ENROLL일 때는 enroll_slot을 비우지 않는다!
        if current_state != State.ENROLL:
            enroll_slot.empty()  # ENROLL을 벗어나는 순간에만 비움

        # 다른 상태 슬롯들은 매 프레임 초기화 가능
        if current_state != State.WELCOME:
            welcome_slot.empty()
        if current_state != State.ASR:
            asr_slot.empty()
        if current_state != State.BYE:
            bye_slot.empty()

        # Message
        with message_slot.container():
            st.markdown(f"**Message:** {sh_message}")

        # ENROLL UI (form created once, with unique key)
        if current_state == State.ENROLL:
            # ENROLL에 들어올 때마다 고유한 폼 키를 1회 생성
            if st.session_state.enroll_form_key is None:
                st.session_state.enroll_form_key = f"form_enroll_{uuid.uuid4().hex}"
                # 폼이 새로 만들어질 것이므로, 표시용 face placeholder도 새로 받음
                ENROLL_UI_BUILT = False
                enroll_face_ph = None

            if not ENROLL_UI_BUILT:
                ENROLL_UI_BUILT = True
                with enroll_slot.container():
                    st.info("알 수 없는 사용자입니다. 아래 폼으로 등록을 진행하세요.")
                    enroll_face_ph = st.empty()

                    form_key = st.session_state.enroll_form_key
                    # 입력 위젯 키도 폼 키에 종속시켜 중복 방지
                    with st.form(key=form_key, clear_on_submit=False):
                        new_name = st.text_input("이름", key=f"{form_key}_name")
                        new_group = st.text_input("그룹(선택)", key=f"{form_key}_group")
                        submitted = st.form_submit_button("등록하기", use_container_width=True)
                    if submitted:
                        ui_enroll_submit(new_name, new_group)
                        # 등록 직후에는 바로 전이되므로(UI상) 폼 키 정리(안전)
                        st.session_state.enroll_form_key = None

            # 얼굴 미리보기는 계속 갱신
            if enroll_face_ph is not None:
                if sh_face_crop is not None and sh_face_crop.size != 0:
                    face_rgb = cv2.cvtColor(sh_face_crop, cv2.COLOR_BGR2RGB)
                    enroll_face_ph.image(face_rgb, caption="등록할 얼굴", use_container_width=True)
                else:
                    enroll_face_ph.warning("얼굴이 감지되지 않았습니다. 카메라를 향해 한 명만 비춰주세요.")

        else:
            # ENROLL이 아니면 폼 리소스 정리 (다음 ENROLL 때 새 키/폼 생성)
            if ENROLL_UI_BUILT:
                ENROLL_UI_BUILT = False
                enroll_face_ph = None
            enroll_slot.empty()
            # 혹시 남아있는 폼 키가 있으면 제거
            st.session_state.enroll_form_key = None

        # WELCOME UI
        if current_state == State.WELCOME:
            with welcome_slot.container():
                st.success(f"Hi, **{sh_current_user}**! 잠시 후 음성 녹음을 시작합니다.")
                remain = max(0.0, sh_timer_end - time.time())
                pct = min(max(1.0 - (remain / 2.0), 0.0), 1.0)
                st.progress(pct, text="Greeting...")

        # ASR UI
        if current_state == State.ASR:
            with asr_slot.container():
                if not ASR_DONE:
                    st.info("Transcribing...")  # 진행중
                else:
                    st.success("ASR Completed")
                    if ASR_TEXT:
                        st.write(ASR_TEXT)
                    st.write(f"**BYE detected:** {'Yes' if BYE_EXIST else 'No'}")
                if sh_audio_file:
                    audio_slot.audio(sh_audio_file)

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

        # Key (kept for compatibility; not used for enroll)
        key = cv2.waitKey(1) & 0xFF

        # Call & transition
        call_state_fn(state, key)
        new_state = state_transition(state)

        if new_state != state:
            print(f"State Change: {state.name} -> {new_state.name}")

            # ENROLL '진입' 시 한 번만 초기화
            if new_state == State.ENROLL:
                ENROLL_SUCCESS = False
                USER_EXIST = False
                # 이전 폼 키가 남아있으면 지움 (안전)
                st.session_state.enroll_form_key = None

            # ENROLL '이탈' 시 폼 키 제거 (중복 방지)
            if state == State.ENROLL and new_state != State.ENROLL:
                st.session_state.enroll_form_key = None

            if new_state == State.WELCOME:
                sh_timer_end = time.time() + 2.0  # 2초 인사
                VAD = False
                REC_DONE = False
                REC_THREAD = None

            if new_state == State.ASR:
                ASR_DONE = False
                BYE_EXIST = False
                ASR_THREAD = None
                ASR_TEXT = ""

            if new_state == State.BYE:
                sh_timer_end = time.time() + 2.0

            state = new_state

        # Update state panel
        render_state_panel(state)

        # ── 영상 그리기/표시 (항상 표시: ASR 포함)
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
                    "FACE_DETECTED" : FACE_DETECTED,
                    "USER_EXIST"    : USER_EXIST,
                    "ENROLL_SUCCESS": ENROLL_SUCCESS,
                    "VAD"           : VAD,
                    "BYE_EXIST"     : BYE_EXIST,
                    "TIMER_EXPIRED" : TIMER_EXPIRED,
                    "current_user"  : sh_current_user,
                    "audio_file"    : sh_audio_file
            })

        time.sleep(0.01)
        run = st.session_state.get("_toggle_run", True)

    # Cleanup
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
        frame_slot.empty()
        st.info("카메라를 종료했습니다.")
