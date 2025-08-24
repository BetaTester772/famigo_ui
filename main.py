# app.py
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

if run:
    # 카메라 열기
    if st.session_state.cap is None or not st.session_state.cap.isOpened():
        st.session_state.cap = open_camera(int(cam_index))
        if not st.session_state.cap.isOpened():
            st.error("카메라를 열 수 없습니다. 인덱스를 바꾸거나 다른 앱을 종료해보세요.")
            st.stop()

    # 프레임 루프
    # Streamlit은 위젯 상호작용 시 스크립트를 다시 실행하므로
    # while 루프 안에서 짧게 sleep을 주고, 토글(run)이 꺼지면 깨끗이 종료합니다.
    while run:
        ok, frame = st.session_state.cap.read()
        if not ok:
            st.warning("프레임을 읽을 수 없습니다.")
            break

        if flip:
            frame = cv2.flip(frame, 1)

        # 표시용 RGB 변환 (OpenCV는 BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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
else:
    # 꺼진 상태에서 cap이 열려 있으면 닫기
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
        frame_slot.empty()
