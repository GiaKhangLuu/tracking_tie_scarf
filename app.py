import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
from time import time

st.title("My first Streamlit app")
st.write("Hello, world")

def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="example",
    video_frame_callback=callback,
    rtc_configuration={  # Add this line
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)
