import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile

st.title("🎯 Object Detection and Tracking")
st.markdown("Real-time object detection using YOLOv8 and OpenCV")

# Load model
model = YOLO("yolov8n.pt")

# File uploader or webcam option
source_option = st.radio("Select Input Source:", ["Webcam", "Upload Video"])

if source_option == "Upload Video":
    video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
else:
    cap = cv2.VideoCapture(0)

# Display output
stframe = st.empty()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    annotated_frame = results[0].plot()
    stframe.image(annotated_frame, channels="BGR", use_column_width=True)

cap.release()
