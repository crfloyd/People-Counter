# utils/video.py
import cv2

def get_video_capture(source=0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise Exception("Unable to open video source")
    return cap
