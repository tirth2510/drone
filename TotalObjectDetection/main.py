from ultralytics import YOLO
import cv2
import os


# Load an official model
model = YOLO("yolov8n.pt")  # build a new model from scratch

# Perform tracking with the model
results = model.track(source="1.mp4", save=True)

