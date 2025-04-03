import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import cv2
import os
import numpy as np
from PIL import Image
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AIVideoDetector(nn.Module):
    def __init__(self):
        super(AIVideoDetector, self).__init__()
        self.model = models.efficientnet_b0(pretrained=False)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.model(x)

def load_model(model_path):
    model = AIVideoDetector().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

model_path = "ai_video_detector.pth"
model = load_model(model_path) if os.path.exists(model_path) else None

def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return transform(image).unsqueeze(0).to(device)

def predict_video(model, video_path, frame_skip=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video.")
        return None

    frame_count = 0
    predictions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            processed_frame = preprocess_frame(frame)
            with torch.no_grad():
                output = model(processed_frame)
                _, predicted = torch.max(output, 1)
                predictions.append(predicted.item())

        frame_count += 1

    cap.release()

    ai_frames = predictions.count(0)
    real_frames = predictions.count(1)

    return "AI-Generated Video" if ai_frames > real_frames else "Real Video"

st.set_page_config(page_title="Centum Logics - AI Video Detection Tool", page_icon="🎥", layout="centered")

st.title('AI Video Detection Tool by Centum Logics')
st.markdown('### Welcome to the AI Video Detection Tool by **Centum Logics**')
st.markdown('Upload a video to identify whether it is **Real** or **AI-Generated**.')

file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

if file:
    st.video(file)
    st.markdown("**Uploaded Video:**")
    
    if model:
        st.write("Processing the video...")

        video_path = f"uploaded_video_{int(time.time())}.mp4"
        
        st.write(f"Saving the file to: {video_path}")

        with open(video_path, "wb") as f:
            f.write(file.getbuffer())

        if os.path.exists(video_path):
            st.write(f"File saved successfully at {video_path}")
            prediction = predict_video(model, video_path)
            st.success(f"The video has been successfully processed. Prediction: **{prediction}**")
        else:
            st.error(f"Error: Failed to save the video at {video_path}")
    else:
        st.error("Model not found. Please ensure the model is loaded correctly.")
else:
    st.warning("Please upload a video to proceed.")
