import cv2
import dlib
from scipy.spatial import distance as dist
from pytube import YouTube
import numpy as np
import pygame
import os

# Constants
DISPLAY_WIDTH = 800  # Display width for the enlarged frame
FRAME_SKIP = 2  # Process and display every N frames
OUTPUT_VIDEO_PATH = "output_video.avi"  # Output video file path

# Initialize variables
blink_counter = 0
is_blinking = False
eye_color = (0, 255, 0)  # Initialize eye color as green

# Helper function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Load the face detector and shape predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download this file from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

# Load the YouTube video
video_url = "https://www.youtube.com/watch?v=f-gLQ6UrMtk" #YOUTUBE LINK
yt = YouTube(video_url)
stream = yt.streams.filter(adaptive=True, file_extension="mp4").first()
video_file_path = "youtube_video.mp4"
stream.download(output_path="", filename="youtube_video.mp4") 

# Initialize pygame for sound playback
pygame.mixer.init()

# Set the path to the ding sound file in the same folder as the script
ding_sound_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ding.mp3") #Ding Sound file - download any sound and name it ding.mp3

# Open the ding sound file
ding_sound = pygame.mixer.Sound(ding_sound_path)

# Initialize VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 30.0, (DISPLAY_WIDTH, DISPLAY_WIDTH))

# Open the downloaded video
cap = cv2.VideoCapture(video_file_path)

frame_number = 0  # Track the frame number

while True:
    ret, frame = cap.read()  # Read the next frame

    if not ret:
        break

    frame_number += 1

    if frame_number % FRAME_SKIP != 0:  # Process and display every N frames
        continue

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame using dlib
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 48)]

        left_eye = landmarks[0:6]
        right_eye = landmarks[6:12]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        ear = (left_ear + right_ear) / 2.0

        # Dynamic thresholding based on the recent history of EAR values
        EAR_THRESHOLD = 0.2  # Initial threshold (you can adjust this)
        if not is_blinking:
            if ear < EAR_THRESHOLD:
                blink_counter += 1
                is_blinking = True
                eye_color = (0, 0, 255)  # Change eye color to red when blinking
                ding_sound.play()  # Play the ding sound on blink
        else:
            if ear >= EAR_THRESHOLD:
                is_blinking = False
                eye_color = (0, 255, 0)  # Reset eye color to green

        # Calculate the bounding box of the face
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # Calculate the region of interest (ROI) for zooming
        roi_x = max(0, x - (DISPLAY_WIDTH - w) // 2)
        roi_y = max(0, y - (DISPLAY_WIDTH - h) // 2)
        roi_w = min(DISPLAY_WIDTH, frame.shape[1] - roi_x)
        roi_h = min(DISPLAY_WIDTH, frame.shape[0] - roi_y)

        # Extract the zoomed-in frame
        zoomed_frame = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        # Draw a square around the detected face
        cv2.rectangle(zoomed_frame, (x - roi_x, y - roi_y), (x - roi_x + w, y - roi_y + h), (255, 0, 0), 2)

        # Draw small dots near the eyes (change color based on blink)
        for (eye_x, eye_y) in left_eye + right_eye:
            cv2.circle(zoomed_frame, (eye_x - roi_x, eye_y - roi_y), 2, eye_color, -1)

        # Display blink count on the zoomed frame
        cv2.putText(zoomed_frame, f'Blinks: {blink_counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Write the frame to the output video
        out.write(zoomed_frame)

        # Show the zoomed frame
        cv2.imshow('Zoomed Frame', zoomed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoWriter
out.release()

# Release the video and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()