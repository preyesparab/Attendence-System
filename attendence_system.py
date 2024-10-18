import cv2
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime, time
from keras.models import load_model

# Set the time range for the model to work
START_TIME = time(9, 30)
END_TIME = time(10, 0)

# Labels for emotions
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def is_within_time_range():
    """Check if the current time is within the allowed time range."""
    now = datetime.now().time()
    return START_TIME <= now <= END_TIME

def load_student_images(student_images):
    """Load student images and encode their faces."""
    known_face_encodings = []
    known_face_names = []
    for name, img_path in student_images.items():
        image = face_recognition.load_image_file(img_path)
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(name)
    return known_face_encodings, known_face_names

def detect_students_and_emotions(known_face_encodings, known_face_names, emotion_model):
    """Detect students, recognize emotions, and mark attendance."""
    video_capture = cv2.VideoCapture(0)  # Use webcam
    student_attendance = {}  # Track attendance
    attendance_data = []  # List to store attendance data

    while True:
        if not is_within_time_range():
            print("Out of time range")
            break

        ret, frame = video_capture.read()

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all face locations and face encodings in the frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # Loop through each face
        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # If a match was found in known_face_encodings, get the student's name
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            # Detect emotion
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_crop = gray_frame[face_location[0]:face_location[2], face_location[3]:face_location[1]]
            face_crop = cv2.resize(face_crop, (48, 48))
            face_crop = np.expand_dims(np.expand_dims(face_crop, -1), 0) / 255.0

            emotion_prediction = emotion_model.predict(face_crop)
            emotion_label = emotion_labels[np.argmax(emotion_prediction)]

            # Mark attendance and store data with time
            if name != "Unknown":
                student_attendance[name] = "Present"
                attendance_data.append([name, emotion_label, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    return attendance_data

def save_attendance_to_file(attendance_data):
    """Save attendance data to a CSV file."""
    df = pd.DataFrame(attendance_data, columns=['Name', 'Emotion', 'Timestamp'])
    df.to_csv('attendance.csv', index=False)
    print("Attendance saved to attendance.csv")

# Load the pre-trained emotion detection model
emotion_model = load_model('path_to_emotion_model.h5')

# Example student images with paths
student_images = {
    "John": "path_to_john_image.jpg",
    "Jane": "path_to_jane_image.jpg"
}

# Load student faces
known_face_encodings, known_face_names = load_student_images(student_images)

# Run the detection system
attendance_data = detect_students_and_emotions(known_face_encodings, known_face_names, emotion_model)

# Save the attendance data
save_attendance_to_file(attendance_data)
