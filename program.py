import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
from skimage.feature import hog
from scipy.spatial.distance import cdist
import glob
import os

# Load Configuration
DATABASE_FOLDER = "C:\\Users\\Admin\\Desktop\\New folder\\"
PHOTOS_FOLDER = "photos/"
FINGERPRINT_EXT = ".tif"
CONNECTION_MAPPING = {'tata': "102_1", 'Alakh': "102_2", 'Ishan': "102_3"}
VIDEO_CAPTURE_INDEX = 0

# Initialize Face Recognition
video_capture = cv2.VideoCapture(VIDEO_CAPTURE_INDEX)

# Load face data
face_data = {
    "Alakh": os.path.join(PHOTOS_FOLDER, "Alakh.jpg"),
    "tata": os.path.join(PHOTOS_FOLDER, "tata.jpg"),
    "Varun": os.path.join(PHOTOS_FOLDER, "Varun.jpg"),
    "Ishan": os.path.join(PHOTOS_FOLDER, "Ishan.jpg"),
}

known_face_encodings = []
known_faces_names = []

for name, path in face_data.items():
    try:
        image = face_recognition.load_image_file(path)
        encoding = face_recognition.face_encodings(image)
        if encoding:
            known_face_encodings.append(encoding[0])
            known_faces_names.append(name)
    except Exception as e:
        print(f"Error loading face data for {name}: {e}")

students = known_faces_names.copy()

# Setup CSV for attendance
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
csv_file_path = f"{current_date}.csv"

# Fingerprint Processing Functions
def preprocess_fingerprint(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}. Please check the file path.")
    image = cv2.equalizeHist(image)
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image

def extract_features(image):
    resized_image = cv2.resize(image, (128, 128))
    features, _ = hog(resized_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, channel_axis=None)
    return features.reshape(1, -1)

def match_fingerprints(query_features, database_features):
    database_features = np.array(database_features).reshape(len(database_features), -1)
    distances = cdist(query_features, database_features, metric='euclidean')
    best_match_idx = np.argmin(distances)
    best_match_distance = distances[0, best_match_idx]
    return best_match_idx, best_match_distance

# Preload fingerprint data
file_paths = glob.glob(os.path.join(DATABASE_FOLDER, f"*{FINGERPRINT_EXT}"))
database_features = []
database_filenames = []

for file_path in file_paths:
    try:
        preprocessed_image = preprocess_fingerprint(file_path)
        features = extract_features(preprocessed_image)
        database_features.append(features)
        database_filenames.append(os.path.basename(file_path))
    except ValueError as e:
        print(e)

# Main Loop
with open(csv_file_path, "w+", newline="") as f:
    lnwriter = csv.writer(f)
    lnwriter.writerow(["Name", "Time", "Fingerprint File"])  # Add header to CSV

    while True:
        _, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distance)

            if matches[best_match_index]:
                name = known_faces_names[best_match_index]

                if name in students:
                    students.remove(name)
                    current_time = now.strftime("%H:%M:%S")

                    # Fingerprint Matching
                    fingerprint_file = ""
                    if name in CONNECTION_MAPPING:
                        try:
                            query_image_path = os.path.join(DATABASE_FOLDER, f"{CONNECTION_MAPPING[name]}{FINGERPRINT_EXT}")
                            query_image = preprocess_fingerprint(query_image_path)
                            query_features = extract_features(query_image)
                            best_match_idx, best_match_distance = match_fingerprints(query_features, database_features)
                            fingerprint_file = database_filenames[best_match_idx]
                            print(f"Fingerprint Match: {fingerprint_file} with distance: {best_match_distance}")
                        except ValueError as e:
                            print(e)

                    # Write to CSV
                    lnwriter.writerow([name, current_time, fingerprint_file])

                # Display on video feed
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10, 100)
                fontScale = 1.5
                fontColor = (255, 0, 0)
                thickness = 3
                lineType = 2
                cv2.putText(frame, name + " Present", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()