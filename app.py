import cv2
import time
import os
import urllib.parse
import getpass
import re
import face_recognition
import numpy as np
import boto3
from botocore.exceptions import NoCredentialsError
import keyboard
import traceback

# AWS S3 Configuration 
bucket_name = 'null'  
folder_path = 'null'  

# AWS credentials 
aws_access_key_id='null'
aws_secret_access_key='null'

# Initialize the S3 client with credentials
s3_client = boto3.client(
    's3',
    region_name='us-east-1',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

def upload_to_s3(bucket, file_name, object_name):
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
        print(f"Upload Successful: {object_name}")
    except FileNotFoundError:
        print("The file was not found")
    except NoCredentialsError:
        print("AWS credentials not available")
    except Exception as e:
        traceback.print_exc()
        print(f"An error occurred: {e}")

# Prompt the user for the camera's IP address, username, and password
ip_address = input("Enter the IP address of the camera: ")
username = getpass.getpass("Enter the camera username: ")
password = getpass.getpass("Enter the camera password: ")

# Encode the password to handle special characters in the RTSP URL
encoded_password = urllib.parse.quote(password)

# Construct the RTSP URL using the provided credentials
rtsp_url = f'rtsp://{username}:{encoded_password}@{ip_address}:554/stream1'

# Define the output folder where captured frames will be saved
output_folder = 'null'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to find the highest numbered image in the output folder
def get_highest_numbered_image(output_folder):
    files = os.listdir(output_folder)
    # Use regex to extract numbers from filenames like 'face_00001.jpg'
    numbers = [int(re.search(r'face_(\d+)\.jpg', file).group(1))
               for file in files if re.search(r'face_(\d+)\.jpg', file)]
    return max(numbers) if numbers else 0  # Return 0 if no files are found

# Initialize the image counter to the next available number
image_counter = get_highest_numbered_image(output_folder) + 1

# Load the pre-trained face detection model (Haar Cascade Classifier)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load known face encodings and their names from the 'known_faces' directory and its subdirectories
known_faces_dir = 'known_faces'  # Directory containing images of known faces
known_face_encodings = []        # List to store face encodings
known_face_names = []            # List to store corresponding names

print("Loading known faces...")
for root, dirs, files in os.walk(known_faces_dir):
    for filename in files:
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(root, filename)
            # Load the image file
            image = face_recognition.load_image_file(image_path)
            # Get face encodings from the image
            encodings = face_recognition.face_encodings(image)
            if len(encodings) > 0:
                # Store the first encoding found
                known_face_encodings.append(encodings[0])
                # Extract the name from the directory name
                name = os.path.basename(root)
                known_face_names.append(name)
                print(f"Loaded encoding for {name}")
            else:
                print(f"No face found in {filename}")

print("Connecting to the stream...")

# Initialize video capture from the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

# Check if the video capture was successful
if not cap.isOpened():
    print("Error: Could not connect to the stream.")
    exit()

print("Connected to the stream. Starting movement detection...")
print("Press 'q' to stop the script.")

# Initialize the background subtractor for motion detection
fgbg = cv2.createBackgroundSubtractorMOG2(
    history=1500, varThreshold=75, detectShadows=False)

# Cooldown time to wait after saving a face image (in seconds)
cooldown_time = 10  
last_saved_time = 0  # Timestamp of the last saved image

# Initialize the paused state
paused_until = 0  # Timestamp until which the program is paused

# Initialize head detection timestamp
head_detected_at = None  # Timestamp when a head was first detected

# Define thresholds for motion detection
min_contour_area = 2000     # Minimum area of contour to be considered motion
min_bbox_width = 50         # Minimum width of bounding box
min_bbox_height = 50        # Minimum height of bounding box

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    current_time = time.time()

    # Check if the program is paused
    if current_time < paused_until:
        # Display the video stream without processing
        cv2.imshow(f'{ip_address} stream with Movement Detection', frame)
        # Check for exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting program...")
            break
        if keyboard.is_pressed('q'):
            print("Exiting program...")
            break
        continue  # Skip processing and continue to the next frame

    # Convert the frame to grayscale for motion detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply background subtraction to detect moving objects
    fgmask = fgbg.apply(gray)

    # Apply morphological operations to reduce noise
    fgmask = cv2.dilate(fgmask, None, iterations=3)
    fgmask = cv2.erode(fgmask, None, iterations=3)

    # Threshold the mask to obtain a binary image
    _, thresh = cv2.threshold(fgmask, 244, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    movement_detected = False  # Flag to indicate if movement is detected
    head_detected = False      # Flag to indicate if a head is detected in the current frame

    # Iterate over each contour detected
    for contour in contours:
        # Calculate the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(contour)

        # Filter out small contours based on area and size thresholds
        if cv2.contourArea(contour) > min_contour_area and w > min_bbox_width and h > min_bbox_height:
            movement_detected = True

            # Define the region of interest (ROI) for face detection
            roi_gray = gray[y:y + h, x:x + w]    # Grayscale ROI
            roi_color = frame[y:y + h, x:x + w]  # Color ROI

            # Detect faces within the ROI using the Haar cascade classifier
            faces = face_cascade.detectMultiScale(
                roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

            # If faces are detected in the ROI
            if len(faces) > 0:
                head_detected = True
                # If this is the first time a head is detected, set head_detected_at
                if head_detected_at is None:
                    head_detected_at = current_time
                    print(f"Head detected at {time.ctime(head_detected_at)}. Waiting 5 seconds before recognition.")
                else:
                    # Check if 5 seconds have passed since head_detected_at
                    if current_time - head_detected_at >= 5:
                        # Proceed to perform face recognition
                        for (fx, fy, fw, fh) in faces:
                            # Draw a blue rectangle around the face
                            cv2.rectangle(roi_color, (fx, fy),
                                          (fx + fw, fy + fh), (255, 0, 0), 2)

                            # Extract the face image for recognition
                            face_image = roi_color[fy:fy + fh, fx:fx + fw]
                            # Convert the face image to RGB color space
                            face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

                            # Get the face encodings for the detected face
                            face_encodings = face_recognition.face_encodings(face_image_rgb)

                            if len(face_encodings) > 0:
                                # Use the first encoding found
                                face_encoding = face_encodings[0]
                                # Compare the face encoding with known faces
                                matches = face_recognition.compare_faces(
                                    known_face_encodings, face_encoding)
                                name = "Unknown Face"  # Default name if no match

                                # Calculate face distances to known faces
                                face_distances = face_recognition.face_distance(
                                    known_face_encodings, face_encoding)
                                if len(face_distances) > 0:
                                    # Find the best match index
                                    best_match_index = np.argmin(face_distances)
                                    if matches[best_match_index]:
                                        # Assign the name of the best match
                                        name = known_face_names[best_match_index]

                                # Put the name label above the face rectangle
                                cv2.putText(roi_color, name, (fx, fy - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                                # Save the image and pause if cooldown time has passed
                                if (current_time - last_saved_time) >= cooldown_time:
                                    # Format the image filename with leading zeros
                                    image_filename = f"face_{image_counter:05d}.jpg"
                                    local_image_path = os.path.join(output_folder, image_filename)
                                    # Save the current frame to the output folder
                                    cv2.imwrite(local_image_path, frame)
                                    print(f"{name} detected! Image saved as {image_filename}")

                                    # Upload the image to AWS S3
                                    s3_object_name = folder_path + image_filename
                                    upload_to_s3(bucket_name, local_image_path, s3_object_name)

                                    last_saved_time = current_time  # Reset cooldown timer
                                    image_counter += 1  # Increment image counter

                                    # Set the paused_until timestamp to current time plus 10 seconds
                                    paused_until = current_time + 10
                                    print(f"Pausing detection for 10 seconds until {time.ctime(paused_until)}.")
                                    # Reset head_detected_at after face recognition
                                    head_detected_at = None
                            else:
                                print("No face encodings found.")
            else:
                # If no faces are detected in this contour, continue to the next one
                continue

    # After processing all contours
    if not head_detected:
        head_detected_at = None  # Reset the head_detected_at timestamp

    # Display the video stream with detections
    cv2.imshow(f'{ip_address} stream with Movement Detection', frame)

    # Check if 'q' is pressed to exit the script (using OpenCV key listener)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting program...")
        break

    if keyboard.is_pressed('q'):
        print("Exiting program...")
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
print("Program terminated. Resources released.")
