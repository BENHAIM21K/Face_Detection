# Face and Motion Detection with AWS Integration

This project captures frames containing movement and human heads from an RTSP camera stream, checks if the detected face matches known faces, and categorizes them as "known" or "unknown." It then uploads these frames to an AWS S3 bucket. A Lambda function analyzes the frames using AWS Rekognition and sends email notifications with the analysis results.

---

## Features
- **Movement Detection**: Detects motion using background subtraction.
- **Face Detection and Recognition**: Identifies faces using a Haar Cascade and compares them to known faces using `face_recognition`.
- **Frame Upload to AWS S3**: Uploads captured frames to an S3 bucket for further processing.
- **AWS Rekognition Analysis**: Analyzes frames for labels and confidence scores using AWS Rekognition.
- **Email Notification via AWS SNS**: Sends results and a pre-signed S3 URL via email.

---


## Architecture

The architecture of the project is as follows:

![Project Architecture](assets/MovementDetectionArchitechture.png.png)

1. **RTSP Camera**:
   - Streams video over RTSP.
   - Frames with detected motion are captured locally.

2. **Python Script**:
   - Processes video frames to detect motion and faces.
   - Recognizes known faces and labels unknown faces.
   - Saves the captured frames locally and uploads them to an AWS S3 bucket.

3. **AWS S3 Bucket**:
   - Stores the uploaded images in a specific folder (`uploads/`).
   - Triggers an AWS Lambda function upon new uploads.

4. **AWS Lambda**:
   - Analyzes the images using AWS Rekognition for additional insights.
   - Sends the analysis results and a pre-signed URL via AWS SNS to an email address.

## Prerequisites
1. **Python Libraries**:
   - `opencv-python`, `dlib`, `numpy`, `face_recognition`, `boto3`, `keyboard`, `getpass`
   - Install dependencies using pip:
     ```bash
     pip install -r requirements.txt
     ```

2. **Camera**:
   - A camera supporting RTSP streams.
   - Ensure the RTSP stream URL is accessible.

3. **AWS Services**:
   - **S3 Bucket**:
     - Create a bucket and configure a folder named `uploads/`.
     - Set a lifecycle policy to delete images after 30 days (optional).
   - **AWS Rekognition**: Ensure permissions for Rekognition analysis.
   - **AWS SNS**: Create an SNS topic and note its ARN.

4. **System Requirements**:
   - This script is designed specifically for Windows.
   - Requires **cmake** for compiling dependencies like `dlib`.
