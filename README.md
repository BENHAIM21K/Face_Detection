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

## Prerequisites
1. **Python Libraries**:
   - `opencv-python`, `dlib`, `numpy`, `face_recognition`, `boto3`, `keyboard`, `getpass`
   - Install dependencies using pip:
     ```bash
     pip install opencv-python dlib face_recognition boto3 keyboard
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
