import json
import boto3

def lambda_handler(event, context):
    # Initialize clients with the correct region (us-east-1)
    s3_client = boto3.client('s3', region_name='us-east-1')
    rekognition_client = boto3.client('rekognition', region_name='us-east-1')
    sns_client = boto3.client('sns', region_name='us-east-1')

    # Define your SNS topic ARN
    sns_topic_arn = 'null'  # Replace with your actual SNS Topic ARN

    # Get bucket and object key from the event
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    # Ensure the file is in the right folder (uploads/)
    if key.startswith('/uploads'):
        try:
            # Generate a pre-signed URL for the image with inline content disposition
            presigned_url = s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': bucket,
                    'Key': key,
                    'ResponseContentDisposition': 'inline'  # Suggest opening the image in the browser
                },
                ExpiresIn=3600  # URL expires in 1 hour
            )

            # Call Rekognition to detect labels in the image
            rekognition_response = rekognition_client.detect_labels(
                Image={'S3Object': {'Bucket': bucket, 'Name': key}},
                MaxLabels=10,
                MinConfidence=80
            )

            # Process the detected labels
            labels = [f"{label['Name']}: {label['Confidence']:.2f}%" for label in rekognition_response['Labels']]
            label_results = "\n".join(labels)
            print(f"Detected labels for {key}: {label_results}")

            # Prepare the message content
            email_subject = f"Rekognition Analysis Results for {key}"
            email_body = (
                f"The image '{key}' has been analyzed. Here are the results:\n\n{label_results}\n\n"
                f"You can view the image using the following link (valid for 1 hour):\n{presigned_url}"
            )

            # Publish the message to the SNS topic
            sns_response = sns_client.publish(
                TopicArn=sns_topic_arn,
                Subject=email_subject,
                Message=email_body
            )

            return {
                'statusCode': 200,
                'body': json.dumps({
                    'file': key,
                    'labels': labels,
                    'sns_status': 'Message sent successfully',
                    'presigned_url': presigned_url
                })
            }

        except Exception as e:
            print(f"Error processing file {key}: {str(e)}")
            return {
                'statusCode': 500,
                'body': json.dumps(f"Error processing file {key}: {str(e)}")
            }

    else:
        return {
            'statusCode': 400,
            'body': json.dumps('File is not in the uploads folder.')
        }
