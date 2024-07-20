import cv2
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from datetime import datetime

# angle, range, false, upscale

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the trained model
model = load_model('./wanted_person/wanted_persons_model.keras')

# Define the wanted person class ID (assuming the wanted person has class ID 1)
wanted_person_class_id = 1

# Initialize video capture
cap = cv2.VideoCapture(0)  # 0 for the first connected camera

def send_email(image_path, location, gps_coordinates):
    from_email = 'devkkverma123@gmail.com'
    to_email = 'devkkverma123@gmail.com'
    subject = 'Wanted Person Detected'
    body = f'A wanted person was detected at the following location:\n{location}\nGPS Coordinates: {gps_coordinates}'
    
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    
    msg.attach(MIMEText(body, 'plain'))
    
    attachment = open(image_path, 'rb')
    part = MIMEBase('application', 'octet-stream')
    part.set_payload(attachment.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', f'attachment; filename={image_path}')
    
    msg.attach(part)
    
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(from_email, 'fpyg gfrz pqdo bdoo')  # Replace with your App Password
    text = msg.as_string()
    server.sendmail(from_email, to_email, text)
    server.quit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale (Haar Cascade works with grayscale images)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Extract the region of interest (the face)
        face = frame[y:y+h, x:x+w]
        
        # Preprocess the face for prediction
        resized_face = cv2.resize(face, (224, 224))
        img_array = img_to_array(resized_face)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        
        # Draw a red box if the wanted person is detected
        if predicted_class == wanted_person_class_id:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            label = f'Wanted Person ID: {predicted_class}'
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            # Save the detected face image
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            image_path = f'./wanted_person/detected_faces/wanted_person_{timestamp}.jpg'
            cv2.imwrite(image_path, face)
            
            # Simulated GPS coordinates (replace with real coordinates if available)
            gps_coordinates = 'Latitude: 37.7749, Longitude: -122.4194'  # Example coordinates
            
            # Send email with the image and location
            location = f'X: {x}, Y: {y}, Width: {w}, Height: {h}'
            send_email(image_path, location, gps_coordinates)
            
        else:
            # Draw a green box for other detected faces
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f'Person ID: {predicted_class}'
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the result using OpenCV
    cv2.imshow('Security Surveillance', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
