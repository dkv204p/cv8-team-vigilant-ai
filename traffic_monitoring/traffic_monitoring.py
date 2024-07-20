import cv2
import numpy as np

# Paths to YOLO files
cfg_path = 'yolo/yolov3.cfg'
weights_path = 'yolo/yolov3.weights'
names_path = 'yolo/coco.names'

# Load YOLO
net = cv2.dnn.readNet(weights_path, cfg_path)

# Get the names of the output layers
layer_names = net.getLayerNames()
out_layer_indices = net.getUnconnectedOutLayers()
output_layers = [layer_names[i - 1] for i in out_layer_indices]

# Load class names
with open(names_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize video capture (replace with 0 for webcam input or a video file path)
video_path = './traffic_monitoring/traffic_video_2.mp4'

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Video capture could not be opened.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    height, width, channels = frame.shape

    # Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Information to capture
    class_ids = []
    confidences = []
    boxes = []

    # Process each detection
    for out in outs:
        for detection in out:
            # Each detection is an array of shape (85,)
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.8:  # Confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression to remove duplicates
    if boxes:
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)  # Green color for bounding box
            if label in ['car', 'truck', 'bus', 'motorbike']:  # Detect only vehicles
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    else:
        print("No boxes detected.")

    # Display the result
    cv2.imshow('Traffic Surveillance', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()