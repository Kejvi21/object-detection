# person_detector.py

import cv2
import numpy as np

# Load class labels from MobileNet SSD
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# Load the pre-trained model files
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt",
                               "MobileNetSSD_deploy.caffemodel")

# Load image
image = cv2.imread("persons.jpeg")
(h, w) = image.shape[:2]

# Preprocess the image for the neural network
blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)

# Set input and perform forward pass
net.setInput(blob)
detections = net.forward()

# Loop over the detections
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    # Filter weak detections
    if confidence > 0.5:
        idx = int(detections[0, 0, i, 1])

        if CLASSES[idx] == "person":
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw bounding box
            label = f"Person: {confidence:.2f}"
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(image, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show the output image
cv2.imshow("Person Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()