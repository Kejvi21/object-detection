from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import os
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# MobileNet-SSD (Single Shot Detector)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt",
                               "MobileNetSSD_deploy.caffemodel")


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if not file:
            return redirect(request.url)

        filename = str(uuid.uuid4()) + ".jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Prepare and load and process image
        image = cv2.imread(filepath)

        # ----- Darkening the image for test (optional) ----- alpha->Kontrast, beta->Hälligkeit
        image = cv2.convertScaleAbs(image, alpha=1.0, beta=-120)  # darker


        (h, w) = image.shape[:2]

        # 1. Preprocessing
        blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)

        # 2. Output from the network
        net.setInput(blob)
        detections = net.forward()

        # 3. Draw bounding boxes and labels on the image
        for i in range(detections.shape[2]):
            # Accesses the confidence (2) value for the i-th detection.
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])
                if CLASSES[idx] == "person":
                    # Multiplication of response (%) with the w,h (pixel) of img to get pixels coordinates
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    # Converts the coordinates of the bounding box from floating-point numbers to integers
                    (startX, startY, endX, endY) = box.astype("int")
                    label = f"Person: {confidence:.2f}"
                    # Draws the rectangle around the detected object
                    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.putText(image, label, (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                elif CLASSES[idx] == "car":
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    label = f"Car: {confidence:.2f}"
                    cv2.rectangle(image, (startX, startY), (endX, endY), (255, 165, 0), 2)
                    cv2.putText(image, label, (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)

        output_path = os.path.join(app.config['UPLOAD_FOLDER'], "out_" + filename)
        cv2.imwrite(output_path, image)

        return render_template('index.html', result_image=output_path)

    return render_template('index.html', result_image=None)


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, port=8083)