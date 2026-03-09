# Person Recognition Web App

A simple web application for detecting **people in images** using **OpenCV** and the **MobileNet-SSD deep learning model**.  
The application allows users to upload an image through a web interface and automatically highlights detected people.

This project was developed as part of a **Computer Vision / Machine Learning course project**.

---

# Features

- Upload images through a web interface
- Detect people in images using **MobileNet-SSD**
- Draw bounding boxes around detected persons
- Display the processed image in the browser
- Clean and simple web interface

---

# Technologies Used

- Python
- Flask
- OpenCV
- MobileNet-SSD (Caffe model)
- HTML / CSS

---

- **app.py** – Flask web server
- **person_detector.py** – person detection logic using OpenCV
- **MobileNetSSD_deploy.caffemodel** – pretrained deep learning model
- **MobileNetSSD_deploy.prototxt** – model architecture
- **templates/index.html** – web interface
- **static/uploads/** – folder where uploaded images are stored

---

# Requirements

Before running the project make sure you have installed:

- Python 3.9 or newer
- pip
- Git

---

# Install Dependencies
```console
pip install -r requirements.txt
```