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
# Build and run

## Install Dependencies
```console
pip install -r requirements.txt
```

## Clone
```console
git clone https://github.com/Kejvi21/object-detection.git
```

---

## Create a Virtual Environment (Recommended)

## Windows
```console
python -m venv venv
venv\Scripts\activate
```
---
## macOS  / Linux
```console
python3 -m venv venv
source venv/bin/activate
```
---
## Run the Application

Start the Flask server:

```console
python app.py
```
---

## Open the Web App

After running the server, open your browser and go to:
http://127.0.0.1:5000 

---

# How It Works

1. The user uploads an image through the web interface.
2. Flask receives the image and saves it in `static/uploads`.
3. The image is processed using **OpenCV and MobileNet-SSD**.
4. The model detects persons in the image.
5. Bounding boxes are drawn around detected persons.
6. The processed image is displayed back in the browser.
