from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64


app = Flask(__name__)
socketio = SocketIO(app)
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Frame skipping parameter
frame_skip = 3
frame_counter = 0

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('image')
def image(data_image):
    global frame_counter
    frame_counter += 1
    
    if frame_counter % frame_skip != 0:
        return  # Skip this frame
    
    print("Received image data")
    # Decode the image from base64
    img_data = data_image.split(",")[1]
    img = base64.b64decode(img_data)
    npimg = np.frombuffer(img, dtype=np.uint8)
    frame = cv2.imdecode(npimg, 1)

    size = frame.shape
    print(size)
    # reduce the size of the frame to speed up the detection
    frame = cv2.resize(frame, (int(size[1] / 1.5), int(size[0] / 1.5)))
    print(frame.shape)
    # Process the frame (e.g., convert to grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Encode the frame in JPEG format
    _, buffer = cv2.imencode('.jpg', img_rgb)
    frame_data = base64.b64encode(buffer).decode('utf-8')
    emit('response_back', 'data:image/jpeg;base64,' + frame_data)
    

if __name__ == '__main__':
    socketio.run(app, debug=True)
