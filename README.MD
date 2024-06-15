# Real Time Face Detection with OpenCV and Flask

This project demonstrates real-time face detection using OpenCV and Flask. It captures live video from the user's camera, processes it to detect faces using OpenCV's Haar Cascade Classifier, and streams the video with annotated face detections back to the client using Flask and Socket.IO.

## Features

- **Real-time Face Detection:** Uses OpenCV to detect faces in live video streams.
- **Web-based Interface:** Built with Flask and Bootstrap for a responsive user interface.
- **WebSocket Communication:** Utilizes Socket.IO for real-time video streaming and communication between the server and client.

## Prerequisites

Before running this project, ensure you have the following installed:

- Python (version 3.6 or higher)
- Flask (`pip install flask`)
- OpenCV (`pip install opencv-python`)
- Flask-SocketIO (`pip install flask-socketio`)

## Getting Started

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Tinny-Robot/Real-Time-Face-Detection-with-Opencv-and-Flask.git

   cd Real-Time-Face-Detection-with-Opencv-and-Flask
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**

   ```bash
   python app.py
   ```

4. **Open your browser:**

   Open your web browser and go to `http://localhost:5000` to see real-time face detection in action.

## Contributing

Contributions are welcome! If you'd like to improve this project or add new features, feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- This project is a beginner-friendly introduction to using OpenCV for real-time computer vision applications.
- Built with Python, Flask, OpenCV, and Socket.IO.
