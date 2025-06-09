# 🔍 Cheating Detection System with Android App

A comprehensive real-time cheating detection system that monitors various behavioral patterns during exams or assessments. The system includes both a Python server for detection processing and an Android app for mobile video streaming via ngrok tunneling.

## 🌟 Features

### Core Detection Features
- **👁️ Eye Movement Detection**: Tracks gaze direction and detects suspicious eye movements
- **🗣️ Head Pose Estimation**: Monitors head position and detects rapid movements
- **📱 Mobile Device Detection**: Identifies mobile phones in the camera view
- **💬 Lip Movement Analysis**: Detects talking or whispering
- **😊 Facial Expression Recognition**: Analyzes facial expressions for signs of stress or confusion
- **👥 Person Detection**: Counts people in frame and detects new entries
- **📚 Object Detection**: Identifies suspicious objects like books, papers, or electronic devices
- **🎤 Audio Analysis**: Detects suspicious speech patterns and keywords
- **🧠 Behavior Analysis**: Machine learning model that combines all features for overall behavior assessment

### New Mobile Features
- **📱 Android App**: Native Android application for video streaming
- **🌐 Ngrok Tunneling**: Secure tunneling for remote server access
- **🔄 Real-time Streaming**: Live video streaming from mobile camera to server
- **📊 Live Results**: Real-time detection results displayed on mobile device
- **⚙️ Configurable Settings**: Adjustable video quality, frame rate, and detection sensitivity

## 🏗️ System Architecture

```
[Android App] ---> [Ngrok Tunnel] ---> [Python Server] ---> [Detection Modules]
     📱                  🌐                  🖥️                    🔍
```

## 📋 Requirements

### Server Requirements
- Python 3.7+
- OpenCV
- dlib
- NumPy
- scikit-learn
- PyTorch
- Ultralytics (YOLOv8)
- Flask
- Flask-SocketIO
- pyngrok

### Android App Requirements
- Android 7.0 (API level 24) or higher
- Camera permission
- Internet connection
- Minimum 2GB RAM recommended

## 🚀 Installation & Setup

### 1. Server Setup

1. **Clone this repository:**
   ```bash
   git clone https://github.com/yourusername/cheating-detection-system.git
   cd cheating-detection-system
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download required models:**
   - Download `shape_predictor_68_face_landmarks.dat` from dlib website and place in `model/` directory
   - YOLOv8 models will be downloaded automatically on first run

4. **Start the server:**
   ```bash
   python start_server.py
   ```

   Or directly:
   ```bash
   python server.py
   ```

### 2. Android App Setup

#### Option A: Build from Source
1. **Open Android Studio**
2. **Import the project** from `android_app/` directory
3. **Build and install** the APK on your device

#### Option B: Install Pre-built APK
1. Download the APK from releases
2. Enable "Unknown Sources" in Android settings
3. Install the APK

### 3. Ngrok Setup (Required for Remote Access)

1. **Download ngrok** from https://ngrok.com/download
2. **Sign up** for a free ngrok account
3. **Install ngrok** and add to your PATH
4. The server will automatically start ngrok tunnel

## Usage

### Running the Detection System

To run the standard detection system:

```
python main.py
```

The system will:
1. Calibrate to your head position (keep your head straight and looking at the screen for the first 5 seconds)
2. Start monitoring for suspicious behaviors
3. Save screenshots to the `log` directory when suspicious activities are detected

### Training the Behavior Analysis Model

To train the behavior analysis model with your own examples:

```
python train_behavior.py
```

During training mode:
- Press 'n' to mark the current behavior as normal
- Press 's' to mark the current behavior as suspicious
- Press 'q' to quit and save the collected data

The more examples you provide, the better the model will become at distinguishing between normal and suspicious behaviors.

## Components

- `main.py`: Main application that integrates all detection modules
- `eye_movement.py`: Detects eye movements and gaze direction
- `head_pose.py`: Tracks head position and detects rapid movements
- `lip_movement.py`: Detects lip movements and talking
- `facial_expression.py`: Analyzes facial expressions
- `person_detection.py`: Counts people and detects new entries
- `object_detection.py`: Detects suspicious objects
- `behavior_analysis.py`: Machine learning for behavior pattern analysis
- `behavior_training.py`: Script to run the system in training mode

## Customization

You can adjust the sensitivity of various detection components by modifying the threshold values in their respective files.

## License

[Your License Information]
