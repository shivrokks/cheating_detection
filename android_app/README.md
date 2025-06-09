# 📱 Cheating Detection Android App

This Android app streams video from your mobile device to the cheating detection server via ngrok tunneling.

## 🚀 Quick Start

### Prerequisites
- Android device with Android 7.0+ (API level 24)
- Camera permission
- Internet connection
- Running cheating detection server with ngrok

### Installation Options

#### Option 1: Install Pre-built APK
1. Download the latest APK from the releases page
2. Enable "Install from Unknown Sources" in your Android settings
3. Install the APK file

#### Option 2: Build from Source
1. Install Android Studio
2. Open this project (`android_app` folder)
3. Connect your Android device or start an emulator
4. Click "Run" to build and install

## 📖 Usage Instructions

### 1. Setup Server Connection
1. Start the Python server on your computer
2. Copy the ngrok URL from the server console (e.g., `https://abc123.ngrok.io`)
3. Open the Android app
4. Paste the ngrok URL in the "Server URL" field
5. Tap "Test Connection" to verify connectivity

### 2. Start Detection
1. Tap "Start Detection" to open the camera
2. Position your face in the camera view
3. Wait for calibration to complete (5 seconds)
4. The app will start streaming video and showing detection results

### 3. Monitor Results
- **Green indicators**: Normal behavior
- **Red indicators**: Suspicious behavior detected
- **Detailed results**: Show specific detection information

## ⚙️ Settings

Access settings from the main screen to configure:

### Video Settings
- **Frame Quality** (30-100%): Higher quality = better detection but slower upload
- **Frame Interval** (0.5-5.0s): How often frames are sent to server

### Connection Settings
- **Auto Reconnect**: Automatically reconnect if connection is lost

### UI Settings
- **Show Detailed Results**: Display detailed detection information
- **Vibrate on Detection**: Vibrate when suspicious behavior is detected

## 🔧 Technical Details

### Architecture
```
Android Camera → Frame Capture → Base64 Encoding → WebSocket → Server
                                                              ↓
Android UI ← Detection Results ← JSON Response ← Processing ← Server
```

### Key Components
- **MainActivity**: Server configuration and connection testing
- **CameraActivity**: Video streaming and detection display
- **SettingsActivity**: App configuration
- **ApiService**: REST API and WebSocket communication

### Network Communication
- **REST API**: For health checks and single frame processing
- **WebSocket**: For real-time video streaming and results
- **Base64 Encoding**: For image transmission over JSON

### Permissions Required
- `CAMERA`: For video capture
- `INTERNET`: For server communication
- `RECORD_AUDIO`: For future audio streaming features
- `WRITE_EXTERNAL_STORAGE`: For saving logs (optional)

## 🐛 Troubleshooting

### Connection Issues
- **"Connection Failed"**: Check if server is running and ngrok URL is correct
- **"Server Error"**: Verify server is healthy using the test connection feature
- **Slow Performance**: Reduce frame quality or increase frame interval in settings

### Camera Issues
- **"Camera Permission Denied"**: Grant camera permission in Android settings
- **"Camera Initialization Failed"**: Restart the app or check if another app is using the camera

### Detection Issues
- **No Detection Results**: Ensure good lighting and face is visible
- **Inaccurate Results**: Wait for calibration to complete and maintain steady position

## 📱 Supported Devices

### Minimum Requirements
- Android 7.0 (API level 24)
- 2GB RAM
- Rear or front camera
- Internet connectivity

### Recommended Specifications
- Android 9.0+ for better camera performance
- 4GB+ RAM for smooth operation
- Good lighting conditions
- Stable WiFi connection

## 🔒 Privacy & Security

- Video frames are only sent to your configured server
- No data is stored on external servers
- All communication uses your ngrok tunnel
- Camera access is only active during detection sessions

## 🆘 Support

If you encounter issues:

1. **Check server logs** for error messages
2. **Verify ngrok tunnel** is active and accessible
3. **Test with different network** (WiFi vs mobile data)
4. **Check Android version compatibility**
5. **Review app permissions** in Android settings

For additional help, please create an issue in the GitHub repository with:
- Android version and device model
- Server logs
- Steps to reproduce the issue
- Screenshots if applicable
