# 🚀 Complete Setup Guide - Cheating Detection System

This guide will walk you through setting up the complete cheating detection system with Android app and ngrok tunneling.

## 📋 Prerequisites

### System Requirements
- **Computer**: Windows, macOS, or Linux with Python 3.7+
- **Android Device**: Android 7.0+ with camera
- **Internet**: Stable connection for both computer and mobile device
- **Storage**: At least 2GB free space for models and dependencies

### Accounts Needed
- **Ngrok Account**: Free account at https://ngrok.com (for remote access)
- **Google Cloud** (Optional): For enhanced speech recognition

## 🔧 Step 1: Server Setup

### 1.1 Clone the Repository
```bash
git clone https://github.com/yourusername/cheating-detection-system.git
cd cheating-detection-system
```

### 1.2 Install Python Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 1.3 Download Required Models
1. **Face Landmarks Model**:
   - Download `shape_predictor_68_face_landmarks.dat` from [dlib website](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
   - Extract and place in `model/` directory

2. **YOLO Models** (will download automatically on first run):
   - `yolov8n.pt` for object detection
   - Custom mobile detection model

### 1.4 Setup Ngrok
1. **Download Ngrok**:
   - Visit https://ngrok.com/download
   - Download for your operating system
   - Extract and add to your PATH

2. **Create Ngrok Account**:
   - Sign up at https://ngrok.com
   - Get your auth token from the dashboard

3. **Configure Ngrok**:
   ```bash
   ngrok authtoken YOUR_AUTH_TOKEN
   ```

### 1.5 Configure Environment (Optional)
Create a `.env` file for advanced configuration:
```env
# Google API key for speech recognition (optional)
GOOGLE_API_KEY=your_google_api_key_here

# Detection thresholds
SUSPICIOUS_SCORE_THRESHOLD=3
BEHAVIOR_CONFIDENCE_THRESHOLD=0.6

# Audio settings
AUDIO_DETECTION_COOLDOWN=2.0
```

## 🖥️ Step 2: Start the Server

### 2.1 Quick Start
```bash
python start_server.py
```

### 2.2 Manual Start
```bash
python server.py
```

### 2.3 Verify Server
The console should show:
```
🚀 Cheating Detection Server Started!
============================================================
Local URL:  http://localhost:5000
Public URL: https://abc123.ngrok.io
============================================================
Use the Public URL in your Android app
============================================================
```

**Important**: Copy the Public URL (ngrok URL) - you'll need it for the Android app!

## 📱 Step 3: Android App Setup

### Option A: Install Pre-built APK
1. Download the APK from the releases page
2. Enable "Install from Unknown Sources" in Android settings
3. Install the APK

### Option B: Build from Source
1. **Install Android Studio**:
   - Download from https://developer.android.com/studio
   - Install with default settings

2. **Open Project**:
   - Open Android Studio
   - Select "Open an existing project"
   - Navigate to `android_app/` folder

3. **Build and Install**:
   - Connect your Android device via USB
   - Enable "Developer Options" and "USB Debugging"
   - Click "Run" button in Android Studio

## 🔗 Step 4: Connect App to Server

### 4.1 Configure Server URL
1. Open the Android app
2. In the "Server URL" field, paste your ngrok URL
3. Example: `https://abc123.ngrok.io`

### 4.2 Test Connection
1. Tap "Test Connection"
2. You should see "✅ Server connection successful!"
3. If it fails, check:
   - Server is running
   - Ngrok URL is correct
   - Internet connection is stable

### 4.3 Grant Permissions
When prompted, grant the following permissions:
- **Camera**: Required for video capture
- **Microphone**: For future audio features
- **Storage**: For saving logs (optional)

## 🎥 Step 5: Start Detection

### 5.1 Begin Detection
1. Tap "Start Detection" in the app
2. Position your face in the camera view
3. Keep your head straight during calibration (5 seconds)

### 5.2 Monitor Results
- **Green indicators**: Normal behavior
- **Red indicators**: Suspicious behavior
- **Detailed view**: Shows specific detection results

### 5.3 Adjust Settings
Access settings to configure:
- Frame quality (30-100%)
- Frame interval (0.5-5.0 seconds)
- Auto-reconnect options
- UI preferences

## 🔧 Step 6: Troubleshooting

### Common Server Issues

**"Module not found" errors**:
```bash
pip install -r requirements.txt
```

**"Ngrok not found"**:
- Ensure ngrok is in your PATH
- Try running `ngrok version` to verify

**"Port already in use"**:
- Kill existing processes on port 5000
- Or modify the port in `server.py`

### Common Android Issues

**"Connection failed"**:
- Verify server is running
- Check ngrok URL is correct
- Try different network (WiFi vs mobile data)

**"Camera permission denied"**:
- Go to Android Settings > Apps > Cheating Detection > Permissions
- Enable Camera permission

**"Poor detection accuracy"**:
- Ensure good lighting
- Keep face centered in camera
- Wait for calibration to complete

### Performance Optimization

**Slow detection**:
- Reduce frame quality in app settings
- Increase frame interval
- Ensure stable internet connection

**High CPU usage on server**:
- Close other applications
- Consider using GPU acceleration if available

## 📊 Step 7: Testing & Validation

### 7.1 Test Server Endpoints
```bash
python test_server.py https://your-ngrok-url.ngrok.io
```

### 7.2 Validate Detection
Test various scenarios:
- Normal behavior (looking at screen)
- Suspicious behavior (looking away, talking)
- Multiple people in frame
- Mobile phone detection

### 7.3 Monitor Logs
- Server logs appear in console
- Detection screenshots saved in `log/` directory
- Android app shows real-time status

## 🔒 Security Considerations

### Network Security
- Ngrok provides HTTPS encryption
- No data stored on external servers
- All processing happens on your server

### Privacy
- Video frames only sent to your server
- No permanent storage of video data
- Audio processing is optional

### Access Control
- Ngrok URLs are randomly generated
- Consider using ngrok password protection for production

## 🆘 Getting Help

### Check Logs
1. **Server logs**: Console output from Python server
2. **Android logs**: Use `adb logcat` or Android Studio
3. **Ngrok logs**: Available in ngrok web interface

### Common Solutions
1. **Restart everything**: Server, ngrok, and Android app
2. **Check firewall**: Ensure ports 5000 and ngrok port are open
3. **Update dependencies**: Run `pip install -r requirements.txt --upgrade`

### Report Issues
If you encounter problems:
1. Check this guide first
2. Review error messages carefully
3. Test with the provided test script
4. Create an issue on GitHub with:
   - System information
   - Error messages
   - Steps to reproduce

## 🎯 Next Steps

Once everything is working:
1. **Customize detection**: Adjust thresholds in detection modules
2. **Train behavior model**: Use `train_behavior.py` for custom training
3. **Add features**: Extend the system with additional detection methods
4. **Deploy**: Consider cloud deployment for production use

## 📚 Additional Resources

- **Ngrok Documentation**: https://ngrok.com/docs
- **Android Development**: https://developer.android.com
- **OpenCV Tutorials**: https://opencv.org/tutorials/
- **Flask Documentation**: https://flask.palletsprojects.com/

---

🎉 **Congratulations!** You now have a complete cheating detection system with mobile app support!
