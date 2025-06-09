#!/usr/bin/env python3
"""
Startup script for Cheating Detection Server
This script handles dependency installation and server startup
"""

import subprocess
import sys
import os
import time

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_ngrok():
    """Check if ngrok is available"""
    try:
        subprocess.check_output(["ngrok", "version"], stderr=subprocess.STDOUT)
        print("✅ Ngrok is available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  Ngrok not found in PATH")
        print("   The server will still work locally, but you'll need ngrok for external access")
        print("   Download from: https://ngrok.com/download")
        return False

def check_model_files():
    """Check if required model files exist"""
    model_files = [
        "model/shape_predictor_68_face_landmarks.dat",
        "model/best_yolov12.pt"
    ]
    
    missing_files = []
    for file_path in model_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("⚠️  Missing model files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nSome detection features may not work properly.")
        print("Please ensure all model files are in the 'model' directory.")
        return False
    
    print("✅ All model files found")
    return True

def setup_environment():
    """Setup environment variables and configuration"""
    print("\n🔧 Setting up environment...")
    
    # Create .env file if it doesn't exist
    if not os.path.exists(".env"):
        print("Creating .env file...")
        with open(".env", "w") as f:
            f.write("# Cheating Detection System Configuration\n")
            f.write("# Add your Google API key here for speech recognition\n")
            f.write("# GOOGLE_API_KEY=your_api_key_here\n")
            f.write("\n")
            f.write("# Audio Detection Settings\n")
            f.write("AUDIO_DETECTION_COOLDOWN=2.0\n")
            f.write("AUDIO_PHRASE_TIME_LIMIT=5.0\n")
            f.write("AUDIO_TIMEOUT=1.0\n")
            f.write("\n")
            f.write("# Detection Thresholds\n")
            f.write("SUSPICIOUS_SCORE_THRESHOLD=3\n")
            f.write("BEHAVIOR_CONFIDENCE_THRESHOLD=0.6\n")
        print("✅ Created .env file")
    
    # Create directories
    os.makedirs("log", exist_ok=True)
    os.makedirs("model", exist_ok=True)
    print("✅ Created necessary directories")

def start_server():
    """Start the Flask server"""
    print("\n🚀 Starting Cheating Detection Server...")
    print("=" * 60)

    try:
        # Import and run the server
        import server
        print("Server module imported successfully")

        # Initialize detection systems
        server.initialize_detection_systems()

        # Start ngrok tunnel
        port = 5000
        ngrok_url = server.start_ngrok_tunnel(port)

        if ngrok_url:
            print(f"\n{'='*60}")
            print(f"🚀 Cheating Detection Server Started!")
            print(f"{'='*60}")
            print(f"Local URL:  http://localhost:{port}")
            print(f"Public URL: {ngrok_url}")
            print(f"{'='*60}")
            print(f"Use the Public URL in your Android app")
            print(f"{'='*60}\n")
        else:
            print(f"\n⚠️  Ngrok tunnel failed to start. Server running locally on port {port}")

        # Start the Flask-SocketIO server
        server.socketio.run(server.app, host='0.0.0.0', port=port, debug=False)

    except ImportError as e:
        print(f"❌ Failed to import server module: {e}")
        return False
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        return False

    return True

def main():
    """Main startup function"""
    print("🔍 Cheating Detection System - Server Startup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Setup environment
    setup_environment()
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Failed to install dependencies. Please check your internet connection and try again.")
        return
    
    # Check ngrok
    check_ngrok()
    
    # Check model files
    check_model_files()
    
    print("\n" + "=" * 60)
    print("🎯 Pre-flight checks completed!")
    print("=" * 60)
    
    # Start server
    start_server()

if __name__ == "__main__":
    main()
