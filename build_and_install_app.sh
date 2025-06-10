#!/bin/bash

echo "🔨 Building and Installing Cheating Detection Android App"
echo "========================================================="

# Navigate to android app directory
cd android_app

# Make gradlew executable
chmod +x gradlew

# Clean and build the app
echo "🧹 Cleaning previous build..."
./gradlew clean

echo "🔨 Building debug APK..."
./gradlew assembleDebug

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    
    # Check if device is connected
    if adb devices | grep -q "device$"; then
        echo "📱 Installing APK to connected device..."
        adb install -r app/build/outputs/apk/debug/app-debug.apk
        
        if [ $? -eq 0 ]; then
            echo "✅ App installed successfully!"
            echo "🚀 You can now launch the app on your device"
        else
            echo "❌ Installation failed"
        fi
    else
        echo "⚠️  No Android device connected via ADB"
        echo "📁 APK built at: android_app/app/build/outputs/apk/debug/app-debug.apk"
        echo "📱 You can manually install this APK on your device"
    fi
else
    echo "❌ Build failed"
    exit 1
fi

echo "========================================================="
