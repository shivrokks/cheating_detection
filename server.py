#!/usr/bin/env python3
"""
Flask Server for Cheating Detection System
Provides REST API endpoints for Android app to stream video and receive detection results
"""

import cv2
import numpy as np
import base64
import json
import time
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import threading
from pyngrok import ngrok
import logging

# Import your existing detection modules
from eye_movement import process_eye_movement
from head_pose import process_head_pose
from mobile_detection import process_mobile_detection
from lip_movement import process_lip_movement
from facial_expression import process_facial_expression
from person_detection import process_person_detection
from object_detection import process_object_detection
from behavior_analysis import process_behavior_analysis, load_training_data
# Removed audio_detection import - now handled client-side

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'cheating_detection_secret_key'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables for detection state
detection_state = {
    'calibrated_angles': None,
    'calibration_start_time': None,
    'is_calibrating': True,
    'audio_initialized': False,
    'last_detection_results': {}
}

# Initialize detection systems
def initialize_detection_systems():
    """Initialize all detection systems"""
    global detection_state
    
    # Create log directory
    os.makedirs("log", exist_ok=True)
    
    # Load behavior training data
    load_training_data()
    
    # Audio detection is now handled client-side (Android app)
    detection_state['audio_initialized'] = True  # Always true since it's handled by client
    logger.info("Audio detection will be handled client-side")
    
    # Set calibration start time
    detection_state['calibration_start_time'] = time.time()
    
    logger.info("Detection systems initialized")

def decode_base64_image(base64_string):
    """Decode base64 image string to OpenCV format"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64 to bytes
        img_bytes = base64.b64decode(base64_string)
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        
        # Decode to OpenCV image
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return frame
    except Exception as e:
        logger.error(f"Error decoding base64 image: {e}")
        return None

def encode_image_to_base64(frame):
    """Encode OpenCV image to base64 string"""
    try:
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{img_base64}"
    except Exception as e:
        logger.error(f"Error encoding image to base64: {e}")
        return None

def process_frame_detection(frame, client_talking_status=False, client_audio_level=0.0):
    """Process a single frame through all detection systems"""
    global detection_state
    
    try:
        # Initialize default values
        head_direction = "Looking at Screen"
        gaze_direction = "Looking Center"
        mobile_detected = False
        is_talking = False
        facial_expression = "Neutral"
        person_count = 1
        multiple_people = False
        new_person_entered = False
        suspicious_objects = False
        detected_objects = []
        
        # Process eye movement
        frame, gaze_direction = process_eye_movement(frame)
        logger.info(f"Eye movement result: {gaze_direction}")
        
        # Process head pose with calibration
        current_time = time.time()

        # Initialize calibration_start_time if not set
        if detection_state['calibration_start_time'] is None:
            detection_state['calibration_start_time'] = current_time

        if detection_state['is_calibrating']:
            if current_time - detection_state['calibration_start_time'] <= 5:
                # Still calibrating
                if detection_state['calibrated_angles'] is None:
                    _, detection_state['calibrated_angles'] = process_head_pose(frame, None)
            else:
                # Calibration complete
                detection_state['is_calibrating'] = False
                logger.info("Head pose calibration completed")
        
        if not detection_state['is_calibrating'] and detection_state['calibrated_angles'] is not None:
            frame, head_direction = process_head_pose(frame, detection_state['calibrated_angles'])
            logger.info(f"Head pose result: {head_direction}")
        
        # Process mobile detection
        frame, mobile_detected = process_mobile_detection(frame)
        logger.info(f"Mobile detection result: {mobile_detected}")

        # Process lip movement detection
        frame, is_talking = process_lip_movement(frame)
        logger.info(f"Lip movement result: {is_talking}")

        # Process facial expression detection
        frame, facial_expression = process_facial_expression(frame)
        logger.info(f"Facial expression result: {facial_expression}")
        
        # Process person detection
        frame, person_count, multiple_people, new_person_entered = process_person_detection(frame)
        
        # Process object detection
        frame, suspicious_objects, detected_objects = process_object_detection(frame)
        
        # Use client-side audio detection results
        audio_result = {
            'is_suspicious': client_talking_status,  # Consider talking as potentially suspicious
            'suspicion_score': 1 if client_talking_status else 0,
            'detected_text': 'Talking detected by client' if client_talking_status else '',
            'suspicious_words': [],
            'recent_detections': 1 if client_talking_status else 0,
            'audio_level': client_audio_level
        }

        # Override is_talking with client-provided status
        is_talking = client_talking_status
        logger.info(f"Client audio status - Talking: {client_talking_status}, Level: {client_audio_level}")
        
        # Prepare data for behavior analysis
        head_data = {
            'direction': head_direction,
            'rapid_movement': head_direction == "Rapid Movement",
            'pitch': getattr(process_head_pose, 'last_pitch', 0),
            'yaw': getattr(process_head_pose, 'last_yaw', 0),
            'roll': getattr(process_head_pose, 'last_roll', 0)
        }
        
        eye_data = {'direction': gaze_direction}
        lip_data = {'is_talking': is_talking}
        expression_data = {'expression': facial_expression}
        person_data = {
            'count': person_count,
            'multiple_people': multiple_people,
            'new_person': new_person_entered
        }
        object_data = {
            'suspicious_objects': suspicious_objects,
            'detected_objects': detected_objects
        }
        
        # Process behavior analysis
        frame, behavior_result = process_behavior_analysis(
            head_data, eye_data, lip_data, expression_data, 
            person_data, object_data, frame, audio_result
        )
        
        # Compile detection results
        detection_results = {
            'timestamp': current_time,
            'calibrating': detection_state['is_calibrating'],
            'gaze_direction': gaze_direction,
            'head_direction': head_direction,
            'mobile_detected': mobile_detected,
            'is_talking': is_talking,
            'facial_expression': facial_expression,
            'person_count': person_count,
            'multiple_people': multiple_people,
            'new_person_entered': new_person_entered,
            'suspicious_objects': suspicious_objects,
            'detected_objects': detected_objects,
            'audio_result': audio_result,
            'behavior': behavior_result.get('behavior', 'Normal'),
            'behavior_confidence': behavior_result.get('confidence', 0.0),
            'overall_suspicious': (
                mobile_detected or 
                multiple_people or 
                suspicious_objects or 
                audio_result.get('is_suspicious', False) or
                (behavior_result.get('behavior') == 'Suspicious' and behavior_result.get('confidence', 0) > 0.6)
            )
        }
        
        # Store last detection results
        detection_state['last_detection_results'] = detection_results
        
        return frame, detection_results
        
    except Exception as e:
        import traceback
        logger.error(f"Error processing frame: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return frame, {'error': str(e), 'timestamp': time.time()}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'calibrating': detection_state['is_calibrating'],
        'audio_initialized': detection_state['audio_initialized']
    })

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Process a single frame and return detection results"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode the image
        frame = decode_base64_image(data['image'])
        if frame is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Extract client audio data (if provided)
        client_talking = data.get('is_talking', False)
        client_audio_level = data.get('audio_level', 0.0)

        # Process the frame
        processed_frame, results = process_frame_detection(frame, client_talking, client_audio_level)
        
        # Encode processed frame back to base64 (optional)
        processed_image_b64 = None
        if data.get('return_processed_image', False):
            processed_image_b64 = encode_image_to_base64(processed_frame)
        
        response = {
            'success': True,
            'results': results,
            'processed_image': processed_image_b64
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in process_frame endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_status', methods=['GET'])
def get_status():
    """Get current detection status"""
    return jsonify({
        'success': True,
        'status': detection_state['last_detection_results'],
        'calibrating': detection_state['is_calibrating'],
        'audio_initialized': detection_state['audio_initialized']
    })

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    logger.info('Client connected via WebSocket')
    emit('status', {
        'connected': True,
        'calibrating': detection_state['is_calibrating'],
        'audio_initialized': detection_state['audio_initialized']
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    logger.info('Client disconnected from WebSocket')

@socketio.on('process_frame')
def handle_frame_processing(data):
    """Handle frame processing via WebSocket"""
    try:
        if 'image' not in data:
            emit('error', {'message': 'No image data provided'})
            return

        # Extract client audio data
        client_talking = data.get('is_talking', False)
        client_audio_level = data.get('audio_level', 0.0)

        # Decode and process frame
        frame = decode_base64_image(data['image'])
        if frame is None:
            emit('error', {'message': 'Invalid image data'})
            return

        processed_frame, results = process_frame_detection(frame, client_talking, client_audio_level)

        # Send results back
        emit('detection_results', {
            'success': True,
            'results': results,
            'timestamp': time.time()
        })

    except Exception as e:
        logger.error(f"WebSocket frame processing error: {e}")
        emit('error', {'message': str(e)})

def start_ngrok_tunnel(port=5000):
    """Start ngrok tunnel and return public URL"""
    try:
        # Kill any existing ngrok processes
        ngrok.kill()
        
        # Start ngrok tunnel
        public_url = ngrok.connect(port)
        logger.info(f"Ngrok tunnel started: {public_url}")
        
        return public_url
    except Exception as e:
        logger.error(f"Failed to start ngrok tunnel: {e}")
        return None

if __name__ == '__main__':
    # Initialize detection systems
    initialize_detection_systems()
    
    # Start ngrok tunnel
    port = 5000
    ngrok_url = start_ngrok_tunnel(port)
    
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
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
