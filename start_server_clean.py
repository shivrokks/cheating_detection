#!/usr/bin/env python3
"""
Clean server startup script that suppresses ALSA warnings
"""

import os
import sys
import subprocess

# Suppress ALSA warnings
os.environ['ALSA_PCM_CARD'] = 'default'
os.environ['ALSA_PCM_DEVICE'] = '0'

# Redirect stderr to filter out ALSA messages
def filter_alsa_output():
    """Filter out ALSA error messages from stderr"""
    import sys
    import re
    
    class ALSAFilter:
        def __init__(self, stream):
            self.stream = stream
            self.alsa_patterns = [
                r'ALSA lib.*',
                r'Cannot connect to server.*',
                r'jack server is not running.*',
                r'JackShmReadWritePtr.*'
            ]
        
        def write(self, data):
            # Filter out ALSA messages
            for pattern in self.alsa_patterns:
                if re.search(pattern, data):
                    return
            self.stream.write(data)
        
        def flush(self):
            self.stream.flush()
    
    sys.stderr = ALSAFilter(sys.stderr)

# Apply the filter
filter_alsa_output()

print("🔇 ALSA warnings suppressed")
print("🚀 Starting Cheating Detection Server...")
print("=" * 60)

# Now import and run the server
try:
    import server
    
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
    
except Exception as e:
    print(f"❌ Server startup failed: {e}")
    sys.exit(1)
