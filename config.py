"""
Configuration file for Cheating Detection System
Handles Google API key setup and other configuration options
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for the cheating detection system"""
    
    # Google Speech Recognition API Configuration
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    
    # Audio Detection Settings
    AUDIO_DETECTION_COOLDOWN = float(os.getenv('AUDIO_DETECTION_COOLDOWN', '2.0'))
    AUDIO_PHRASE_TIME_LIMIT = float(os.getenv('AUDIO_PHRASE_TIME_LIMIT', '5.0'))
    AUDIO_TIMEOUT = float(os.getenv('AUDIO_TIMEOUT', '1.0'))
    
    # Detection Thresholds
    SUSPICIOUS_SCORE_THRESHOLD = int(os.getenv('SUSPICIOUS_SCORE_THRESHOLD', '3'))
    BEHAVIOR_CONFIDENCE_THRESHOLD = float(os.getenv('BEHAVIOR_CONFIDENCE_THRESHOLD', '0.6'))
    
    # File Paths
    LOG_DIR = os.getenv('LOG_DIR', 'log')
    MODEL_DIR = os.getenv('MODEL_DIR', 'model')
    
    @classmethod
    def validate_config(cls):
        """Validate configuration and provide helpful messages"""
        issues = []
        
        if not cls.GOOGLE_API_KEY:
            issues.append("No Google API key found. Speech recognition will use free tier with limitations.")
        
        # Create directories if they don't exist
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        
        return issues
    
    @classmethod
    def print_config_status(cls):
        """Print current configuration status"""
        print("=== Cheating Detection System Configuration ===")
        print(f"Google API Key: {'✓ Configured' if cls.GOOGLE_API_KEY else '✗ Not configured (using free tier)'}")
        print(f"Audio Detection Cooldown: {cls.AUDIO_DETECTION_COOLDOWN}s")
        print(f"Suspicious Score Threshold: {cls.SUSPICIOUS_SCORE_THRESHOLD}")
        print(f"Behavior Confidence Threshold: {cls.BEHAVIOR_CONFIDENCE_THRESHOLD}")
        print(f"Log Directory: {cls.LOG_DIR}")
        print(f"Model Directory: {cls.MODEL_DIR}")
        
        issues = cls.validate_config()
        if issues:
            print("\n=== Configuration Issues ===")
            for issue in issues:
                print(f"⚠️  {issue}")
        
        print("=" * 50)

# Alternative configuration methods for different deployment scenarios

class GoogleAPIConfig:
    """Helper class for Google API configuration"""
    
    @staticmethod
    def setup_from_service_account(service_account_path):
        """
        Set up Google API using service account JSON file
        This method is for production environments
        """
        if os.path.exists(service_account_path):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_path
            return True
        return False
    
    @staticmethod
    def setup_from_api_key(api_key):
        """
        Set up Google API using direct API key
        This method is for development/testing
        """
        os.environ['GOOGLE_API_KEY'] = api_key
        return True
    
    @staticmethod
    def get_setup_instructions():
        """
        Return setup instructions for Google API
        """
        return """
=== Google API Key Setup Instructions ===

Method 1: Environment Variable (.env file) - RECOMMENDED
1. Create a .env file in the project root directory
2. Add your API key: GOOGLE_API_KEY=your_actual_api_key_here
3. The system will automatically load it

Method 2: System Environment Variable
1. Export the variable: export GOOGLE_API_KEY=your_actual_api_key_here
2. Or add it to your ~/.bashrc or ~/.zshrc

Method 3: Service Account (Production)
1. Download service account JSON from Google Cloud Console
2. Save it as google-credentials.json in project root
3. Set GOOGLE_APPLICATION_CREDENTIALS=./google-credentials.json

To get a Google API Key:
1. Go to Google Cloud Console (https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the Speech-to-Text API
4. Go to Credentials → Create Credentials → API Key
5. Copy the generated API key

Free Tier Limitations:
- Without API key: Limited requests per day
- With API key: 60 minutes free per month, then pay-per-use
        """

if __name__ == "__main__":
    # Print configuration status when run directly
    Config.print_config_status()
    print(GoogleAPIConfig.get_setup_instructions())