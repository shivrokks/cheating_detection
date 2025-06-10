import speech_recognition as sr
import threading
import time
import queue
import numpy as np
from collections import deque
import cv2
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class AudioDetection:
    def __init__(self, device_index=None):  # Added device_index parameter
        self.recognizer = sr.Recognizer()
        # Use specified device_index if provided, otherwise use default
        self.microphone = sr.Microphone(device_index=device_index)
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.detection_thread = None
        self.last_detection_time = 0
        self.detection_cooldown = 2.0  # Seconds between detections
        
        # Get Google API key from environment
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        if self.google_api_key:
            print("Google API key loaded from environment")
        else:
            print("No Google API key found - using free tier (may have limitations)")
        
        # History of detected words/phrases
        self.audio_history = deque(maxlen=50)
        self.suspicious_detections = deque(maxlen=20)
        
        # Suspicious words/phrases that might indicate cheating
        self.suspicious_keywords = {
            # Answer-related words
            'answer': 3,
            'answers': 3,
            'solution': 3,
            'solutions': 3,
            'result': 2,
            'results': 2,
            
            # Question-related words
            'question': 2,
            'questions': 2,
            'problem': 2,
            'problems': 2,
            
            # Communication words
            'help': 2,
            'tell me': 3,
            'give me': 3,
            'send me': 3,
            'share': 2,
            'copy': 3,
            'paste': 3,
            
            # Choice-related words
            'option a': 4,
            'option b': 4,
            'option c': 4,
            'option d': 4,
            'choice a': 4,
            'choice b': 4,
            'choice c': 4,
            'choice d': 4,
            'first option': 3,
            'second option': 3,
            'third option': 3,
            'fourth option': 3,
            
            # Test-related suspicious phrases
            'correct answer': 4,
            'right answer': 4,
            'wrong answer': 3,
            'which one': 2,
            'what is': 2,
            'how do': 2,
            'can you': 2,
            'do you know': 3,
            
            # Common cheating phrases
            'quick': 2,
            'fast': 2,
            'hurry': 2,
            'urgent': 3,
            'deadline': 2,
            'time up': 2,
            'finish': 2,
            'submit': 2,
            
            # Technology-related
            'google': 3,
            'search': 3,
            'internet': 3,
            'browser': 3,
            'website': 2,
            'online': 2,
            'phone': 2,
            'mobile': 2,
            'device': 2,
            
            # Communication platforms
            'whatsapp': 4,
            'telegram': 4,
            'discord': 4,
            'zoom': 3,
            'teams': 3,
            'skype': 3,
            'call': 2,
            'message': 2,
            'text': 2,
            'chat': 2,
        }
        
        # Initialize microphone
        self._initialize_microphone()
    
    def _initialize_microphone(self):
        """Initialize microphone with noise adjustment"""
        try:
            with self.microphone as source:
                print("Adjusting microphone for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=2)  # Increased duration from 1 to 2
                print("Microphone initialized successfully")
        except Exception as e:
            print(f"Error initializing microphone: {e}")
            raise e  # Re-raise the exception to be caught by the caller
    
    def start_listening(self):
        """Start audio detection in a separate thread"""
        if not self.is_listening:
            self.is_listening = True
            self.detection_thread = threading.Thread(target=self._audio_detection_loop, daemon=True)
            self.detection_thread.start()
            print("Audio detection started")
    
    def stop_listening(self):
        """Stop audio detection"""
        self.is_listening = False
        if self.detection_thread:
            self.detection_thread.join(timeout=2)
        print("Audio detection stopped")
    
    def _audio_detection_loop(self):
        """Main audio detection loop running in separate thread"""
        while self.is_listening:
            try:
                # Listen for audio with timeout
                with self.microphone as source:
                    # Reduce timeout for more responsive detection
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                
                # Process audio in main thread
                self.audio_queue.put(audio)
                
            except sr.WaitTimeoutError:
                # Timeout is normal, continue listening
                continue
            except Exception as e:
                print(f"Audio detection error: {e}")
                time.sleep(0.5)
    
    def process_audio(self):
        # Initialize overall results for this call
        final_is_suspicious = False
        final_suspicion_score = 0
        final_detected_text = ""
        final_suspicious_words = []
        
        any_audio_transcribed_in_this_call = False # Flag to check if any speech was transcribed

        # Process all audio in queue
        while not self.audio_queue.empty():
            try:
                audio = self.audio_queue.get_nowait()
                text_this_chunk = ""
                
                # Convert speech to text
                try:
                    if self.google_api_key:
                        try:
                            # Try with API key first
                            text_this_chunk = self.recognizer.recognize_google(audio, key=self.google_api_key, language='en-US').lower()
                        except sr.RequestError as api_error:
                            print(f"API key recognition failed: {api_error}, falling back to free service")
                            # Fall back to free service
                            text_this_chunk = self.recognizer.recognize_google(audio, language='en-US').lower()
                    else:
                        text_this_chunk = self.recognizer.recognize_google(audio, language='en-US').lower()
                except sr.UnknownValueError:
                    # Could not understand audio for this chunk, continue to next item
                    continue 
                except sr.RequestError as e:
                    print(f"Speech recognition error for chunk: {e}")
                    continue # to next item

                # If text was transcribed from this chunk
                if text_this_chunk:
                    any_audio_transcribed_in_this_call = True # Mark that audio was heard
                    
                    if not final_detected_text: # Store the first detected text for the result
                        final_detected_text = text_this_chunk
                    
                    self.audio_history.append((time.time(), text_this_chunk))
                    
                    # Analyze text for suspicious content (keywords)
                    score_from_keywords, words_from_keywords = self._analyze_text_for_cheating(text_this_chunk)
                    final_suspicion_score = max(final_suspicion_score, score_from_keywords)
                    final_suspicious_words.extend(words_from_keywords)
                    
                    print(f"Detected speech: '{text_this_chunk}' (Keyword score: {score_from_keywords})")
                
            except queue.Empty:
                break
            except Exception as e: # General exception for queue processing
                print(f"Error processing audio queue item: {e}")
                continue
        
        # Core logic: if any audio was transcribed, it's considered "cheating"
        if any_audio_transcribed_in_this_call:
            final_is_suspicious = True
            final_suspicion_score = max(final_suspicion_score, 5) # Ensure score is at least 5

        # Check recent history for patterns ONLY if no direct audio was transcribed
        if not any_audio_transcribed_in_this_call:
            pattern_score = self._check_suspicious_patterns()
            if pattern_score >= 3: # Threshold for pattern to be suspicious
                final_is_suspicious = True 
                final_suspicion_score = max(final_suspicion_score, pattern_score)
                if not final_detected_text: 
                    final_detected_text = "Suspicious audio pattern detected"
        
        # Add to suspicious_detections list if overall suspicious
        if final_is_suspicious:
            current_time = time.time()
            if current_time - self.last_detection_time >= self.detection_cooldown:
                self.suspicious_detections.append({
                    'time': current_time,
                    'text': final_detected_text,
                    'score': final_suspicion_score,
                    'words': list(set(final_suspicious_words)) # Make words unique
                })
                self.last_detection_time = current_time
        
        return {
            'is_suspicious': final_is_suspicious,
            'suspicion_score': final_suspicion_score,
            'detected_text': final_detected_text,
            'suspicious_words': list(set(final_suspicious_words)), # Make words unique
            'recent_detections': len(self.suspicious_detections)
        }
    
    def _analyze_text_for_cheating(self, text):
        """Analyze text for cheating-related keywords"""
        suspicious_score = 0
        found_words = []
        
        # Check for exact keyword matches
        for keyword, score in self.suspicious_keywords.items():
            if keyword in text:
                suspicious_score += score
                found_words.append(keyword)
        
        # Check for additional patterns
        # Multiple question words
        question_words = ['what', 'how', 'which', 'where', 'when', 'why', 'who']
        question_count = sum(1 for word in question_words if word in text)
        if question_count >= 2:
            suspicious_score += 2
            found_words.append('multiple_questions')
        
        # Numbers that might be answer choices
        choice_numbers = ['1', '2', '3', '4', 'one', 'two', 'three', 'four', 'first', 'second', 'third', 'fourth']
        choice_count = sum(1 for choice in choice_numbers if choice in text)
        if choice_count >= 1 and any(word in text for word in ['option', 'choice', 'answer']):
            suspicious_score += 3
            found_words.append('answer_choice')
        
        # Urgency indicators
        urgency_words = ['quick', 'fast', 'hurry', 'urgent', 'now', 'immediately']
        urgency_count = sum(1 for word in urgency_words if word in text)
        if urgency_count >= 1:
            suspicious_score += 1
            found_words.append('urgency')
        
        return suspicious_score, found_words
    
    def _check_suspicious_patterns(self):
        """Check for suspicious patterns in recent audio history"""
        if len(self.audio_history) < 2:
            return 0
        
        recent_time = time.time() - 30  # Last 30 seconds
        recent_texts = [text for timestamp, text in self.audio_history if timestamp > recent_time]
        
        if len(recent_texts) < 2:
            return 0
        
        pattern_score = 0
        
        # Frequent question asking
        question_count = sum(1 for text in recent_texts if any(q in text for q in ['what', 'how', 'which', 'where']))
        if question_count >= 3:
            pattern_score += 2
        
        # Repeated answer-seeking
        answer_count = sum(1 for text in recent_texts if any(a in text for a in ['answer', 'solution', 'result']))
        if answer_count >= 2:
            pattern_score += 3
        
        # Multiple option references
        option_count = sum(1 for text in recent_texts if any(o in text for o in ['option', 'choice', 'a', 'b', 'c', 'd']))
        if option_count >= 2:
            pattern_score += 2
        
        return pattern_score

# Global audio detection instance
audio_detector = None

def list_microphones():
    """Prints a list of available microphone names and their indices."""
    print("Available microphones:")
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"  Mic #{index}: {name}")

def initialize_audio_detection(device_index=None):  # Added device_index parameter
    """Initialize global audio detection"""
    global audio_detector
    try:
        # Suppress ALSA error messages during initialization
        import sys
        from contextlib import redirect_stderr

        print("Initializing audio detection...")

        # Try to list microphones with error suppression
        try:
            with open(os.devnull, 'w') as devnull:
                with redirect_stderr(devnull):
                    list_microphones()  # Print available microphones
        except:
            print("Could not list microphones, but continuing...")

        print(f"Attempting to use microphone with index: {device_index if device_index is not None else 'Default'}")

        # Try to initialize audio detector with error suppression
        with open(os.devnull, 'w') as devnull:
            with redirect_stderr(devnull):
                audio_detector = AudioDetection(device_index=device_index)
                audio_detector.start_listening()

        print("Audio detection initialized successfully")
        return True
    except Exception as e:
        print(f"Failed to initialize audio detection: {e}")
        print("Audio detection will be disabled - system will continue without audio monitoring")
        return False

def process_audio_detection():
    """Process audio detection and return results"""
    global audio_detector
    
    if audio_detector is None:
        return {
            'is_suspicious': False,
            'suspicion_score': 0,
            'detected_text': '',
            'suspicious_words': [],
            'recent_detections': 0
        }
    
    return audio_detector.process_audio()

def cleanup_audio_detection():
    """Clean up audio detection resources"""
    global audio_detector
    if audio_detector:
        audio_detector.stop_listening()
        audio_detector = None

def draw_audio_info(frame, audio_result):
    """Draw audio detection information on frame"""
    y_offset = 360
    
    # Draw main audio status
    status_color = (0, 0, 255) if audio_result['is_suspicious'] else (0, 255, 0)
    
    if audio_result['is_suspicious']:
        # Check if the detected text is the specific pattern message
        if audio_result['detected_text'] and audio_result['detected_text'] != "Suspicious audio pattern detected":
            status_text = "CHEATING (Audio Detected)"
        elif audio_result['detected_text'] == "Suspicious audio pattern detected":
            status_text = "CHEATING (Audio Pattern)"
        else: # Fallback if is_suspicious is true but text is empty or unexpected (e.g. only keyword match from empty string)
             # With new logic, if is_suspicious is true, detected_text should ideally not be empty.
            status_text = "CHEATING (Suspicious Audio)"
    else:
        status_text = "Audio Normal"
        
    cv2.putText(frame, f"Audio: {status_text}", (20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    # Draw suspicion score
    if audio_result['suspicion_score'] > 0:
        cv2.putText(frame, f"Audio Score: {audio_result['suspicion_score']}", 
                    (20, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    # Draw recent detections count
    if audio_result['recent_detections'] > 0:
        cv2.putText(frame, f"Suspicious Audio Events: {audio_result['recent_detections']}", 
                    (20, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Draw detected text (if any)
    if audio_result['detected_text']:
        # Truncate long text
        text = audio_result['detected_text'][:50] + "..." if len(audio_result['detected_text']) > 50 else audio_result['detected_text']
        cv2.putText(frame, f"Last: '{text}'", (20, y_offset + 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame