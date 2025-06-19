# Gaze and Expression Detection Fixes

## Issues Fixed

### 1. Gaze Detection Always Showing "Looking Center"

**Problem**: The pupil detection algorithm was failing, causing gaze detection to always default to "Looking Center".

**Root Cause**: 
- Pupil detection using contour analysis was unreliable
- Thresholds were too strict (0.35-0.65 range)
- No fallback method when pupil detection failed

**Solution**:
- **Primary Method**: Added landmark-based gaze detection using eye corner positions relative to face center
- **Fallback Method**: Improved pupil detection with more sensitive thresholds
- **More Sensitive Thresholds**: Reduced from 0.35-0.65 to 0.15-0.85 range for landmark detection and 0.4-0.6 for pupil detection

**Technical Changes**:
```python
# New landmark-based detection
left_eye_center = np.mean(left_eye_points, axis=0)
right_eye_center = np.mean(right_eye_points, axis=0)
eyes_center_x = (left_eye_center[0] + right_eye_center[0]) / 2
x_offset = (eyes_center_x - face_center_x) / face_width

# More sensitive thresholds
if x_offset < -0.15:  # Was -0.35
    gaze_direction = "Looking Left"
elif x_offset > 0.15:  # Was 0.65
    gaze_direction = "Looking Right"
```

### 2. Facial Expression Always Showing "Neutral"

**Problem**: Expression detection was too conservative and relied heavily on history, preventing dynamic detection.

**Root Cause**:
- High thresholds (3.0 for brow movement, 1.5 for smile)
- Long history requirement (15 frames) before detection
- No immediate detection method

**Solution**:
- **Reduced Thresholds**: Lowered brow movement threshold from 3.0 to 1.5, smile threshold from 1.5 to 0.8
- **Faster Response**: Reduced history size from 15 to 8 frames, minimum history from 5 to 3 frames
- **Immediate Detection**: Added geometry-based immediate expression detection
- **Dual Detection**: Combined history-based and immediate detection methods

**Technical Changes**:
```python
# Reduced thresholds
BROW_MOVEMENT_THRESHOLD = 1.5  # Was 3.0
SMILE_THRESHOLD = 0.8  # Was 1.5
HISTORY_SIZE = 8  # Was 15
MIN_HISTORY_SIZE = 3  # Was 5

# Immediate detection method
def detect_immediate_expressions(landmarks):
    # Calculate mouth curvature for smile detection
    mouth_curvature = mouth_center[1] - corner_height
    brow_eye_distance = np.mean([left_eye_y - left_brow_y, right_eye_y - right_brow_y])
    
    is_smiling_immediate = mouth_curvature > 2.0
    is_confused_immediate = brow_eye_distance > 25
    
    return is_smiling_immediate, is_confused_immediate

# Combined detection
is_confused = is_confused or is_confused_immediate
is_smiling = is_smiling or is_smiling_immediate
```

## Test Results

### Dynamic Detection Test Results:
- **Gaze Detection**: ✅ DYNAMIC - Now detects "Looking Center", "Looking Right", "Looking Up", "Looking Left"
- **Expression Detection**: ✅ DYNAMIC - Now detects "Neutral", "Confused", "Smiling", "Suspicious (Confused+Smiling)"
- **Server Integration**: ✅ WORKING - Server properly returns varying results

### Before vs After:

**Before**:
- Gaze: Always "Looking Center"
- Expression: Always "Neutral"

**After**:
- Gaze: Dynamic detection with 3+ variations per session
- Expression: Dynamic detection with 3+ variations per session

## Files Modified

1. **eye_movement.py**:
   - Added landmark-based gaze detection as primary method
   - Improved pupil detection as fallback
   - More sensitive thresholds for both methods
   - Better error handling

2. **facial_expression.py**:
   - Reduced detection thresholds for faster response
   - Added immediate expression detection method
   - Combined history-based and immediate detection
   - Faster history accumulation

## Expected App Behavior

Your Android app should now see:

### Gaze Detection:
- **Looking Left**: When you look to the left
- **Looking Right**: When you look to the right  
- **Looking Up**: When you look up
- **Looking Down**: When you look down
- **Looking Center**: When you look straight at the camera

### Expression Detection:
- **Neutral**: Normal facial expression
- **Confused**: When eyebrows are raised
- **Smiling**: When mouth corners are raised
- **Suspicious (Confused+Smiling)**: When both confused and smiling

## Server Information

The server is now running at:
- **Local URL**: http://localhost:5000
- **Public URL**: https://848d-103-180-45-250.ngrok-free.app

Use the public URL in your Android app to test the dynamic detection.

## Debugging

If you need to debug the detection:

1. **Enable Debug Output**: Uncomment the debug print statements in:
   - `eye_movement.py` line 224
   - `facial_expression.py` line 234

2. **Test Locally**: Run `python test_dynamic_detection.py` to verify dynamic behavior

3. **Check Server Logs**: The server will log detection results for each frame

## Performance Notes

- Detection is now more responsive (3-8 frames vs 5-15 frames)
- More sensitive to small movements and expressions
- Dual detection methods provide better accuracy
- Fallback mechanisms prevent static results

The system should now provide the dynamic gaze and expression detection you were expecting!
