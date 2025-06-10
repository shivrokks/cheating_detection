package com.cheatingdetection.app

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import androidx.core.app.ActivityCompat
import kotlinx.coroutines.*
import kotlin.math.abs
import kotlin.math.sqrt

class AudioDetector(private val context: Context) {
    
    companion object {
        private const val TAG = "AudioDetector"
        private const val SAMPLE_RATE = 44100
        private const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        private const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
        private const val BUFFER_SIZE_FACTOR = 2
        
        // Audio detection thresholds
        private const val TALKING_THRESHOLD = 1000.0 // Amplitude threshold for talking detection
        private const val SILENCE_DURATION_MS = 1000L // Duration of silence before stopping talking detection
        private const val ANALYSIS_INTERVAL_MS = 100L // How often to analyze audio (100ms)
    }
    
    private var audioRecord: AudioRecord? = null
    private var isRecording = false
    private var detectionJob: Job? = null
    private var bufferSize: Int = 0
    
    // Audio analysis variables
    private var isTalking = false
    private var lastTalkingTime = 0L
    private var audioLevelHistory = mutableListOf<Double>()
    private val maxHistorySize = 10
    
    // Callback for audio detection results
    var onAudioDetected: ((isTalking: Boolean, audioLevel: Double) -> Unit)? = null
    
    init {
        calculateBufferSize()
    }
    
    private fun calculateBufferSize() {
        bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)
        if (bufferSize == AudioRecord.ERROR || bufferSize == AudioRecord.ERROR_BAD_VALUE) {
            bufferSize = SAMPLE_RATE * BUFFER_SIZE_FACTOR
        } else {
            bufferSize *= BUFFER_SIZE_FACTOR
        }
        Log.d(TAG, "Audio buffer size: $bufferSize")
    }
    
    fun hasAudioPermission(): Boolean {
        return ActivityCompat.checkSelfPermission(
            context,
            Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED
    }
    
    fun startDetection(): Boolean {
        if (!hasAudioPermission()) {
            Log.e(TAG, "Audio permission not granted")
            return false
        }
        
        if (isRecording) {
            Log.w(TAG, "Audio detection already running")
            return true
        }
        
        try {
            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE,
                CHANNEL_CONFIG,
                AUDIO_FORMAT,
                bufferSize
            )
            
            if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
                Log.e(TAG, "AudioRecord initialization failed")
                return false
            }
            
            audioRecord?.startRecording()
            isRecording = true
            
            // Start audio analysis in a coroutine
            detectionJob = CoroutineScope(Dispatchers.IO).launch {
                audioDetectionLoop()
            }
            
            Log.d(TAG, "Audio detection started successfully")
            return true
            
        } catch (e: SecurityException) {
            Log.e(TAG, "Security exception when starting audio detection", e)
            return false
        } catch (e: Exception) {
            Log.e(TAG, "Error starting audio detection", e)
            return false
        }
    }
    
    fun stopDetection() {
        isRecording = false
        detectionJob?.cancel()
        
        try {
            audioRecord?.stop()
            audioRecord?.release()
            audioRecord = null
            Log.d(TAG, "Audio detection stopped")
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping audio detection", e)
        }
    }
    
    private suspend fun audioDetectionLoop() {
        val buffer = ShortArray(bufferSize)
        
        while (isRecording && !Thread.currentThread().isInterrupted) {
            try {
                val bytesRead = audioRecord?.read(buffer, 0, buffer.size) ?: 0
                
                if (bytesRead > 0) {
                    val audioLevel = calculateAudioLevel(buffer, bytesRead)
                    analyzeAudioLevel(audioLevel)
                    
                    // Notify callback on main thread
                    withContext(Dispatchers.Main) {
                        onAudioDetected?.invoke(isTalking, audioLevel)
                    }
                }
                
                // Wait before next analysis
                delay(ANALYSIS_INTERVAL_MS)
                
            } catch (e: Exception) {
                Log.e(TAG, "Error in audio detection loop", e)
                break
            }
        }
    }
    
    private fun calculateAudioLevel(buffer: ShortArray, length: Int): Double {
        var sum = 0.0
        for (i in 0 until length) {
            sum += abs(buffer[i].toDouble())
        }
        return sum / length
    }
    
    private fun analyzeAudioLevel(audioLevel: Double) {
        val currentTime = System.currentTimeMillis()
        
        // Add to history
        audioLevelHistory.add(audioLevel)
        if (audioLevelHistory.size > maxHistorySize) {
            audioLevelHistory.removeAt(0)
        }
        
        // Calculate average audio level
        val averageLevel = if (audioLevelHistory.size >= 3) {
            audioLevelHistory.average()
        } else {
            audioLevel
        }
        
        // Determine if talking based on threshold
        val isTalkingNow = audioLevel > TALKING_THRESHOLD
        
        if (isTalkingNow) {
            isTalking = true
            lastTalkingTime = currentTime
        } else {
            // Check if we should stop talking detection due to silence
            if (isTalking && (currentTime - lastTalkingTime) > SILENCE_DURATION_MS) {
                isTalking = false
            }
        }
        
        Log.v(TAG, "Audio level: $audioLevel, Average: $averageLevel, Talking: $isTalking")
    }
    
    fun getCurrentTalkingStatus(): Boolean {
        return isTalking
    }
    
    fun getCurrentAudioLevel(): Double {
        return audioLevelHistory.lastOrNull() ?: 0.0
    }
    
    // Adjust sensitivity dynamically
    fun adjustSensitivity(newThreshold: Double) {
        // This could be used to adjust the TALKING_THRESHOLD based on environment
        Log.d(TAG, "Audio sensitivity adjustment requested: $newThreshold")
    }
}
