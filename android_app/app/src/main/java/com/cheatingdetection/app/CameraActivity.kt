package com.cheatingdetection.app

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Bundle
import android.util.Base64
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.cheatingdetection.app.databinding.ActivityCameraBinding
import io.socket.client.IO
import io.socket.client.Socket
import org.json.JSONObject
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response
import java.io.ByteArrayOutputStream
import java.net.URISyntaxException
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

class CameraActivity : AppCompatActivity() {
    
    private lateinit var binding: ActivityCameraBinding
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var apiService: ApiService
    private var serverUrl: String = ""
    
    // WebSocket connection
    private var socket: Socket? = null
    private var isConnected = false
    
    // Camera components
    private var imageCapture: ImageCapture? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    
    // Detection state
    private var isDetectionActive = false
    private var lastFrameTime = 0L
    private val frameInterval = 1000L // Send frame every 1 second
    
    companion object {
        private const val TAG = "CameraActivity"
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityCameraBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        // Get server URL from intent
        serverUrl = intent.getStringExtra("server_url") ?: ""
        if (serverUrl.isEmpty()) {
            Toast.makeText(this, "No server URL provided", Toast.LENGTH_SHORT).show()
            finish()
            return
        }
        
        // Initialize API service
        apiService = ApiService.create(serverUrl)
        
        // Initialize camera executor
        cameraExecutor = Executors.newSingleThreadExecutor()
        
        setupUI()
        startCamera()
        connectWebSocket()
    }
    
    private fun setupUI() {
        binding.textViewServerUrl.text = "Server: $serverUrl"
        
        // Start/Stop detection button
        binding.buttonToggleDetection.setOnClickListener {
            if (isDetectionActive) {
                stopDetection()
            } else {
                startDetection()
            }
        }
        
        // Back button
        binding.buttonBack.setOnClickListener {
            finish()
        }
        
        // Initially disable detection until camera is ready
        binding.buttonToggleDetection.isEnabled = false
    }
    
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        
        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
            
            // Preview
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.viewFinder.surfaceProvider)
            }
            
            // Image capture
            imageCapture = ImageCapture.Builder().build()
            
            // Image analyzer for frame processing
            imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, FrameAnalyzer())
                }
            
            // Select front camera
            val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA
            
            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()
                
                // Bind use cases to camera
                camera = cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture, imageAnalyzer
                )
                
                // Enable detection button
                binding.buttonToggleDetection.isEnabled = true
                updateConnectionStatus("Camera ready")
                
            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
                Toast.makeText(this, "Camera initialization failed", Toast.LENGTH_SHORT).show()
            }
            
        }, ContextCompat.getMainExecutor(this))
    }
    
    private fun connectWebSocket() {
        try {
            val wsUrl = serverUrl.replace("http", "ws")
            socket = IO.socket(wsUrl)
            
            socket?.on(Socket.EVENT_CONNECT) {
                runOnUiThread {
                    isConnected = true
                    updateConnectionStatus("WebSocket connected")
                }
            }
            
            socket?.on(Socket.EVENT_DISCONNECT) {
                runOnUiThread {
                    isConnected = false
                    updateConnectionStatus("WebSocket disconnected")
                }
            }
            
            socket?.on("detection_results") { args ->
                if (args.isNotEmpty()) {
                    val data = args[0] as JSONObject
                    runOnUiThread {
                        handleDetectionResults(data)
                    }
                }
            }
            
            socket?.on("error") { args ->
                if (args.isNotEmpty()) {
                    val error = args[0] as JSONObject
                    runOnUiThread {
                        val message = error.optString("message", "Unknown error")
                        updateConnectionStatus("Error: $message")
                    }
                }
            }
            
            socket?.connect()
            
        } catch (e: URISyntaxException) {
            Log.e(TAG, "WebSocket connection failed", e)
            updateConnectionStatus("WebSocket connection failed")
        }
    }
    
    private fun startDetection() {
        isDetectionActive = true
        binding.buttonToggleDetection.text = "Stop Detection"
        binding.buttonToggleDetection.setBackgroundColor(
            ContextCompat.getColor(this, android.R.color.holo_red_dark)
        )
        updateConnectionStatus("Detection started")
    }
    
    private fun stopDetection() {
        isDetectionActive = false
        binding.buttonToggleDetection.text = "Start Detection"
        binding.buttonToggleDetection.setBackgroundColor(
            ContextCompat.getColor(this, android.R.color.holo_green_dark)
        )
        updateConnectionStatus("Detection stopped")
    }
    
    private fun updateConnectionStatus(status: String) {
        binding.textViewConnectionStatus.text = status
    }
    
    private fun handleDetectionResults(data: JSONObject) {
        try {
            val success = data.optBoolean("success", false)
            if (!success) {
                val error = data.optString("error", "Unknown error")
                updateDetectionResults("Error: $error", false)
                return
            }
            
            val results = data.optJSONObject("results")
            if (results != null) {
                val calibrating = results.optBoolean("calibrating", false)
                val overallSuspicious = results.optBoolean("overallSuspicious", false)
                val behavior = results.optString("behavior", "Normal")
                val confidence = results.optDouble("behaviorConfidence", 0.0)
                
                val statusText = if (calibrating) {
                    "🔄 Calibrating... Keep head straight"
                } else {
                    "Behavior: $behavior (${String.format("%.1f", confidence * 100)}%)"
                }
                
                updateDetectionResults(statusText, overallSuspicious)
                
                // Update detailed results
                updateDetailedResults(results)
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Error handling detection results", e)
            updateDetectionResults("Error parsing results", false)
        }
    }
    
    private fun updateDetectionResults(status: String, suspicious: Boolean) {
        binding.textViewDetectionStatus.text = status
        binding.textViewDetectionStatus.setTextColor(
            ContextCompat.getColor(
                this,
                if (suspicious) android.R.color.holo_red_dark else android.R.color.holo_green_dark
            )
        )
    }
    
    private fun updateDetailedResults(results: JSONObject) {
        val details = StringBuilder()
        
        details.append("👁️ Gaze: ${results.optString("gazeDirection", "Unknown")}\n")
        details.append("🗣️ Head: ${results.optString("headDirection", "Unknown")}\n")
        details.append("📱 Mobile: ${if (results.optBoolean("mobileDetected")) "Detected" else "None"}\n")
        details.append("🗣️ Talking: ${if (results.optBoolean("isTalking")) "Yes" else "No"}\n")
        details.append("😊 Expression: ${results.optString("facialExpression", "Unknown")}\n")
        details.append("👥 People: ${results.optInt("personCount", 1)}\n")
        
        val audioResult = results.optJSONObject("audioResult")
        if (audioResult != null) {
            val audioSuspicious = audioResult.optBoolean("isSuspicious", false)
            details.append("🎤 Audio: ${if (audioSuspicious) "Suspicious" else "Normal"}\n")
        }
        
        binding.textViewDetailedResults.text = details.toString()
    }
    
    private inner class FrameAnalyzer : ImageAnalysis.Analyzer {
        override fun analyze(imageProxy: ImageProxy) {
            if (!isDetectionActive || !isConnected) {
                imageProxy.close()
                return
            }
            
            val currentTime = System.currentTimeMillis()
            if (currentTime - lastFrameTime < frameInterval) {
                imageProxy.close()
                return
            }
            
            lastFrameTime = currentTime
            
            try {
                val bitmap = imageProxyToBitmap(imageProxy)
                val base64Image = bitmapToBase64(bitmap)
                
                // Send frame via WebSocket
                val frameData = JSONObject().apply {
                    put("image", base64Image)
                    put("timestamp", currentTime)
                }
                
                socket?.emit("process_frame", frameData)
                
            } catch (e: Exception) {
                Log.e(TAG, "Error processing frame", e)
            } finally {
                imageProxy.close()
            }
        }
    }
    
    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap {
        val buffer: ByteBuffer = imageProxy.planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }
    
    private fun bitmapToBase64(bitmap: Bitmap): String {
        val byteArrayOutputStream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 70, byteArrayOutputStream)
        val byteArray = byteArrayOutputStream.toByteArray()
        return Base64.encodeToString(byteArray, Base64.DEFAULT)
    }
    
    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        socket?.disconnect()
    }
}
