package com.cheatingdetection.app

import retrofit2.Call
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.*
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import java.util.concurrent.TimeUnit

// Data classes for API responses
data class HealthResponse(
    val status: String,
    val timestamp: Double,
    val calibrating: Boolean,
    val audioInitialized: Boolean
)

data class ProcessFrameRequest(
    val image: String,
    val returnProcessedImage: Boolean = false
)

data class DetectionResults(
    val timestamp: Double,
    val calibrating: Boolean,
    val gazeDirection: String,
    val headDirection: String,
    val mobileDetected: Boolean,
    val isTalking: Boolean,
    val facialExpression: String,
    val personCount: Int,
    val multiplePeople: Boolean,
    val newPersonEntered: Boolean,
    val suspiciousObjects: Boolean,
    val detectedObjects: List<String>,
    val audioResult: AudioResult,
    val behavior: String,
    val behaviorConfidence: Double,
    val overallSuspicious: Boolean
)

data class AudioResult(
    val isSuspicious: Boolean,
    val suspicionScore: Int,
    val detectedText: String,
    val suspiciousWords: List<String>,
    val recentDetections: Int
)

data class ProcessFrameResponse(
    val success: Boolean,
    val results: DetectionResults?,
    val processedImage: String?,
    val error: String?
)

data class StatusResponse(
    val success: Boolean,
    val status: DetectionResults?,
    val calibrating: Boolean,
    val audioInitialized: Boolean,
    val error: String?
)

// API Service Interface
interface ApiService {
    
    @GET("health")
    fun healthCheck(): Call<HealthResponse>
    
    @POST("process_frame")
    fun processFrame(@Body request: ProcessFrameRequest): Call<ProcessFrameResponse>
    
    @GET("get_status")
    fun getStatus(): Call<StatusResponse>
    
    companion object {
        fun create(baseUrl: String): ApiService {
            // Create logging interceptor
            val loggingInterceptor = HttpLoggingInterceptor().apply {
                level = HttpLoggingInterceptor.Level.BASIC
            }
            
            // Create OkHttp client with timeouts
            val okHttpClient = OkHttpClient.Builder()
                .addInterceptor(loggingInterceptor)
                .connectTimeout(30, TimeUnit.SECONDS)
                .readTimeout(60, TimeUnit.SECONDS)
                .writeTimeout(60, TimeUnit.SECONDS)
                .build()
            
            // Ensure baseUrl ends with /
            val formattedBaseUrl = if (baseUrl.endsWith("/")) baseUrl else "$baseUrl/"
            
            // Create Retrofit instance
            val retrofit = Retrofit.Builder()
                .baseUrl(formattedBaseUrl)
                .client(okHttpClient)
                .addConverterFactory(GsonConverterFactory.create())
                .build()
            
            return retrofit.create(ApiService::class.java)
        }
    }
}

// WebSocket Event Classes
data class WebSocketMessage(
    val type: String,
    val data: Any?
)

data class FrameData(
    val image: String,
    val timestamp: Long
)

data class DetectionResultsEvent(
    val success: Boolean,
    val results: DetectionResults?,
    val timestamp: Double,
    val error: String?
)

data class StatusEvent(
    val connected: Boolean,
    val calibrating: Boolean,
    val audioInitialized: Boolean
)

data class ErrorEvent(
    val message: String,
    val timestamp: Double = System.currentTimeMillis() / 1000.0
)
