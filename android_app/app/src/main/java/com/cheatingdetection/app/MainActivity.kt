package com.cheatingdetection.app

import android.Manifest
import android.content.Intent
import android.content.SharedPreferences
import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.PermissionChecker
import com.cheatingdetection.app.databinding.ActivityMainBinding
import com.karumi.dexter.Dexter
import com.karumi.dexter.MultiplePermissionsReport
import com.karumi.dexter.PermissionToken
import com.karumi.dexter.listener.PermissionRequest
import com.karumi.dexter.listener.multi.MultiplePermissionsListener

class MainActivity : AppCompatActivity() {
    
    private lateinit var binding: ActivityMainBinding
    private lateinit var sharedPreferences: SharedPreferences
    
    companion object {
        const val PREFS_NAME = "CheatingDetectionPrefs"
        const val KEY_SERVER_URL = "server_url"
        const val DEFAULT_SERVER_URL = "https://7fad-103-180-45-255.ngrok-free.app"
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        sharedPreferences = getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
        
        setupUI()
        checkPermissions()
    }
    
    private fun setupUI() {
        // Load saved server URL
        val savedUrl = sharedPreferences.getString(KEY_SERVER_URL, DEFAULT_SERVER_URL)
        binding.editTextServerUrl.setText(savedUrl)
        
        // Start Detection button
        binding.buttonStartDetection.setOnClickListener {
            val serverUrl = binding.editTextServerUrl.text.toString().trim()
            if (serverUrl.isEmpty()) {
                Toast.makeText(this, "Please enter server URL", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            
            // Save server URL
            sharedPreferences.edit()
                .putString(KEY_SERVER_URL, serverUrl)
                .apply()
            
            // Start camera activity
            val intent = Intent(this, CameraActivity::class.java)
            intent.putExtra("server_url", serverUrl)
            startActivity(intent)
        }
        
        // Settings button
        binding.buttonSettings.setOnClickListener {
            val intent = Intent(this, SettingsActivity::class.java)
            startActivity(intent)
        }
        
        // Test Connection button
        binding.buttonTestConnection.setOnClickListener {
            val serverUrl = binding.editTextServerUrl.text.toString().trim()
            if (serverUrl.isEmpty()) {
                Toast.makeText(this, "Please enter server URL", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            
            testServerConnection(serverUrl)
        }
        
        // Help button
        binding.buttonHelp.setOnClickListener {
            showHelpDialog()
        }
    }
    
    private fun checkPermissions() {
        Dexter.withContext(this)
            .withPermissions(
                Manifest.permission.CAMERA,
                Manifest.permission.RECORD_AUDIO,
                Manifest.permission.WRITE_EXTERNAL_STORAGE,
                Manifest.permission.READ_EXTERNAL_STORAGE
            )
            .withListener(object : MultiplePermissionsListener {
                override fun onPermissionsChecked(report: MultiplePermissionsReport) {
                    // Debug: Log permission status
                    android.util.Log.d("MainActivity", "Permission check results:")
                    report.grantedPermissionResponses.forEach {
                        android.util.Log.d("MainActivity", "Granted: ${it.permissionName}")
                    }
                    report.deniedPermissionResponses.forEach {
                        android.util.Log.d("MainActivity", "Denied: ${it.permissionName}")
                    }

                    // Check if essential permissions (camera and audio) are granted
                    val cameraGranted = report.grantedPermissionResponses.any {
                        it.permissionName == Manifest.permission.CAMERA
                    }
                    val audioGranted = report.grantedPermissionResponses.any {
                        it.permissionName == Manifest.permission.RECORD_AUDIO
                    }

                    if (cameraGranted && audioGranted) {
                        // Essential permissions granted - enable start detection
                        binding.buttonStartDetection.isEnabled = true

                        // Check storage permissions for optional features
                        val storageGranted = report.grantedPermissionResponses.any {
                            it.permissionName == Manifest.permission.WRITE_EXTERNAL_STORAGE
                        }

                        if (!storageGranted) {
                            Toast.makeText(
                                this@MainActivity,
                                "✅ Camera and audio permissions granted. Storage permission optional for logs.",
                                Toast.LENGTH_SHORT
                            ).show()
                        } else {
                            Toast.makeText(
                                this@MainActivity,
                                "✅ All permissions granted!",
                                Toast.LENGTH_SHORT
                            ).show()
                        }
                    } else {
                        // Essential permissions denied
                        val missingPermissions = mutableListOf<String>()
                        if (!cameraGranted) missingPermissions.add("Camera")
                        if (!audioGranted) missingPermissions.add("Microphone")

                        Toast.makeText(
                            this@MainActivity,
                            "❌ Required permissions missing: ${missingPermissions.joinToString(", ")}",
                            Toast.LENGTH_LONG
                        ).show()
                        binding.buttonStartDetection.isEnabled = false
                    }
                }

                override fun onPermissionRationaleShouldBeShown(
                    permissions: List<PermissionRequest>,
                    token: PermissionToken
                ) {
                    token.continuePermissionRequest()
                }
            })
            .check()
    }
    
    private fun testServerConnection(serverUrl: String) {
        binding.buttonTestConnection.isEnabled = false
        binding.buttonTestConnection.text = "Testing..."
        
        // Create API service and test connection
        val apiService = ApiService.create(serverUrl)
        
        apiService.healthCheck().enqueue(object : retrofit2.Callback<HealthResponse> {
            override fun onResponse(
                call: retrofit2.Call<HealthResponse>,
                response: retrofit2.Response<HealthResponse>
            ) {
                runOnUiThread {
                    binding.buttonTestConnection.isEnabled = true
                    binding.buttonTestConnection.text = "Test Connection"
                    
                    if (response.isSuccessful) {
                        val healthResponse = response.body()
                        if (healthResponse?.status == "healthy") {
                            Toast.makeText(
                                this@MainActivity,
                                "✅ Server connection successful!",
                                Toast.LENGTH_SHORT
                            ).show()
                            
                            // Update status info
                            binding.textViewConnectionStatus.text = "✅ Connected"
                            binding.textViewServerInfo.text = 
                                "Calibrating: ${healthResponse.calibrating}\n" +
                                "Audio: ${healthResponse.audioInitialized}"
                        } else {
                            Toast.makeText(
                                this@MainActivity,
                                "❌ Server responded but not healthy",
                                Toast.LENGTH_SHORT
                            ).show()
                            binding.textViewConnectionStatus.text = "❌ Server Error"
                        }
                    } else {
                        Toast.makeText(
                            this@MainActivity,
                            "❌ Server error: ${response.code()}",
                            Toast.LENGTH_SHORT
                        ).show()
                        binding.textViewConnectionStatus.text = "❌ Error ${response.code()}"
                    }
                }
            }
            
            override fun onFailure(call: retrofit2.Call<HealthResponse>, t: Throwable) {
                runOnUiThread {
                    binding.buttonTestConnection.isEnabled = true
                    binding.buttonTestConnection.text = "Test Connection"
                    
                    Toast.makeText(
                        this@MainActivity,
                        "❌ Connection failed: ${t.message}",
                        Toast.LENGTH_LONG
                    ).show()
                    binding.textViewConnectionStatus.text = "❌ Connection Failed"
                    binding.textViewServerInfo.text = t.message ?: "Unknown error"
                }
            }
        })
    }
    
    private fun showHelpDialog() {
        val helpText = """
            📱 Cheating Detection App Help
            
            🔧 Setup Instructions:
            1. Start the Python server on your computer
            2. Copy the ngrok URL from the server console
            3. Paste the URL in the "Server URL" field
            4. Click "Test Connection" to verify
            5. Click "Start Detection" to begin monitoring
            
            🎥 During Detection:
            • Keep your face visible to the camera
            • The app will stream video to the server
            • Detection results will be shown in real-time
            • Red indicators show suspicious behavior
            • Green indicators show normal behavior
            
            ⚙️ Settings:
            • Adjust streaming quality
            • Configure detection sensitivity
            • View connection logs
            
            ❓ Troubleshooting:
            • Ensure stable internet connection
            • Check if server is running
            • Verify ngrok URL is correct
            • Grant all required permissions
        """.trimIndent()
        
        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("Help & Instructions")
            .setMessage(helpText)
            .setPositiveButton("OK") { dialog, _ -> dialog.dismiss() }
            .show()
    }
}
