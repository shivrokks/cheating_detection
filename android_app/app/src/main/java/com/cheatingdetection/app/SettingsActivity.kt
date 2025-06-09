package com.cheatingdetection.app

import android.content.SharedPreferences
import android.os.Bundle
import android.widget.SeekBar
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.cheatingdetection.app.databinding.ActivitySettingsBinding

class SettingsActivity : AppCompatActivity() {
    
    private lateinit var binding: ActivitySettingsBinding
    private lateinit var sharedPreferences: SharedPreferences
    
    companion object {
        const val KEY_FRAME_QUALITY = "frame_quality"
        const val KEY_FRAME_INTERVAL = "frame_interval"
        const val KEY_AUTO_RECONNECT = "auto_reconnect"
        const val KEY_SHOW_DETAILED_RESULTS = "show_detailed_results"
        const val KEY_VIBRATE_ON_DETECTION = "vibrate_on_detection"
        
        const val DEFAULT_FRAME_QUALITY = 70
        const val DEFAULT_FRAME_INTERVAL = 1000
        const val DEFAULT_AUTO_RECONNECT = true
        const val DEFAULT_SHOW_DETAILED_RESULTS = true
        const val DEFAULT_VIBRATE_ON_DETECTION = true
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivitySettingsBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        sharedPreferences = getSharedPreferences(MainActivity.PREFS_NAME, MODE_PRIVATE)
        
        setupUI()
        loadSettings()
    }
    
    private fun setupUI() {
        // Back button
        binding.buttonBack.setOnClickListener {
            finish()
        }
        
        // Save button
        binding.buttonSave.setOnClickListener {
            saveSettings()
        }
        
        // Reset button
        binding.buttonReset.setOnClickListener {
            resetToDefaults()
        }
        
        // Frame quality seekbar
        binding.seekBarFrameQuality.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                binding.textViewFrameQualityValue.text = "$progress%"
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })
        
        // Frame interval seekbar
        binding.seekBarFrameInterval.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                val intervalMs = (progress + 1) * 500 // 500ms to 5000ms
                val intervalSec = intervalMs / 1000.0
                binding.textViewFrameIntervalValue.text = "${intervalSec}s"
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })
    }
    
    private fun loadSettings() {
        // Load frame quality
        val frameQuality = sharedPreferences.getInt(KEY_FRAME_QUALITY, DEFAULT_FRAME_QUALITY)
        binding.seekBarFrameQuality.progress = frameQuality
        binding.textViewFrameQualityValue.text = "$frameQuality%"
        
        // Load frame interval
        val frameInterval = sharedPreferences.getInt(KEY_FRAME_INTERVAL, DEFAULT_FRAME_INTERVAL)
        val seekBarProgress = (frameInterval / 500) - 1 // Convert back to seekbar range
        binding.seekBarFrameInterval.progress = seekBarProgress.coerceIn(0, 9)
        val intervalSec = frameInterval / 1000.0
        binding.textViewFrameIntervalValue.text = "${intervalSec}s"
        
        // Load boolean settings
        binding.switchAutoReconnect.isChecked = 
            sharedPreferences.getBoolean(KEY_AUTO_RECONNECT, DEFAULT_AUTO_RECONNECT)
        
        binding.switchShowDetailedResults.isChecked = 
            sharedPreferences.getBoolean(KEY_SHOW_DETAILED_RESULTS, DEFAULT_SHOW_DETAILED_RESULTS)
        
        binding.switchVibrateOnDetection.isChecked = 
            sharedPreferences.getBoolean(KEY_VIBRATE_ON_DETECTION, DEFAULT_VIBRATE_ON_DETECTION)
    }
    
    private fun saveSettings() {
        val editor = sharedPreferences.edit()
        
        // Save frame quality
        val frameQuality = binding.seekBarFrameQuality.progress
        editor.putInt(KEY_FRAME_QUALITY, frameQuality)
        
        // Save frame interval
        val frameInterval = (binding.seekBarFrameInterval.progress + 1) * 500
        editor.putInt(KEY_FRAME_INTERVAL, frameInterval)
        
        // Save boolean settings
        editor.putBoolean(KEY_AUTO_RECONNECT, binding.switchAutoReconnect.isChecked)
        editor.putBoolean(KEY_SHOW_DETAILED_RESULTS, binding.switchShowDetailedResults.isChecked)
        editor.putBoolean(KEY_VIBRATE_ON_DETECTION, binding.switchVibrateOnDetection.isChecked)
        
        editor.apply()
        
        Toast.makeText(this, "Settings saved successfully", Toast.LENGTH_SHORT).show()
    }
    
    private fun resetToDefaults() {
        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("Reset Settings")
            .setMessage("Are you sure you want to reset all settings to default values?")
            .setPositiveButton("Reset") { _, _ ->
                // Reset to defaults
                binding.seekBarFrameQuality.progress = DEFAULT_FRAME_QUALITY
                binding.textViewFrameQualityValue.text = "$DEFAULT_FRAME_QUALITY%"
                
                val defaultSeekBarProgress = (DEFAULT_FRAME_INTERVAL / 500) - 1
                binding.seekBarFrameInterval.progress = defaultSeekBarProgress
                val defaultIntervalSec = DEFAULT_FRAME_INTERVAL / 1000.0
                binding.textViewFrameIntervalValue.text = "${defaultIntervalSec}s"
                
                binding.switchAutoReconnect.isChecked = DEFAULT_AUTO_RECONNECT
                binding.switchShowDetailedResults.isChecked = DEFAULT_SHOW_DETAILED_RESULTS
                binding.switchVibrateOnDetection.isChecked = DEFAULT_VIBRATE_ON_DETECTION
                
                Toast.makeText(this, "Settings reset to defaults", Toast.LENGTH_SHORT).show()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }
}
