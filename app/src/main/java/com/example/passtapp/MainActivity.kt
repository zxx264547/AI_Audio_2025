package com.example.passtapp

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.example.passtapp.databinding.ActivityMainBinding
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var audioSceneAnalyzer: AudioSceneAnalyzer

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            startCapture()
        } else {
            binding.resultText.text = getString(R.string.permission_denied)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        audioSceneAnalyzer = AudioSceneAnalyzer(this)

        binding.captureButton.setOnClickListener {
            if (hasAudioPermission()) {
                startCapture()
            } else {
                permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
            }
        }
    }

    private fun hasAudioPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun startCapture() {
        binding.captureButton.isEnabled = false
        binding.statusText.text = getString(R.string.recording)
        binding.resultText.text = getString(R.string.listening)

        lifecycleScope.launch {
            try {
                val result = audioSceneAnalyzer.captureAndClassify()
                binding.statusText.text = getString(R.string.detected)
                binding.resultText.text = result.formatForDisplay()
            } catch (ex: Exception) {
                binding.statusText.text = getString(R.string.failed)
                binding.resultText.text = ex.localizedMessage ?: ex.toString()
            } finally {
                binding.captureButton.isEnabled = true
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        audioSceneAnalyzer.release()
    }
}
