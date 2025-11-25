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
    private var isStreaming = false

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            startStreaming()
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
            if (!isStreaming) {
                if (hasAudioPermission()) {
                    startStreaming()
                } else {
                    permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                }
            } else {
                stopStreaming()
            }
        }
    }

    private fun hasAudioPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun startStreaming() {
        binding.captureButton.text = getString(R.string.stop_listening)
        binding.statusText.text = getString(R.string.streaming)
        binding.resultText.text = getString(R.string.listening)
        audioSceneAnalyzer.startStreaming(
            scope = lifecycleScope,
            onResult = { result ->
                binding.statusText.text = getString(R.string.detected)
                binding.resultText.text = result.formatForDisplay()
            },
            onStatus = { status ->
                binding.statusText.text = status
            },
            onError = { message ->
                binding.resultText.text = message
            }
        )
        isStreaming = true
    }

    private fun stopStreaming() {
        binding.captureButton.text = getString(R.string.start_listening)
        lifecycleScope.launch {
            audioSceneAnalyzer.stopStreaming()
            binding.statusText.text = getString(R.string.stopped)
        }
        isStreaming = false
    }

    override fun onDestroy() {
        lifecycleScope.launch {
            audioSceneAnalyzer.stopStreaming()
            audioSceneAnalyzer.release()
        }
        super.onDestroy()
    }
}
