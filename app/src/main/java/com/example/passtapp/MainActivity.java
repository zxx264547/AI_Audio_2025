package com.example.passtapp;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import com.example.passtapp.databinding.ActivityMainBinding;

public class MainActivity extends AppCompatActivity {

    private ActivityMainBinding binding;
    private AudioSceneAnalyzer audioSceneAnalyzer;
    private boolean isStreaming = false;

    private final ActivityResultLauncher<String> permissionLauncher =
            registerForActivityResult(
                    new ActivityResultContracts.RequestPermission(),
                    granted -> {
                        if (granted) {
                            startStreaming();
                        } else if (binding != null) {
                            binding.resultText.setText(getString(R.string.permission_denied));
                        }
                    });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        audioSceneAnalyzer = new AudioSceneAnalyzer(this);

        binding.captureButton.setOnClickListener(
                v -> {
                    if (!isStreaming) {
                        if (hasAudioPermission()) {
                            startStreaming();
                        } else {
                            permissionLauncher.launch(Manifest.permission.RECORD_AUDIO);
                        }
                    } else {
                        stopStreaming();
                    }
                });
    }

    private boolean hasAudioPermission() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                == PackageManager.PERMISSION_GRANTED;
    }

    private void startStreaming() {
        binding.captureButton.setText(getString(R.string.stop_listening));
        binding.statusText.setText(getString(R.string.streaming));
        binding.resultText.setText(getString(R.string.listening));
        binding.inferenceTimeText.setText(getString(R.string.inference_time_placeholder));
        audioSceneAnalyzer.startStreaming(
                result -> {
                    binding.statusText.setText(getString(R.string.detected));
                    binding.resultText.setText(result.formatForDisplay());
                },
                status -> binding.statusText.setText(status),
                message -> binding.resultText.setText(message),
                timeMs ->
                        binding.inferenceTimeText.setText(
                                getString(R.string.inference_time_ms, timeMs)));
        isStreaming = true;
    }

    private void stopStreaming() {
        binding.captureButton.setText(getString(R.string.start_listening));
        audioSceneAnalyzer.stopStreaming();
        binding.statusText.setText(getString(R.string.stopped));
        isStreaming = false;
    }

    @Override
    protected void onDestroy() {
        audioSceneAnalyzer.stopStreaming();
        audioSceneAnalyzer.release();
        super.onDestroy();
    }
}
