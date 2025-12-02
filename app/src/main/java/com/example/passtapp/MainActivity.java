package com.example.passtapp;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.widget.Toast;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.app.AlertDialog;
import androidx.core.content.ContextCompat;
import com.example.passtapp.databinding.ActivityMainBinding;

public class MainActivity extends AppCompatActivity {

    private ActivityMainBinding binding;
    private AudioSceneAnalyzer audioSceneAnalyzer;
    private boolean isStreaming = false;
    private AlertDialog noiseDialog;
    private boolean noiseModeEnabled = false;

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

        binding.playBufferButton.setOnClickListener(
                v -> {
                    boolean ok = audioSceneAnalyzer.playCurrentBuffer();
                    if (ok) {
                        binding.statusText.setText(getString(R.string.playing_buffer));
                    } else {
                        binding.statusText.setText(getString(R.string.no_buffer_available));
                        Toast.makeText(this, R.string.no_buffer_available, Toast.LENGTH_SHORT)
                                .show();
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
                    maybeShowNoiseDialog(result);
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

    private void maybeShowNoiseDialog(SceneResult result) {
        if (result == null || result.getPredictions() == null) {
            return;
        }
        boolean speechDetected =
                result.getPredictions().stream()
                        .map(Prediction::getLabel)
                        .filter(label -> label != null)
                        .map(String::toLowerCase)
                        .anyMatch(
                                label ->
                                        label.contains("speech")
                                                || label.contains("speaking")
                                                || label.contains("talk")
                                                || label.contains("voice")
                                                || label.contains("lecture")
                                                || label.contains("presentation")
                                                || label.contains("说话")
                                                || label.contains("讲话")
                                                || label.contains("演讲")
                                                || label.contains("人声")
                                                || label.contains("语音"));
        if (speechDetected == noiseModeEnabled) {
            return;
        }
        noiseModeEnabled = speechDetected;
        if (speechDetected) {
            noiseDialog =
                    new AlertDialog.Builder(this)
                            .setTitle(R.string.noise_reduction_title)
                            .setMessage(R.string.noise_reduction_message)
                            .setPositiveButton(R.string.ok, null)
                            .show();
        } else if (noiseDialog != null && noiseDialog.isShowing()) {
            noiseDialog.dismiss();
        }
    }
}
