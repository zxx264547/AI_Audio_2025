package com.example.passtapp;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.graphics.Typeface;
import android.text.SpannableStringBuilder;
import android.text.style.AbsoluteSizeSpan;
import android.text.style.StyleSpan;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import java.util.List;
import java.util.Locale;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.coordinatorlayout.widget.CoordinatorLayout;
import androidx.core.content.ContextCompat;
import com.example.passtapp.databinding.ActivityMainBinding;
import com.google.android.material.snackbar.Snackbar;

public class MainActivity extends AppCompatActivity {

    private ActivityMainBinding binding;
    private AudioSceneAnalyzer audioSceneAnalyzer;
    private boolean isStreaming = false;
    private String lastScene = null;

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

        binding.playRawButton.setOnClickListener(
                v -> {
                    pauseStreamingIfNeeded();
                    AudioSceneAnalyzer.PlaybackResult res = audioSceneAnalyzer.playRawBuffer();
                    handlePlaybackResult(res, R.string.play_raw_buffer);
                });

        binding.playProcessedButton.setOnClickListener(
                v -> {
                    pauseStreamingIfNeeded();
                    AudioSceneAnalyzer.PlaybackResult res =
                            audioSceneAnalyzer.playProcessedBuffer();
                    handlePlaybackResult(res, R.string.play_processed_buffer);
                });

        binding.saveBufferButton.setOnClickListener(
                v -> {
                    pauseStreamingIfNeeded();
                    AudioSceneAnalyzer.SaveResult res = audioSceneAnalyzer.saveCurrentBuffers();
                    handleSaveResult(res);
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
                    renderResult(result);
                    maybeShowSceneDialog(result);
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

    private void pauseStreamingIfNeeded() {
        if (isStreaming) {
            stopStreaming();
        }
    }

    @Override
    protected void onDestroy() {
        audioSceneAnalyzer.stopStreaming();
        audioSceneAnalyzer.release();
        super.onDestroy();
    }

    private void maybeShowSceneDialog(SceneResult result) {
        if (result == null || result.getScene() == null) {
            return;
        }
        String sceneName = result.getScene().getScene();
        if (sceneName == null || sceneName.isEmpty()) {
            return;
        }
        if (sceneName.equals(lastScene)) {
            return;
        }
        lastScene = sceneName;
        String message = getString(R.string.scene_switch_message, sceneName);
        showCenteredSnackbar(message);
    }

    private void handlePlaybackResult(AudioSceneAnalyzer.PlaybackResult res, int successResId) {
        if (res == null || !res.success) {
            binding.statusText.setText(getString(R.string.no_buffer_available));
            showCenteredSnackbar(getString(R.string.no_buffer_available));
            return;
        }
        String message = getString(successResId);
        binding.statusText.setText(message);
        showCenteredSnackbar(message);
    }

    private void handleSaveResult(AudioSceneAnalyzer.SaveResult res) {
        if (res == null || !res.success) {
            binding.statusText.setText(getString(R.string.save_buffers_failed));
            showCenteredSnackbar(getString(R.string.no_buffer_available));
            return;
        }
        binding.statusText.setText(getString(R.string.save_buffers_success));
        String msg =
                getString(R.string.save_buffers_success)
                        + "\n降噪前: "
                        + res.rawPath
                        + "\n降噪后: "
                        + res.processedPath;
        showCenteredSnackbar(msg);
    }

    private void renderResult(SceneResult result) {
        if (result == null) {
            binding.resultText.setText("");
            return;
        }
        binding.resultText.setGravity(Gravity.START);
        String sceneName =
                result.getScene() != null && result.getScene().getScene() != null
                        ? result.getScene().getScene()
                        : getString(R.string.unknown_scene);
        StringBuilder sb = new StringBuilder();
        sb.append("模式: ").append(sceneName);
        String debug =
                result.getScene() != null ? result.getScene().getDebug() : null;
        if (debug != null && !debug.isEmpty()) {
            sb.append("\n判定标签:\n");
            String[] parts = debug.split("\\|");
            for (String part : parts) {
                String trimmed = part.trim();
                if (!trimmed.isEmpty()) {
                    sb.append(trimmed).append('\n');
                }
            }
            if (sb.length() > 0 && sb.charAt(sb.length() - 1) == '\n') {
                sb.deleteCharAt(sb.length() - 1);
            }
        }

        List<Prediction> preds = result.getPredictions();
        if (preds != null && !preds.isEmpty()) {
            sb.append("\n识别结果:\n");
            for (int i = 0; i < preds.size(); i++) {
                Prediction p = preds.get(i);
                sb.append(i + 1)
                        .append(". ")
                        .append(p.getLabel())
                        .append(" (")
                        .append(String.format(Locale.getDefault(), "%.2f", p.getConfidence()))
                        .append(')')
                        .append('\n');
            }
            // remove last newline
            if (sb.length() > 0 && sb.charAt(sb.length() - 1) == '\n') {
                sb.deleteCharAt(sb.length() - 1);
            }
        }

        SpannableStringBuilder builder = new SpannableStringBuilder(sb.toString());
        int endOfFirstLine = sb.indexOf("\n");
        int spanEnd = endOfFirstLine >= 0 ? endOfFirstLine : sb.length();
        builder.setSpan(new StyleSpan(Typeface.BOLD), 0, spanEnd, 0);
        builder.setSpan(new AbsoluteSizeSpan(26, true), 0, spanEnd, 0);
        if (endOfFirstLine >= 0) {
            builder.setSpan(new AbsoluteSizeSpan(14, true), endOfFirstLine, sb.length(), 0);
        }
        binding.resultText.setText(builder);
    }

    private void showCenteredSnackbar(String message) {
        Snackbar snackbar = Snackbar.make(binding.getRoot(), message, Snackbar.LENGTH_SHORT);
        View sbView = snackbar.getView();
        ViewGroup.LayoutParams params = sbView.getLayoutParams();
        if (params instanceof FrameLayout.LayoutParams) {
            FrameLayout.LayoutParams lp = (FrameLayout.LayoutParams) params;
            lp.gravity = Gravity.CENTER;
            lp.width = FrameLayout.LayoutParams.WRAP_CONTENT;
            sbView.setLayoutParams(lp);
        } else if (params instanceof CoordinatorLayout.LayoutParams) {
            CoordinatorLayout.LayoutParams lp = (CoordinatorLayout.LayoutParams) params;
            lp.gravity = Gravity.CENTER;
            lp.width = CoordinatorLayout.LayoutParams.WRAP_CONTENT;
            sbView.setLayoutParams(lp);
        }
        int padding = (int) (16 * getResources().getDisplayMetrics().density);
        sbView.setPadding(padding * 2, padding, padding * 2, padding);
        snackbar.show();
    }
}
