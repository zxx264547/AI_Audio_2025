package com.example.passtapp;

import android.annotation.SuppressLint;
import android.content.Context;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

public class AudioSceneAnalyzer {

    private static final int SAMPLE_RATE = 32_000;
    private static final int CLIP_SECONDS = 10;
    private static final int CHUNK_SIZE = 2048;
    private static final long INFERENCE_INTERVAL_MS = 2000L;
    private static final float MIN_AVG_AMPLITUDE = 1e-4f;

    private final PaSSTModule passtModule;
    private final int expectedSamples;
    private final float normalizer;
    private final Handler mainHandler = new Handler(Looper.getMainLooper());

    private final AtomicBoolean running = new AtomicBoolean(false);
    private final AtomicBoolean inferring = new AtomicBoolean(false);
    private final ExecutorService inferenceExecutor =
            Executors.newSingleThreadExecutor(
                    r -> {
                        Thread t = new Thread(r, "AudioInference");
                        t.setDaemon(true);
                        return t;
                    });
    private Thread streamingThread;

    public AudioSceneAnalyzer(Context context) {
        this.passtModule = new PaSSTModule(context, SAMPLE_RATE);
        this.expectedSamples = SAMPLE_RATE * CLIP_SECONDS;
        this.normalizer = 1f / Short.MAX_VALUE;
    }

    public SceneResult captureAndClassify() {
        float[] buffer = new float[expectedSamples];
        AudioRecord audioRecord = buildRecorder();
        try {
            audioRecord.startRecording();
            int totalRead = 0;
            short[] pcm = new short[expectedSamples];
            while (totalRead < pcm.length) {
                int read =
                        audioRecord.read(
                                pcm,
                                totalRead,
                                pcm.length - totalRead,
                                AudioRecord.READ_BLOCKING);
                if (read < 0) {
                    throw new IllegalStateException("AudioRecord read failed: " + read);
                }
                if (read == 0) {
                    break;
                }
                totalRead += read;
            }
            for (int i = 0; i < buffer.length; i++) {
                buffer[i] = i < totalRead ? pcm[i] * normalizer : 0f;
            }
            return passtModule.classify(buffer, totalRead);
        } finally {
            try {
                audioRecord.stop();
            } catch (Exception ignored) {
                // ignore stop failure
            }
            audioRecord.release();
        }
    }

    public synchronized void startStreaming(
            ResultCallback onResult,
            StatusCallback onStatus,
            ErrorCallback onError,
            InferenceTimeCallback onInferenceTime) {
        if (running.get()) {
            return;
        }
        running.set(true);
        streamingThread =
                new Thread(
                        () -> runStreamingLoop(onResult, onStatus, onError, onInferenceTime),
                        "AudioSceneStreaming");
        streamingThread.start();
    }

    public synchronized void stopStreaming() {
        running.set(false);
        if (streamingThread != null) {
            try {
                streamingThread.join(500);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            streamingThread = null;
        }
    }

    @SuppressLint("MissingPermission")
    private AudioRecord buildRecorder() {
        int minBuffer =
                AudioRecord.getMinBufferSize(
                        SAMPLE_RATE,
                        AudioFormat.CHANNEL_IN_MONO,
                        AudioFormat.ENCODING_PCM_16BIT);
        int bufferSize = Math.max(minBuffer, CHUNK_SIZE * 2);
        return new AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufferSize);
    }

    public void release() {
        stopStreaming();
        passtModule.release();
        inferenceExecutor.shutdownNow();
    }

    private void runStreamingLoop(
            ResultCallback onResult,
            StatusCallback onStatus,
            ErrorCallback onError,
            InferenceTimeCallback onInferenceTime) {
        float[] ringBuffer = new float[expectedSamples];
        short[] pcmChunk = new short[CHUNK_SIZE];
        int writePos = 0;
        boolean filled = false;
        long lastInference = 0L;
        AudioRecord recorder = buildRecorder();
        try {
            recorder.startRecording();
            postStatus(onStatus, "正在收音...");
            while (running.get() && !Thread.currentThread().isInterrupted()) {
                int read =
                        recorder.read(pcmChunk, 0, pcmChunk.length, AudioRecord.READ_BLOCKING);
                if (read < 0) {
                    postError(onError, "录音失败: " + read);
                    continue;
                }
                if (read == 0) {
                    continue;
                }
                for (int i = 0; i < read; i++) {
                    ringBuffer[writePos] = pcmChunk[i] * normalizer;
                    writePos = (writePos + 1) % ringBuffer.length;
                    if (writePos == 0) {
                        filled = true;
                    }
                }
                long now = SystemClock.elapsedRealtime();
                if (filled && now - lastInference >= INFERENCE_INTERVAL_MS) {
                    lastInference = now;
                    float[] snapshot = new float[ringBuffer.length];
                    int idx = writePos;
                    float sumAbs = 0f;
                    for (int i = 0; i < snapshot.length; i++) {
                        float value = ringBuffer[idx];
                        snapshot[i] = value;
                        sumAbs += Math.abs(value);
                        idx = (idx + 1) % snapshot.length;
                    }
                    float avg = sumAbs / snapshot.length;
                    if (avg < MIN_AVG_AMPLITUDE) {
                        postError(onError, "音量过小，未检测到有效信号。");
                    } else {
                        dispatchInference(snapshot, onResult, onStatus, onError, onInferenceTime);
                    }
                }
            }
        } catch (Exception ex) {
            postError(
                    onError,
                    ex.getLocalizedMessage() != null
                            ? ex.getLocalizedMessage()
                            : ex.toString());
        } finally {
            try {
                recorder.stop();
            } catch (Exception ignored) {
                // ignore stop failure
            }
            recorder.release();
            postStatus(onStatus, "已停止");
        }
    }

    private void dispatchInference(
            float[] snapshot,
            ResultCallback onResult,
            StatusCallback onStatus,
            ErrorCallback onError,
            InferenceTimeCallback onInferenceTime) {
        // If a previous inference is running, skip this round to keep capture responsive.
        if (!inferring.compareAndSet(false, true)) {
            return;
        }
        postStatus(onStatus, "开始推理...");
        inferenceExecutor.execute(
                () -> {
                    long inferStart = SystemClock.elapsedRealtime();
                    try {
                        SceneResult result = passtModule.classify(snapshot, snapshot.length);
                        long duration = SystemClock.elapsedRealtime() - inferStart;
                        postResult(onResult, result);
                        postInferenceTime(onInferenceTime, duration);
                        postStatus(onStatus, "推理结束，用时 " + duration + " ms");
                    } catch (Exception ex) {
                        postError(
                                onError,
                                ex.getLocalizedMessage() != null
                                        ? ex.getLocalizedMessage()
                                        : ex.toString());
                    } finally {
                        inferring.set(false);
                    }
                });
    }

    private void postResult(ResultCallback callback, SceneResult result) {
        if (callback == null) {
            return;
        }
        mainHandler.post(() -> callback.onResult(result));
    }

    private void postStatus(StatusCallback callback, String status) {
        if (callback == null) {
            return;
        }
        mainHandler.post(() -> callback.onStatus(status));
    }

    private void postError(ErrorCallback callback, String message) {
        if (callback == null) {
            return;
        }
        mainHandler.post(() -> callback.onError(message));
    }

    private void postInferenceTime(InferenceTimeCallback callback, long durationMs) {
        if (callback == null) {
            return;
        }
        mainHandler.post(() -> callback.onInferenceTime(durationMs));
    }

    public interface ResultCallback {
        void onResult(SceneResult result);
    }

    public interface StatusCallback {
        void onStatus(String status);
    }

    public interface ErrorCallback {
        void onError(String message);
    }

    public interface InferenceTimeCallback {
        void onInferenceTime(long durationMs);
    }
}
