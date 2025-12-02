package com.example.passtapp;

import android.annotation.SuppressLint;
import android.content.Context;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioRecord;
import android.media.AudioTrack;
import android.media.MediaRecorder;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.os.Environment;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

public class AudioSceneAnalyzer {

    private static final int SAMPLE_RATE = 32_000;
    private static final int CLIP_SECONDS = 10;
    private static final int CHUNK_SIZE = 2048;
    private static final float MIN_AVG_AMPLITUDE = 1e-4f;

    private final PaSSTModule passtModule;
    private final int expectedSamples;
    private final float normalizer;
    private final Context appContext;
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
    private volatile float[] lastSnapshot;
    private volatile float[] lastRawSnapshot;
    private volatile NoiseMode currentNoiseMode = NoiseMode.STANDARD;

    public AudioSceneAnalyzer(Context context) {
        this.appContext = context.getApplicationContext();
        this.passtModule = new PaSSTModule(this.appContext, SAMPLE_RATE);
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
        AudioRecord recorder = buildRecorder();
        try {
            recorder.startRecording();
            postStatus(onStatus, "正在收音...");
            postStatus(onStatus, "推理后端: " + passtModule.getBackendName());
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
                // Start inference immediately when previous one完成
                if (filled && !inferring.get()) {
                    float[] snapshot = new float[ringBuffer.length];
                    int idx = writePos;
                    float sumAbs = 0f;
                    for (int i = 0; i < snapshot.length; i++) {
                        float value = ringBuffer[idx];
                        snapshot[i] = value;
                        sumAbs += Math.abs(value);
                        idx = (idx + 1) % snapshot.length;
                    }
                    lastRawSnapshot = snapshot;
                    float[] processed = applyNoiseReduction(snapshot, currentNoiseMode);
                    lastSnapshot = processed;
                    float avg = sumAbs / snapshot.length;
                    if (avg < MIN_AVG_AMPLITUDE) {
                        postError(onError, "音量过小，未检测到有效信号。");
                    } else {
                        dispatchInference(processed, onResult, onStatus, onError, onInferenceTime);
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
                        updateNoiseModeFromScene(result);
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

    private void updateNoiseModeFromScene(SceneResult result) {
        if (result == null || result.getScene() == null) {
            return;
        }
        String sceneName = result.getScene().getScene();
        if (sceneName == null) {
            return;
        }
        String normalized = sceneName.toLowerCase(Locale.getDefault());
        if (normalized.contains("会议")) {
            currentNoiseMode = NoiseMode.MEETING;
        } else if (normalized.contains("户外")) {
            currentNoiseMode = NoiseMode.OUTDOOR;
        } else {
            currentNoiseMode = NoiseMode.STANDARD;
        }
    }

    private float[] applyNoiseReduction(float[] input, NoiseMode mode) {
        if (input == null) {
            return new float[0];
        }
        float[] out = new float[input.length];
        float gate;
        int smoothWindow;
        switch (mode) {
            case MEETING:
                gate = 0.003f;
                smoothWindow = 3;
                break;
            case OUTDOOR:
                gate = 0.008f;
                smoothWindow = 5;
                break;
            case STANDARD:
            default:
                gate = 0.0f;
                smoothWindow = 1;
        }

        // Noise gate
        for (int i = 0; i < input.length; i++) {
            float v = input[i];
            out[i] = Math.abs(v) < gate ? 0f : v;
        }

        // Simple moving average smoothing
        if (smoothWindow > 1) {
            float[] tmp = new float[out.length];
            float sum = 0f;
            int half = smoothWindow / 2;
            for (int i = 0; i < out.length; i++) {
                // add current
                sum += out[i];
                // remove element leaving window
                if (i >= smoothWindow) {
                    sum -= out[i - smoothWindow];
                }
                int count = Math.min(i + 1, smoothWindow);
                tmp[i] = sum / count;
            }
            out = tmp;
        }

        return out;
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

    public PlaybackResult playRawBufferAndExport() {
        return playBuffer(lastRawSnapshot, "raw");
    }

    public PlaybackResult playProcessedBufferAndExport() {
        return playBuffer(lastSnapshot, "denoised");
    }

    private PlaybackResult playBuffer(float[] snapshot, String tag) {
        if (snapshot == null || snapshot.length == 0) {
            return PlaybackResult.failed("empty buffer");
        }
        short[] pcm = new short[snapshot.length];
        for (int i = 0; i < snapshot.length; i++) {
            float v = Math.max(-1f, Math.min(1f, snapshot[i]));
            pcm[i] = (short) (v * Short.MAX_VALUE);
        }
        int bufferSize = pcm.length * 2;
        AudioTrack track =
                new AudioTrack(
                        AudioManager.STREAM_MUSIC,
                        SAMPLE_RATE,
                        AudioFormat.CHANNEL_OUT_MONO,
                        AudioFormat.ENCODING_PCM_16BIT,
                        bufferSize,
                        AudioTrack.MODE_STATIC);
        track.write(pcm, 0, pcm.length);
        track.play();
        String filePath = writeWav(snapshot, tag);
        new Thread(
                        () -> {
                            try {
                                long durationMs = (long) (pcm.length * 1000L / SAMPLE_RATE) + 200;
                                Thread.sleep(durationMs);
                            } catch (InterruptedException ignored) {
                                Thread.currentThread().interrupt();
                            } finally {
                                try {
                                    track.release();
                                } catch (Exception ignored) {
                                }
                            }
                        },
                        "AudioPlayback")
                .start();
        return PlaybackResult.success(filePath);
    }

    private String writeWav(float[] data, String tag) {
        if (data == null || data.length == 0) {
            return null;
        }
        File dir =
                appContext.getExternalFilesDir(Environment.DIRECTORY_MUSIC) != null
                        ? appContext.getExternalFilesDir(Environment.DIRECTORY_MUSIC)
                        : appContext.getFilesDir();
        String name =
                String.format(
                        Locale.getDefault(),
                        "%s_%d.wav",
                        tag,
                        System.currentTimeMillis());
        File outFile = new File(dir, name);
        try (FileOutputStream fos = new FileOutputStream(outFile)) {
            int numSamples = data.length;
            int byteRate = SAMPLE_RATE * 2;
            int totalDataLen = numSamples * 2 + 36;
            // WAV header
            fos.write(new byte[] {'R', 'I', 'F', 'F'});
            writeInt(fos, totalDataLen);
            fos.write(new byte[] {'W', 'A', 'V', 'E'});
            fos.write(new byte[] {'f', 'm', 't', ' '});
            writeInt(fos, 16); // PCM header size
            writeShort(fos, (short) 1); // PCM format
            writeShort(fos, (short) 1); // mono
            writeInt(fos, SAMPLE_RATE);
            writeInt(fos, byteRate);
            writeShort(fos, (short) 2); // block align
            writeShort(fos, (short) 16); // bits per sample
            fos.write(new byte[] {'d', 'a', 't', 'a'});
            writeInt(fos, numSamples * 2);
            // data
            for (float v : data) {
                float clamped = Math.max(-1f, Math.min(1f, v));
                short s = (short) (clamped * Short.MAX_VALUE);
                writeShort(fos, s);
            }
            fos.flush();
            return outFile.getAbsolutePath();
        } catch (IOException e) {
            return null;
        }
    }

    private void writeInt(FileOutputStream fos, int value) throws IOException {
        fos.write(new byte[] {
            (byte) (value & 0xff),
            (byte) ((value >> 8) & 0xff),
            (byte) ((value >> 16) & 0xff),
            (byte) ((value >> 24) & 0xff)
        });
    }

    private void writeShort(FileOutputStream fos, short value) throws IOException {
        fos.write(new byte[] {(byte) (value & 0xff), (byte) ((value >> 8) & 0xff)});
    }

    public static class PlaybackResult {
        public final boolean success;
        public final String filePath;
        public final String error;

        private PlaybackResult(boolean success, String filePath, String error) {
            this.success = success;
            this.filePath = filePath;
            this.error = error;
        }

        public static PlaybackResult success(String filePath) {
            return new PlaybackResult(true, filePath, null);
        }

        public static PlaybackResult failed(String error) {
            return new PlaybackResult(false, null, error);
        }
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

    private enum NoiseMode {
        STANDARD,
        MEETING,
        OUTDOOR
    }
}
