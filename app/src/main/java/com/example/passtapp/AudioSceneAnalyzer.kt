package com.example.passtapp

import android.annotation.SuppressLint
import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.SystemClock
import kotlin.math.abs
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class AudioSceneAnalyzer(context: Context) {

    private val sampleRate = 32_000
    private val clipSeconds = 10
    private val chunkSize = 2048
    private val inferenceIntervalMs = 2000L
    private val passtModule = PaSSTModule(context, sampleRate)
    private val expectedSamples = sampleRate * clipSeconds
    private val normalizer = 1f / Short.MAX_VALUE

    private var streamingJob: Job? = null

    suspend fun captureAndClassify(): SceneResult = withContext(Dispatchers.IO) {
        val buffer = FloatArray(expectedSamples)
        val audioRecord = buildRecorder()
        try {
            audioRecord.startRecording()
            var totalRead = 0
            val pcm = ShortArray(expectedSamples)
            while (totalRead < pcm.size) {
                val read = audioRecord.read(
                    pcm,
                    totalRead,
                    pcm.size - totalRead,
                    AudioRecord.READ_BLOCKING
                )
                if (read < 0) {
                    throw IllegalStateException("AudioRecord read failed: $read")
                }
                if (read == 0) break
                totalRead += read
            }
            for (i in buffer.indices) {
                buffer[i] = if (i < totalRead) pcm[i] * normalizer else 0f
            }
            passtModule.classify(buffer, totalRead)
        } finally {
            audioRecord.stop()
            audioRecord.release()
        }
    }

    fun startStreaming(
        scope: CoroutineScope,
        onResult: (SceneResult) -> Unit,
        onStatus: (String) -> Unit,
        onError: (String) -> Unit
    ) {
        if (streamingJob != null) return
        streamingJob = scope.launch(Dispatchers.IO) {
            val ringBuffer = FloatArray(expectedSamples)
            val pcmChunk = ShortArray(chunkSize)
            var writePos = 0
            var filled = false
            var lastInference = 0L
            val recorder = buildRecorder()
            try {
                recorder.startRecording()
                withContext(Dispatchers.Main) {
                    onStatus("实时识别中…")
                }
                while (isActive) {
                    val read = recorder.read(pcmChunk, 0, pcmChunk.size, AudioRecord.READ_BLOCKING)
                    if (read < 0) {
                        withContext(Dispatchers.Main) {
                            onError("录音失败: $read")
                        }
                        continue
                    }
                    if (read == 0) continue
                    for (i in 0 until read) {
                        ringBuffer[writePos] = pcmChunk[i] * normalizer
                        writePos = (writePos + 1) % ringBuffer.size
                        if (writePos == 0) filled = true
                    }
                    val now = SystemClock.elapsedRealtime()
                    if (filled && now - lastInference >= inferenceIntervalMs) {
                        lastInference = now
                        val snapshot = FloatArray(ringBuffer.size)
                        var idx = writePos
                        var sumAbs = 0f
                        for (i in snapshot.indices) {
                            val value = ringBuffer[idx]
                            snapshot[i] = value
                            sumAbs += abs(value)
                            idx = (idx + 1) % snapshot.size
                        }
                        val avg = sumAbs / snapshot.size
                        if (avg < 1e-4f) {
                            withContext(Dispatchers.Main) {
                                onError("音量过低，未检测到有效音频")
                            }
                        } else {
                            val result = passtModule.classify(snapshot, snapshot.size)
                            withContext(Dispatchers.Main) {
                                onResult(result)
                            }
                        }
                    }
                }
            } catch (ex: Exception) {
                withContext(Dispatchers.Main) {
                    onError(ex.localizedMessage ?: ex.message ?: ex.toString())
                }
            } finally {
                recorder.stop()
                recorder.release()
                withContext(Dispatchers.Main) {
                    onStatus("已停止监听")
                }
            }
        }
    }

    suspend fun stopStreaming() {
        val job = streamingJob ?: return
        streamingJob = null
        job.cancelAndJoin()
    }

    @SuppressLint("MissingPermission")
    private fun buildRecorder(): AudioRecord {
        val minBuffer = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )
        val bufferSize = maxOf(minBuffer, chunkSize * 2)
        return AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufferSize
        )
    }

    fun release() {
        passtModule.release()
    }
}
