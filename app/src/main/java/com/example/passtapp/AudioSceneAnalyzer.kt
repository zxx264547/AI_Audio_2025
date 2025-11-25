package com.example.passtapp

import android.annotation.SuppressLint
import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import kotlin.math.abs
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class AudioSceneAnalyzer(context: Context) {

    private val sampleRate = 32_000
    private val clipSeconds = 10
    private val passtModule = PaSSTModule(context, sampleRate)
    private val desiredSamples = sampleRate * clipSeconds
    private val pcmNormalizer = 1f / Short.MAX_VALUE

    suspend fun captureAndClassify(): SceneResult = withContext(Dispatchers.IO) {
        val pcmBuffer = ShortArray(desiredSamples)
        val audioRecord = buildRecorder()
        try {
            audioRecord.startRecording()
            var totalRead = 0
            while (totalRead < pcmBuffer.size) {
                val read = audioRecord.read(
                    pcmBuffer,
                    totalRead,
                    pcmBuffer.size - totalRead,
                    AudioRecord.READ_BLOCKING
                )
                if (read < 0) {
                    throw IllegalStateException("AudioRecord read failed: $read")
                }
                if (read == 0) {
                    break
                }
                totalRead += read
            }
            if (totalRead == 0) {
                throw IllegalStateException("未采集到任何音频数据")
            }

            val floatBuffer = FloatArray(desiredSamples)
            var sumAbs = 0f
            for (i in 0 until desiredSamples) {
                val sample = if (i < totalRead) pcmBuffer[i] else 0
                val normalized = sample * pcmNormalizer
                floatBuffer[i] = normalized
                if (i < totalRead) {
                    sumAbs += abs(normalized)
                }
            }
            val avgAmplitude = sumAbs / totalRead
            if (avgAmplitude < 1e-4f) {
                throw IllegalStateException("录音音量过低，请靠近麦克风或提高声源音量")
            }

            passtModule.classify(floatBuffer, totalRead)
        } finally {
            audioRecord.stop()
            audioRecord.release()
        }
    }

    @SuppressLint("MissingPermission")
    private fun buildRecorder(): AudioRecord {
        val minBufferBytes = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )
        val bufferBytes = maxOf(minBufferBytes, desiredSamples * 2)
        return AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufferBytes
        )
    }

    fun release() {
        passtModule.release()
    }
}
