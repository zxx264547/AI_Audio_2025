package com.example.passtapp

import android.content.Context
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.Locale
import kotlin.math.exp
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor

class PaSSTModule(private val context: Context, sampleRate: Int) {

    private val expectedSamples = sampleRate * 10
    private val moduleHolder = lazy { loadModuleSafely(MODEL_FILE) }
    private val labelsHolder = lazy { loadLabels() }

    private val module: Module
        get() = moduleHolder.value
    private val labels: List<String>
        get() = labelsHolder.value

    fun classify(buffer: FloatArray, validSamples: Int): SceneResult {
        val waveform = FloatArray(expectedSamples)
        val usableSamples = if (validSamples > 0) minOf(validSamples, buffer.size) else buffer.size
        val copyLength = minOf(usableSamples, waveform.size)
        System.arraycopy(buffer, 0, waveform, 0, copyLength)

        val inputTensor = Tensor.fromBlob(
            waveform,
            longArrayOf(1, waveform.size.toLong())
        )
        val logits = module.forward(IValue.from(inputTensor)).toTensor().dataAsFloatArray
        val predictions = buildPredictions(logits)
        return SceneResult(predictions)
    }

    private fun buildPredictions(logits: FloatArray): List<Prediction> {
        if (logits.isEmpty()) {
            return emptyList()
        }
        val confidences = FloatArray(logits.size) { index -> sigmoid(logits[index]) }
        return confidences.indices
            .sortedByDescending { confidences[it] }
            .take(3)
            .map { idx ->
                val label = labels.getOrElse(idx) { "类别#${idx}" }
                Prediction(label, confidences[idx])
            }
    }

    private fun loadModuleSafely(fileName: String): Module {
        val filePath = copyAsset(fileName)
        return try {
            Module.load(filePath)
        } catch (ex: Exception) {
            throw IllegalStateException(
                "无法加载模型，请确认已将 TorchScript 文件命名为 $fileName 并放在 assets 目录下。",
                ex
            )
        }
    }

    private fun loadLabels(): List<String> {
        val candidates = listOf("labels_zh.csv", "labels.csv")
        val errors = mutableListOf<String>()
        for (file in candidates) {
            try {
                return readLabelsFromFile(file)
            } catch (ex: IOException) {
                errors.add("$file: ${ex.localizedMessage}")
            }
        }
        throw IllegalStateException(
            "无法加载 labels.csv/labels_zh.csv，请确认文件存在于 assets 目录。错误: ${errors.joinToString()}"
        )
    }

    private fun readLabelsFromFile(fileName: String): List<String> {
        return context.assets.open(fileName).bufferedReader().useLines { lines ->
            lines.drop(1)
                .filter { it.isNotBlank() }
                .map { line ->
                    val tokens = CSV_SPLIT_REGEX.split(line)
                    when {
                        tokens.size >= 4 && tokens[3].isNotBlank() -> tokens[3].trim().trim('"')
                        tokens.size >= 3 -> tokens[2].trim().trim('"')
                        tokens.size >= 2 -> tokens[1].trim().trim('"')
                        else -> line.trim()
                    }
                }
                .toList()
        }
    }

    private fun copyAsset(assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }
        try {
            context.assets.open(assetName).use { input ->
                FileOutputStream(file).use { output ->
                    input.copyTo(output)
                }
            }
        } catch (ex: IOException) {
            throw IllegalStateException(
                "无法找到 $assetName，请将文件放在 app/src/main/assets 之后重新构建。",
                ex
            )
        }
        return file.absolutePath
    }

    fun release() {
        if (moduleHolder.isInitialized()) {
            try {
                module.destroy()
            } catch (_: Exception) {
            }
        }
    }

    companion object {
        private const val MODEL_FILE = "passt_model.pt"
        private val CSV_SPLIT_REGEX = ",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)".toRegex()

        private fun sigmoid(value: Float): Float {
            val expValue = exp(value.toDouble()).toFloat()
            return expValue / (1f + expValue)
        }
    }
}

data class Prediction(val label: String, val confidence: Float)

data class SceneResult(private val predictions: List<Prediction>) {
    fun formatForDisplay(): String {
        if (predictions.isEmpty()) {
            return "未得到模型输出"
        }
        val builder = StringBuilder()
        predictions.forEachIndexed { index, prediction ->
            builder.append(index + 1)
                .append(". ")
                .append(prediction.label)
                .append(" 置信度=")
                .append(String.format(Locale.getDefault(), "%.2f", prediction.confidence))
                .append('\n')
        }
        return builder.toString().trim()
    }
}
