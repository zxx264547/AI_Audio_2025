package com.example.passtapp;

import android.content.Context;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.regex.Pattern;
import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

public class PaSSTModule {

    private static final String MODEL_FILE = "passt_model.pt";
    private static final Pattern CSV_SPLIT_REGEX =
            Pattern.compile(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)");
    // AudioSet indices for scene classification
    private static final int IDX_SPEECH = 0;
    private static final int IDX_WIND = 285;
    private static final int[] IDXS_INDOOR = {506, 507, 508};
    private static final int[] IDXS_OUTDOOR = {509, 510};

    // Thresholds
    private static final float TH_SPEECH = 0.50f;
    private static final float TH_WIND = 0.25f;
    private static final float TH_INDOOR = 0.04f;
    private static final float TH_OUTDOOR = 0.04f;

    private final Context context;
    private final int expectedSamples;
    private Module module;
    private List<String> labels;
    private String backend = "CPU";

    public PaSSTModule(Context context, int sampleRate) {
        this.context = context.getApplicationContext();
        this.expectedSamples = sampleRate * 10;
    }

    public SceneResult classify(float[] buffer, int validSamples) {
        Module localModule = getModule();
        List<String> localLabels = getLabels();

        float[] waveform = new float[expectedSamples];
        int usableSamples = validSamples > 0 ? Math.min(validSamples, buffer.length) : buffer.length;
        int copyLength = Math.min(usableSamples, waveform.length);
        System.arraycopy(buffer, 0, waveform, 0, copyLength);

        Tensor inputTensor = Tensor.fromBlob(waveform, new long[] {1, waveform.length});
        float[] logits = localModule.forward(IValue.from(inputTensor)).toTensor().getDataAsFloatArray();
        float[] probs = buildProbabilities(logits);
        SceneClassification scene = classifyScene(probs);
        List<Prediction> predictions = buildPredictions(probs, localLabels);
        return new SceneResult(predictions, scene);
    }

    private float[] buildProbabilities(float[] logits) {
        if (logits == null) {
            return new float[0];
        }
        float[] probs = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            probs[i] = sigmoid(logits[i]);
        }
        return probs;
    }

    private List<Prediction> buildPredictions(float[] probs, List<String> localLabels) {
        if (probs == null || probs.length == 0) {
            return Collections.emptyList();
        }
        List<Integer> indices = new ArrayList<>(probs.length);
        for (int i = 0; i < probs.length; i++) {
            indices.add(i);
        }
        indices.sort((a, b) -> Float.compare(probs[b], probs[a]));

        List<Prediction> results = new ArrayList<>(5);
        int limit = Math.min(5, indices.size());
        for (int i = 0; i < limit; i++) {
            int idx = indices.get(i);
            String label = idx < localLabels.size() ? localLabels.get(idx) : "unknown#" + idx;
            results.add(new Prediction(label, probs[idx]));
        }
        return results;
    }

    private synchronized Module getModule() {
        if (module == null) {
            module = loadModuleSafely(MODEL_FILE);
        }
        return module;
    }

    private synchronized List<String> getLabels() {
        if (labels == null) {
            labels = loadLabels();
        }
        return labels;
    }

    public String getBackendName() {
        getModule(); // ensure initialized
        return backend;
    }

    private Module loadModuleSafely(String fileName) {
        String filePath = copyAsset(fileName);
        try {
            backend = "CPU";
            return Module.load(filePath);
        } catch (Exception ex) {
            throw new IllegalStateException(
                    "Failed to load model. Ensure TorchScript file " + fileName + " exists in assets.",
                    ex);
        }
    }

    private List<String> loadLabels() {
        List<String> candidates = Arrays.asList("labels_zh.csv", "labels.csv");
        List<String> errors = new ArrayList<>();
        for (String file : candidates) {
            try {
                return readLabelsFromFile(file);
            } catch (IOException ex) {
                errors.add(file + ": " + ex.getLocalizedMessage());
            }
        }
        throw new IllegalStateException(
                "Unable to load labels.csv/labels_zh.csv from assets. Errors: "
                        + String.join("; ", errors));
    }

    private List<String> readLabelsFromFile(String fileName) throws IOException {
        List<String> result = new ArrayList<>();
        try (java.io.InputStream input = context.getAssets().open(fileName);
                java.io.BufferedReader lines =
                        new java.io.BufferedReader(new java.io.InputStreamReader(input))) {
            // skip header
            lines.readLine();
            String line;
            while ((line = lines.readLine()) != null) {
                if (line.trim().isEmpty()) {
                    continue;
                }
                String[] tokens = CSV_SPLIT_REGEX.split(line);
                if (tokens.length >= 4 && !tokens[3].trim().isEmpty()) {
                    result.add(tokens[3].trim().replace("\"", ""));
                } else if (tokens.length >= 3) {
                    result.add(tokens[2].trim().replace("\"", ""));
                } else if (tokens.length >= 2) {
                    result.add(tokens[1].trim().replace("\"", ""));
                } else {
                    result.add(line.trim());
                }
            }
        }
        return result;
    }

    private String copyAsset(String assetName) {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }
        try (java.io.InputStream input = context.getAssets().open(assetName);
                FileOutputStream output = new FileOutputStream(file)) {
            byte[] buffer = new byte[8 * 1024];
            int read;
            while ((read = input.read(buffer)) != -1) {
                output.write(buffer, 0, read);
            }
            output.flush();
        } catch (IOException ex) {
            throw new IllegalStateException(
                    "Missing asset " + assetName + ". Place it under app/src/main/assets.", ex);
        }
        return file.getAbsolutePath();
    }

    public void release() {
        synchronized (this) {
            if (module != null) {
                try {
                    module.destroy();
                } catch (Exception ignored) {
                    // ignore cleanup errors
                }
                module = null;
            }
        }
    }

    private static float sigmoid(float value) {
        float expValue = (float) Math.exp(value);
        return expValue / (1f + expValue);
    }

    private SceneClassification classifyScene(float[] probs) {
        if (probs == null || probs.length <= IDX_WIND) {
            return new SceneClassification("未知", "概率维度不足");
        }
        float pSpeech = probs[IDX_SPEECH];
        float pWind = probs[IDX_WIND];
        float pIndoorMax = maxAt(probs, IDXS_INDOOR);
        float pOutdoorMax = maxAt(probs, IDXS_OUTDOOR);

        boolean hasSpeech = pSpeech >= TH_SPEECH;
        boolean hasIndoor = pIndoorMax >= TH_INDOOR;
        boolean hasWind = pWind >= TH_WIND;
        boolean hasOutdoor = pOutdoorMax >= TH_OUTDOOR;

        String scene;
        if (hasSpeech && hasIndoor) {
            scene = "会议模式";
        } else if (hasWind && hasOutdoor) {
            scene = "户外降噪";
        } else {
            scene = "标准降噪";
        }

        String debug =
                "scene="
                        + scene
                        + " | speech="
                        + fmt(pSpeech, hasSpeech, TH_SPEECH)
                        + " | indoorMax="
                        + fmt(pIndoorMax, hasIndoor, TH_INDOOR)
                        + " | wind="
                        + fmt(pWind, hasWind, TH_WIND)
                        + " | outdoorMax="
                        + fmt(pOutdoorMax, hasOutdoor, TH_OUTDOOR);
        return new SceneClassification(scene, debug);
    }

    private static float maxAt(float[] probs, int[] indices) {
        float max = 0f;
        for (int idx : indices) {
            if (idx >= 0 && idx < probs.length) {
                max = Math.max(max, probs[idx]);
            }
        }
        return max;
    }

    private static String fmt(float value, boolean hit, float th) {
        return String.format(Locale.getDefault(), "%.3f/%s(th=%.2f)", value, hit ? "√" : "×", th);
    }
}

class Prediction {
    private final String label;
    private final float confidence;

    Prediction(String label, float confidence) {
        this.label = label;
        this.confidence = confidence;
    }

    public String getLabel() {
        return label;
    }

    public float getConfidence() {
        return confidence;
    }
}

class SceneResult {
    private final List<Prediction> predictions;
    private final SceneClassification scene;

    SceneResult(List<Prediction> predictions, SceneClassification scene) {
        this.predictions = predictions;
        this.scene = scene;
    }

    public List<Prediction> getPredictions() {
        return predictions;
    }

    public SceneClassification getScene() {
        return scene;
    }

    public String formatForDisplay() {
        if (predictions == null || predictions.isEmpty()) {
            return "暂无预测结果";
        }
        StringBuilder builder = new StringBuilder();
        if (scene != null) {
            builder.append("模式: ").append(scene.getScene()).append('\n');
        }
        for (int i = 0; i < predictions.size(); i++) {
            Prediction prediction = predictions.get(i);
            builder.append(i + 1)
                    .append(". ")
                    .append(prediction.getLabel())
                    .append("  confidence=")
                    .append(String.format(Locale.getDefault(), "%.2f", prediction.getConfidence()))
                    .append('\n');
        }
        return builder.toString().trim();
    }
}

class SceneClassification {
    private final String scene;
    private final String debug;

    SceneClassification(String scene, String debug) {
        this.scene = scene;
        this.debug = debug;
    }

    public String getScene() {
        return scene;
    }

    public String getDebug() {
        return debug;
    }
}
