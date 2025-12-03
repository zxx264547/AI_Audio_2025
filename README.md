# PaSST Android Scene Detector

Android app that captures 32 kHz PCM16 audio, feeds 10 s clips to a PaSST TorchScript model, and displays the top labels plus an inferred noise-reduction mode.

## What it does
- Real-time capture via `AudioRecord`, ring buffer sized for 10 s.
- TorchScript inference (`app/src/main/assets/passt_model.pt`) with labels from `labels.csv` / `labels_zh.csv`.
- Scene classification (priority: Meeting > Outdoor > Standard):
  - Speech idx 0 ≥ 0.50 and Indoor max idx 506/507/508 ≥ 0.04 → Meeting mode.
  - Wind idx 285 ≥ 0.25 and Outdoor max idx 509/510 ≥ 0.04 → Outdoor mode.
  - Otherwise Standard mode.
- Mode-specific noise processing before inference:
  - Standard: no gate, no smoothing.
  - Meeting: light gate (~0.003) + 3-point smoothing.
  - Outdoor: stronger gate (~0.008) + 5-point smoothing.
- UI shows the current mode (large, bold), per-label confidences, and per-mode decision lines (with thresholds).
- Snackbar on mode changes and play/export actions.
- Playback/export buttons:
  - Play & export raw 10 s buffer (WAV to `.../files/Music`).
  - Play & export denoised 10 s buffer (same path).
  - Playback pauses streaming first to avoid conflicts.

## Files of interest
- `app/src/main/java/com/example/passtapp/MainActivity.java`: permissions, UI wiring, mode/result rendering, Snackbar prompts, playback triggers.
- `app/src/main/java/com/example/passtapp/AudioSceneAnalyzer.java`: audio capture, ring buffer, noise processing per mode, asynchronous inference, playback + WAV export.
- `app/src/main/java/com/example/passtapp/PaSSTModule.java`: model/label loading, logits→probs, top-5 predictions, scene classification with debug lines.
- `app/src/main/res/layout/activity_main.xml`: buttons for start/stop, play raw, play denoised; status/result text.
- `app/src/main/res/values/strings.xml`: UI strings (Chinese).
- `app/src/main/assets/`: place `passt_model.pt`, `labels.csv`, `labels_zh.csv`.

## Requirements
- Android Studio with Gradle 8.9 / AGP 8.6.1 / JDK 21 (as configured).
- Device: ARM64 recommended, API 24+, RECORD_AUDIO permission.
- TorchScript model exported for 10 s @ 32 kHz mono to match the app’s capture settings.

## Usage
1) Put `passt_model.pt` and labels into `app/src/main/assets/`.
2) Build/install. On first run grant microphone permission.
3) Tap “开始实时识别” to start; app shows mode, top labels, and decision lines; Snackbar on mode switches.
4) Buttons “播放降噪前声音” / “播放降噪后声音” pause streaming and play the respective buffer; use the “保存当前音频” button to export paired raw/denoised WAV files with matching names and shown paths.

## Notes / Troubleshooting
- If model fails to load, check the asset path and verify with your own PC script.
- If labels misalign, ensure CSV has 527 entries in AudioSet order.
- WAV export uses app external files (Music) when available; otherwise internal files dir.
- NNAPI/GPU: current build loads on CPU; NNAPI reflection hook present in `PaSSTModule` scaffold but not enforced.
