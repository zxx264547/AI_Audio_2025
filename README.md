# PaSST 场景识别安卓端

这个 Android Studio 工程在端侧调用 PaSST 预训练模型，对 10 秒麦克风音频做声景识别。核心流程：

1. 32 kHz/PCM16 录制 10 秒波形；
2. 归一化成 `[1, samples]` Tensor；
3. 调用导出的 TorchScript (`passt_model.pt`) 得到 logits；
4. 经过 sigmoid 与标签映射，输出 Top-N 场景及置信度。

## 准备 TorchScript 模型

1. 克隆 [PaSST](https://github.com/kkoutini/PaSST) 仓库并安装依赖：
   ```bash
   conda activate passt
   pip install hear21passt torch torchvision torchaudio
   ```
2. 在 `PaSST` 目录运行导出脚本（仓库随附 `export_torchscript.py`）：
   ```bash
   python export_torchscript.py --output ..\PaSSTApp\app\src\main\assets\passt_model.pt
   ```
   默认导出 `passt_s_swa_p16_128_ap476`，输入 10 秒 @ 32 kHz。可用 `--clip-seconds/--sample-rate` 调整，并同步更新 APP 内的录音参数。
3. 拷贝 AudioSet 的 `class_labels_indices.csv` 为 `app/src/main/assets/labels.csv`，用于 527 个标签映射。

> **提示**：仓库提供 `tools/verify_torchscript.py`，执行 `python tools/verify_torchscript.py` 可快速确认 `passt_model.pt` 能否在 PC 上加载并输出 `(1, 527)` logits。

## 搭建与运行

1. Android Studio 选择 “Open an Existing Project”，指向 `PaSSTApp`。
2. 使用 **Gradle 8.9 / AGP 8.6.1 / Kotlin 1.9.24 / JDK 21** 同步（工程已配置）。
3. 连接 ARM64 真机（例如小米 14）：
   - 首次运行授予麦克风权限。
   - 点击 “录制并识别” 后，应用在 IO 线程录音并检测音量，若未读到数据或音量过低会提示重录。
   - 成功后显示置信度最高的 3 个场景。

## 工程结构

- `MainActivity`：UI、权限与协程调度。
- `AudioSceneAnalyzer`：PCM16 录音、归一化、振幅检测与异常提示。
- `PaSSTModule`：加载 TorchScript + 标签，输出 Top-N 文本；支持懒加载与资源释放。
- `assets/`：`passt_model.pt` 与 `labels.csv`。
- `export_torchscript.py` & `tools/verify_torchscript.py`：PC 端导出/校验脚本。

## 常见问题

- **无法加载模型**：确认 `app/src/main/assets/passt_model.pt` 存在且能被 `tools/verify_torchscript.py` 读取。
- **标签数量不符**：需使用完整 527 行的 `class_labels_indices.csv`。
- **录音全 0/结果固定**：应用会检测 `AudioRecord` 返回值与平均振幅；若提示音量过低请靠近麦克风或提高声源。
- **libpytorch_jni.so not found**：确保依赖为 `org.pytorch:pytorch_android:1.13.1`（工程已设置），或仅在 ARM64 设备上运行。

## 快速导出/调试

- `python export_torchscript.py --output ..\PaSSTApp\app\src\main\assets\passt_model.pt`
- `python tools/verify_torchscript.py`

更多自定义（只导 Transformer、改录音长度、裁剪 ABI 等）可按需修改脚本与 `AudioSceneAnalyzer`/`build.gradle`。

# AI_Audio_2025
