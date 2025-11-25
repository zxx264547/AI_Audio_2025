# PaSST 场景识别安卓端

该 Android Studio 工程会在端侧实时采集 32 kHz/PCM16 音频，并将最近 10 秒窗口输入 PaSST 预训练模型，持续输出场景识别结果。

实时管线如下：
1. 使用 `AudioRecord` 持续采集 PCM16 数据，写入 10 秒长度的环形缓冲；
2. 每隔约 2 秒取出完整窗口并归一化成 `[1, samples]` Tensor；
3. 调用导出的 TorchScript (`passt_model.pt`) 得到 logits；
4. 经过 sigmoid 与标签映射，展示 Top-N 场景（支持中文标签 `labels_zh.csv`）。

## 准备 TorchScript 模型

1. 克隆 [PaSST](https://github.com/kkoutini/PaSST) 仓库并安装依赖：
   ```bash
   conda activate passt
   pip install hear21passt torch torchvision torchaudio
   ```
2. 在 `PaSST` 目录运行仓库提供的导出脚本：
   ```bash
   python export_torchscript.py --output ..\PaSSTApp\app\src\main\assets\passt_model.pt
   ```
   默认导出 `passt_s_swa_p16_128_ap476`，输入 10 秒 @ 32 kHz。可用 `--clip-seconds/--sample-rate` 调整，并同步更新 APP 内的录音参数。
3. 准备标签：
   - 将 AudioSet `class_labels_indices.csv` 复制为 `app/src/main/assets/labels.csv`（英文）。
   - 若需要中文，可执行 `python translate_labels.py` 生成 `labels_zh.csv`，应用会优先读取其中的 `zh_name` 列。

> **提示**：`tools/verify_torchscript.py` 可快速验证 `passt_model.pt` 是否可在 PC 上加载并输出 `(1, 527)` logits。

## 搭建与运行

1. 在 Android Studio 选择 “Open an Existing Project”，指向 `PaSSTApp`。
2. 工程已配置 **Gradle 8.9 / AGP 8.6.1 / Kotlin 1.9.24 / JDK 21**，同步即可。
3. 连接 ARM64 真机（示例：小米 14）：
   - 首次运行授予麦克风权限；
   - 点击“开始实时识别”后开始监听，如音量过低会提示；
   - 再次点击可停止监听。

## 工程结构

- `MainActivity`：权限、UI 以及实时开关逻辑；
- `AudioSceneAnalyzer`：PCM16 录音、环形缓冲、振幅检测、周期性推理；
- `PaSSTModule`：加载 TorchScript + 标签（英文/中文），输出 Top-N 文本；
- `assets/`：`passt_model.pt`、`labels.csv`、`labels_zh.csv`；
- `export_torchscript.py`、`tools/verify_torchscript.py`、`translate_labels.py`：PC 端导出/校验/翻译脚本。

## 常见问题

- **无法加载模型**：确认 `app/src/main/assets/passt_model.pt` 存在且可通过 `tools/verify_torchscript.py` 读取。
- **标签数量不符**：确保使用完整 527 行 CSV，且若自定义中文列请保留原始列顺序。
- **音量过低或录音失败**：应用会直接在 UI 提示，请检查麦克风权限/硬件；必要时调大声源。
- **`libpytorch_jni.so` 未找到**：依赖已切换为 `org.pytorch:pytorch_android:1.13.1`，在 ARM64 真机上即可加载；若需支持 x86，请改用对应 ABI 或 LFS 模型。

## 快速脚本

- 生成模型：`python export_torchscript.py --output ..\PaSSTApp\app\src\main\assets\passt_model.pt`
- 生成中文标签：`python translate_labels.py`
- 校验模型：`python tools/verify_torchscript.py`

如需进一步自定义（例如仅导出 Transformer、修改窗口长度、限制 ABI），可按需修改脚本以及 `AudioSceneAnalyzer`/`build.gradle`。
