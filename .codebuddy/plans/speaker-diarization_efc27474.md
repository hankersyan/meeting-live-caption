---
name: speaker-diarization
overview: 为 Meeting Live Caption 添加基于 pyannote.audio 的说话人分离（Speaker Diarization）功能，在 UI 中近实时显示说话人标签（Speaker 1 / Speaker 2 ...），并写入保存的 TXT 文件。
todos:
  - id: transcriber-segment-store
    content: 改造 Transcriber：存储带时间戳的转录片段列表，新增 SegmentStore 和 AudioBuffer 数据结构
    status: completed
  - id: speaker-diarizer-class
    content: 创建 SpeakerDiarizer 类（pyannote 管线、周期性后台线程、音频缓冲读取）和 LabelMapper 映射逻辑
    status: completed
    dependencies:
      - transcriber-segment-store
  - id: ui-controls-and-display
    content: 添加 UI 控制区域（启用开关、HF Token 输入、间隔设置）和说话人标签显示区域
    status: completed
    dependencies:
      - speaker-diarizer-class
  - id: app-integration
    content: 集成到 MeetingRecorderApp 流程（加载/启动/停止/保存 TXT 含标签/配置持久化）
    status: completed
    dependencies:
      - ui-controls-and-display
  - id: update-docs
    content: 更新 README.md、USAGE.md、TECHNICAL_DOCS.md 和 config.json
    status: completed
---

## 产品概述

为现有的实时会议录音/转录应用 Meeting Live Caption 添加说话人分离（Speaker Diarization）功能，能够自动识别并标记不同说话人的语音段落，在转录文本中显示说话人标签（如 "Speaker 1"、"Speaker 2"）。

## 核心功能

- 基于 pyannote.audio 的说话人分离引擎，在录音过程中周期性运行
- 在 UI 中以近实时方式显示带说话人标签的转录文本
- 在保存的 TXT 文件中包含说话人标签
- 支持开关控制和 HuggingFace Token 配置
- 支持可调节的分离间隔和说话人数上限设置

## 技术栈

### 新增依赖

- **pyannote.audio 3.1+**: 说话人分离管线，基于 PyTorch
- **torch**: PyTorch 深度学习框架（pyannote 依赖）
- **huggingface_hub**: 用于访问 HuggingFace 模型仓库和认证

### 技术兼容性说明

- 当前项目使用 `faster-whisper` (基于 `ctranslate2`)，新增 `torch` 不会有直接冲突
- pyannote 默认使用 CPU 推理，如需 GPU 加速需安装 CUDA 版 torch
- 两个框架有独立的 CUDA 上下文，可共存但需注意 GPU 显存分配

## 实现方案

### 整体策略

采用 **周期性批处理 + 滑动窗口** 的方式实现近实时说话人分离：

1. **音频积累**：在录音过程中持续累积原始音频数据到内存环形缓冲区（滑动窗口，默认保留最新 120 秒）
2. **周期性分离**：每 N 秒（默认 30 秒）在后台线程中对累积音频运行 pyannote 管线
3. **标签映射**：将说话人分离结果（时间区间 + 说话人 ID）与 Whisper 转录片段（时间戳 + 文本）进行时间轴对齐映射
4. **UI 更新**：将映射后的带标签转录文本输出到 UI 的专属区域

### 架构设计

```mermaid
flowchart TD
    AR[AudioRecorder] -->|audio chunks| AC[AudioCollector<br/>环形缓冲区]
    AR -->|audio chunks| TR[Transcriber]
    TR -->|segments: (start,end,text)| SS[SegmentStore<br/>线程安全存储]
    TR -->|raw text| UI[Live Transcription]
    
    AC -->|accumulated audio| SD[SpeakerDiarizer<br/>周期性后台线程]
    SD -->|pyannote pipeline| DR[diarization results<br/>(start,end,speaker)]
    DR --> LM[LabelMapper]
    SS -->|transcription segments| LM
    LM -->|labeled segments| UI2[Speaker-labeled<br/>Transcription]
    LM -->|labeled segments| TF[Text File<br/>with speaker labels]
    
    subgraph "每 N 秒触发一次"
        AC
        SD
        DR
        LM
    end
```

### 核心数据结构

```python
# 转录片段（Transcriber 产出）
@dataclass
class TranscriptionSegment:
    start: float       # 起始时间（秒）
    end: float         # 结束时间（秒）
    text: str          # 文本内容
    speaker: str       # 说话人标签（初始为 None，由 LabelMapper 填充）

# 说话人分离片段（SpeakerDiarizer 产出）
@dataclass
class DiarizationSegment:
    start: float       # 起始时间（秒）
    end: float         # 结束时间（秒）
    speaker: str       # 说话人标签，如 "SPEAKER_01"
```

### 组件设计

#### 1. Transcriber 改造

- 将原本只拼接文本的逻辑改为存储 `List[TranscriptionSegment]`
- 记录每个 segment 的 `start` 和 `end` 时间戳（faster-whisper 的 segment 对象已包含这些属性）
- 提供线程安全的 `get_segments()` 方法供 LabelMapper 读取
- 同时保留现有实时文本回调，确保原始转录文本持续显示

#### 2. AudioCollector（新增）

- 运行在 AudioRecorder 侧，收集原始 mono int16 音频数据
- 使用 numpy 数组作为环形缓冲区，限制最大时长（默认 120 秒）
- 提供 `get_audio() -> np.ndarray` 方法供 SpeakerDiarizer 提取当前缓冲区

#### 3. SpeakerDiarizer（新增）

```python
class SpeakerDiarizer:
    def __init__(self, hf_token, audio_provider, interval=30, device="cpu", max_speakers=None):
        # hf_token: HuggingFace 访问令牌
        # audio_provider: 返回当前音频缓冲区的回调
        # interval: 分离运行间隔（秒）
        # device: 推理设备 ("cpu" 或 "cuda")
        # max_speakers: 可选，说话人数上限
        
    def start(self):   # 启动后台线程
    def stop(self):    # 停止并等待线程结束
    def get_latest_mapping(self) -> Dict[Tuple[float,float], str]:  # 获取最新说话人映射
```

- 在 `start()` 中加载 pyannote 管线（首次加载可能较慢，需在后台进行）
- 后台循环：等待 interval 秒 → 调用 audio_provider 获取音频 → 运行 pyannote 管线 → 更新映射结果
- 使用 threading.Event 控制停止信号
- 线程安全地存储最新分离结果

#### 4. LabelMapper（逻辑封装）

- 接收转录片段列表和说话人分离映射
- 对于每个转录片段，计算与哪些说话人时间段重叠
- 取重叠比例最高的说话人作为该片段的标签
- 返回带标签的文本行列表

#### 5. UI 集成

- 在 "Brief Key Points" 区域下方或旁边新增 "Speaker-labeled Transcription" 文本显示区域
- 或：改造现有 "Live Transcription" 区域，在分离结果到达后插入说话人标签前缀
- 推荐方案：新增独立的标签显示区域，保留原始转录不变

#### 6. 文件输出改造

- TXT 文件格式改为包含说话人标签：

```
[Speaker 1] Hello, welcome to the meeting.
[Speaker 2] Thank you, let's start with the agenda.
[Speaker 1] First item is the quarterly report.
```

- 文件保存分为两个阶段：实时写入原始文本 + 最终写入带标签的完整版

### 性能与可靠性考量

| 关注点 | 方案 |
| --- | --- |
| pyannote 推理延迟 | 默认 30 秒间隔，60 秒滑动窗口，CPU 推理约 3-8 秒完成 |
| 首次加载时间 | pyannote 管线首次加载约 10-30 秒，在后台线程异步加载 |
| 内存占用 | 音频缓冲区最大 120 秒 × 16000Hz × 2 bytes ≈ 3.8MB |
| torch 与 ctranslate2 共存 | pyannote 固定使用 CPU 避免 GPU 竞争，可选 CUDA |
| 线程安全 | 所有共享数据使用 threading.Lock 保护 |
| 短录音场景 | 录音时长 < 分离间隔时不运行分离，避免无效计算 |


### 关键设计决策

1. **滑动窗口机制**：限制每次处理的音频长度，避免全量处理导致延迟累积
2. **分离频率 < 转录频率**：转录是连续的，分离是周期性的，确保转录不中断
3. **标签映射时机**：分离完成后再映射历史片段，不影响当前正在进行的转录
4. **降级策略**：分离失败时，转录继续正常运行，用户只看到原始文本

### 目录结构

```
meeting-live-caption/
├── main.py              # [MODIFY] 集成说话人分离的完整逻辑
│   ├── AudioRecorder    # 新增 AudioProvider 接口（提供累积音频）
│   ├── Transcriber      # 改造：存储分段转录信息
│   ├── SpeakerDiarizer  # [NEW] 说话人分离类
│   ├── LabelMapper      # [NEW] 标签映射逻辑
│   └── MeetingRecorderApp # 改造：新增 UI 控制和显示区域
├── config.json          # [MODIFY] 新增说话人分离相关配置项
├── README.md            # [MODIFY] 更新功能列表
├── USAGE.md             # [MODIFY] 新增使用说明
└── TECHNICAL_DOCS.md    # [MODIFY] 更新技术文档
```

### 配置变更 (config.json)

新增配置项:

```
{
  "diarization_enabled": false,
  "hf_token": "",
  "diarization_interval": 30,
  "max_speakers": 0,
  "diarization_device": "cpu"
}
```

### 关键代码结构

```python
# SegmentStore - 线程安全的转录片段存储
class SegmentStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._segments: List[TranscriptionSegment] = []
    
    def add_segment(self, segment: TranscriptionSegment):
        with self._lock:
            self._segments.append(segment)
    
    def get_segments_since(self, since_time: float = 0) -> List[TranscriptionSegment]:
        with self._lock:
            return [s for s in self._segments if s.end >= since_time]
    
    def get_all_segments(self) -> List[TranscriptionSegment]:
        with self._lock:
            return list(self._segments)

# AudioBuffer - 环形音频缓冲区
class AudioBuffer:
    def __init__(self, max_duration=120.0, sample_rate=16000):
        self.max_samples = int(max_duration * sample_rate)
        self.sample_rate = sample_rate
        self._buffer = np.array([], dtype=np.int16)
        self._lock = threading.Lock()
    
    def append(self, chunk: np.ndarray):
        with self._lock:
            self._buffer = np.concatenate([self._buffer, chunk])
            if len(self._buffer) > self.max_samples:
                self._buffer = self._buffer[-self.max_samples:]
    
    def get_audio(self) -> np.ndarray:
        with self._lock:
            return self._buffer.copy()
    
    def clear(self):
        with self._lock:
            self._buffer = np.array([], dtype=np.int16)
    
    def duration(self) -> float:
        with self._lock:
            return len(self._buffer) / self.sample_rate
```

# Agent Extensions

无需使用扩展，本任务为纯 Python 后端功能开发，不涉及技能列表中包含的文档处理、浏览器自动化或多模态生成能力。