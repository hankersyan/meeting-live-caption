---
name: Fix UI Freeze on Long Runs
overview: 诊断并修复 main.py 中因长时间运行导致 UI 卡死的问题。根本原因是转录文本在内存和 tkinter 文本组件中无限累积，导致内存膨胀和界面渲染变慢。方案包括：限制用于 Ollama 摘要的内存历史文本长度，并定期裁剪 UI 文本组件中的旧内容，同时保留磁盘上的完整转录文件。
todos:
  - id: add-text-bounds
    content: 在 MeetingRecorderApp 中增加内存历史上限与 UI 显示上限常量及截断逻辑
    status: completed
  - id: batch-ui-flush
    content: 将逐条 UI 刷新改为 pending_buffer + 定时批量刷新，减少主线程调度压力
    status: completed
    dependencies:
      - add-text-bounds
  - id: limit-keypoints-area
    content: 为 key_points_area 增加显示上限，防止关键要点无限累积
    status: completed
    dependencies:
      - add-text-bounds
---

## 用户反馈

长时间运行时 UI 出现卡死（freeze）。

## 问题分析

经过代码审查，确认以下导致长时间运行卡死的根因：

1. **内存转录文本无限增长**：`MeetingRecorderApp.full_transcription_text` 在 `append_text()` 中通过 `+= text` 无限累积，最终被完整传递给 Ollama 做 key-point extraction，请求体随时间线性膨胀。
2. **UI 文本框无限增长**：`text_area`（`ScrolledText`）持续插入新文本，tkinter 在大文本量下渲染和重排性能急剧下降。
3. **无清理逻辑**：`caption_total_chars` 虽在增长，但没有触发任何截断或清理动作。

## 修复目标

- 限制用于 Ollama key-point extraction 的内存历史文本长度（保留最近 N 字符）。
- 限制 `ScrolledText` 显示内容的最大长度，超出时自动截断顶部旧内容，保持 UI 流畅。
- 磁盘上的 `.txt` 转录文件继续保留完整内容（`_save_text` 不受此影响）。
- 保持现有功能与界面不变，仅优化长时间运行的稳定性。

## 技术栈

- Python 3.7+
- tkinter（内置 GUI）
- faster-whisper、pyaudiowpatch、numpy

## 实现方案

### 核心策略

在 `MeetingRecorderApp` 中引入双上限机制：**内存历史上限** + **UI 显示上限**。

1. **内存历史上限**：为 `full_transcription_text` 设置一个合理的最大字符数（例如 15000 字符）。当新文本追加后超出上限时，从头部截断旧内容。这样传给 Ollama 的 `captions_text` 始终有界，避免请求体膨胀。
2. **UI 显示上限**：为 `text_area` 设置最大行数或最大字符数上限（例如 10000 行或 200000 字符）。当内容超出上限时，删除顶部旧文本。`ScrolledText` 的 `delete(1.0, index)` 配合 `see(END)` 可保持流畅。
3. **批量/合并刷新**：`append_text` 每收到一个 transcription chunk 就通过 `root.after(0, ...)` 触发一次 UI 更新。长时间运行后调用次数极多。改为在 `append_text` 中先将文本暂存到一个线程安全的 pending buffer，再由一个定时器（如每 200ms）统一刷入 UI，减少 tkinter 主线程调度压力。
4. **关键要点区域上限**：同理对 `key_points_area` 设置上限，防止关键要点无限累积。

### 性能与可靠性

- 截断操作使用字符串切片和 `tk.Text.delete`，时间复杂度 O(1) ~ O(n)，仅在达到阈值时触发，不引入持续开销。
- pending buffer 使用 `threading.Lock` 保护，与现有 `caption_lock` 保持一致。
- 保留完整的磁盘写入逻辑（`_save_text`），用户文件不受任何影响。

## 架构设计

无需改动整体多线程架构。修改集中在 `MeetingRecorderApp` 的数据管理与 UI 刷新路径：

```
AudioRecorder → audio_queue → Transcriber → text_callback
                                                   ↓
                                      [caption_lock] + pending_buffer
                                                   ↓
                                           root.after(batch_flush)
                                                   ↓
                                        text_area (bounded)
                                                   ↓
                                    get_recent_captions (bounded)
                                                   ↓
                                     KeyPointExtractor → Ollama
```

## 涉及文件

- `c:/Test/captions/meeting-live-caption/main.py`：[MODIFY] 增加文本上限、批量刷新、截断逻辑。