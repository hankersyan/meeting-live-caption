---
name: add-line-breaks-smart
overview: 在 Transcriber 中添加智能按句换行逻辑，无需 UI 控件和配置项，默认在句末标点后自动换行。
todos:
  - id: add-format-function
    content: Add module-level format_line_breaks function before Transcriber class
    status: completed
  - id: modify-transcriber
    content: Apply format_line_breaks in Transcriber._transcribe_loop replacing the space-concat logic
    status: completed
    dependencies:
      - add-format-function
---

## Product Overview

为 Meeting Live Caption 添加智能换行功能，使实时转录文本在句末标点后自动换行，提升可读性。

## Core Features

- 在句末标点（. ! ? 等）后自动插入换行符，替代当前的连续空格拼接
- 换行逻辑同时作用于 UI 显示和文本文件保存
- 不需要 UI 控件，默认启用智能换行

## Tech Stack

- Python 3 + Tkinter（复用现有项目技术栈）

## Implementation Approach

在 `Transcriber` 类之前新增一个模块级函数 `format_line_breaks(text)`，使用正则在句末标点后插入 `\n`。然后在 `_transcribe_loop` 中将 `full_text.strip() + " "` 替换为调用该函数。

## Implementation Notes

- 正则匹配 `[.!?。！？]+(\s|$)` — 支持中英文句末标点，在其后插入换行
- 非句末句号（如缩写 "Mr.", "Dr."）可能被误判，但会议转录中较少，基础正则足够
- 每个转录 chunk 之间保留换行分隔，chunk 内部按句换行

## Architecture Design

仅修改 `main.py`，两处改动：

1. 新增 `format_line_breaks` 函数
2. `_transcribe_loop` 中调用该函数替换原有的空格拼接

### Directory Structure

```
c:\Test\captions\meeting-live-caption\
├── main.py       # [MODIFY] 新增 format_line_breaks 函数，修改 _transcribe_loop 中的文本格式化逻辑
```