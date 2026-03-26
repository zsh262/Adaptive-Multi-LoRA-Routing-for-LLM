# 多模态语音助手系统（Speech + LLM + RAG/LoRA + Memory）

## 一、项目概述

本项目目标是构建一个 **多模态智能语音助手系统**，支持：

- 🎤 **语音输入**（自然对话）
- 🧠 **智能理解**（LLM解析用户意图）
- 🔗 **任务执行**（调用已有系统API：论文RAG / Multi-LoRA）
- 🔊 **语音输出**（TTS）
- 📝 **历史记忆**（上下文管理、历史查询）

本项目的核心创新点：

1. **多模态输入/输出**：语音 ↔ LLM ↔ 语音
2. **动态任务拆解**：通过 LLM 将自然语言解析成结构化任务 JSON
3. **历史记忆**：上下文保留，支持长对话、连续任务
4. **可复用接口**：无缝调用前两个项目（Paper-RAG、Multi-LoRA）
5. **无需额外训练**：依赖现有 LLM + Whisper + TTS，即可实现智能理解

------

e:\agent_Speech_System03\
├── web/
│   ├── static/
│   │   ├── index.html      # 主工作区 - 实时语音对话界面
│   │   ├── settings.html   # 系统设置页面
│   │   └── history.html    # 历史记录管理页面
│   └── server.py           # Flask Web 服务器
├── asr/
├── llm/
├── router/
├── memory/
├── tts/
├── demo/
├── main.py
├── config.py
└── requirements.txt


## 二、系统架构

```
用户语音输入
       │
       ▼
  语音识别 ASR
   (Whisper)
       │
       ▼
     文本
       │
       ▼
   LLM 解析意图
   - 拆解任务
   - 判断调用哪个系统
       │
       ▼
 任务调度 / API 调用
 ├─ Paper-RAG API（论文问答）
 └─ Multi-LoRA API（领域专家回答）
       │
       ▼
   生成文本回答
       │
       ▼
  TTS 合成语音输出
       │
       ▼
 用户听到回答
```

### 核心模块

| 模块        | 功能                     | 技术/工具                    |
| ----------- | ------------------------ | ---------------------------- |
| ASR 输入    | 将语音转为文本           | Whisper                      |
| LLM 理解    | 拆解用户意图为任务 JSON  | Qwen / TinyLlama             |
| Task Router | 判断调用哪个系统         | JSON 路由 + Embedding 或规则 |
| API 调用    | 调用前两个系统接口       | Paper-RAG / Multi-LoRA       |
| Memory 管理 | 保留历史对话，用于上下文 | List / Vector DB + 检索      |
| TTS 输出    | 将文本回答转换为语音     | VITS / 实习模型              |
| Demo 展示   | 前端交互                 | Gradio / Streamlit           |

------

## 三、数据处理与历史管理

1. **无需训练数据集**
   - Whisper 和 LLM 已自带能力，无需额外标注或训练
   - 历史上下文可以通过 prompt 拼接或者向量数据库检索
2. **历史存储方案**

| 方法               | 优缺点             | 实现方式                                             |
| ------------------ | ------------------ | ---------------------------------------------------- |
| List + Prompt 拼接 | 简单               | `history = [user_input, llm_output]` 拼入 prompt     |
| Vector DB + RAG    | 支持长对话，跨会话 | 每条对话做 embedding，检索 Top-K 相关历史拼入 prompt |

------

## 四、核心流程

### 1. 语音识别

```
import whisper

model = whisper.load_model("base")
result = model.transcribe("user_audio.wav")
text_input = result["text"]
```

------

### 2. LLM 解析任务

- 用户自然语言 → 拆解成 **任务 JSON**
- 示例：

```
用户说：“帮我找两篇 diffusion 的论文，然后对比方法”
```

LLM 输出：

```
[
  {"task": "search_paper", "query": "diffusion", "topk": 2},
  {"task": "compare", "target": "methods"}
]
```

------

### 3. Task Router 调度

- 根据任务 JSON，选择调用的系统：

| 任务类型               | 调用                   |
| ---------------------- | ---------------------- |
| search_paper / compare | Paper-RAG API          |
| speech_model / dataset | Multi-LoRA Speech LoRA |
| coding                 | Multi-LoRA Coding LoRA |
| scientific / paper QA  | Multi-LoRA Paper LoRA  |

------

### 4. 历史记忆整合

```
history.append({"user": text_input, "bot": response_text})

prompt = f"""
历史对话：
{history}

用户最新问题：
{text_input}
"""
llm_response = llm(prompt)
```

- 可选升级：embedding + vector db → Top-K 相关历史

------

### 5. 生成回答 & TTS

```
# 假设 response_text 已生成
from TTS.api import TTS
tts = TTS(model_name="VITS_model")
tts.tts_to_file(text=response_text, file_path="output.wav")
```

- 输出音频给用户
- 保留文本回答用于前端显示

------

## 五、前端交互（Demo）

- 使用 Gradio 展示：
  - 语音输入
  - 文本显示
  - 语音播放
  - 历史记录列表

示意：

```
🎤 用户说话
↓
ASR → 文本
↓
LLM → 任务解析
↓
调用系统 API
↓
生成文本回答
↓
TTS → 播放语音
↓
显示在 UI
```

------

## 六、文件结构示例

```
Speech_Assistant/
│
├── main.py              # 系统入口
├── asr/
│   └── whisper_asr.py   # 语音识别模块
├── llm/
│   └── task_parser.py   # LLM任务解析
├── router/
│   └── task_router.py   # 调度调用前两个项目
├── memory/
│   └── history_manager.py # 历史管理
├── tts/
│   └── vits_tts.py      # 文本转语音
├── demo/
│   └── gradio_demo.py   # 前端交互
└── requirements.txt
```

------

## 七、技术栈

| 功能                | 技术                         |
| ------------------- | ---------------------------- |
| 语音识别            | Whisper                      |
| 语言理解 / 任务拆解 | Qwen / TinyLlama             |
| 调用系统接口        | Python requests / FastAPI    |
| 历史管理            | Python List / FAISS / Chroma |
| 语音合成            | VITS / 实习模型              |
| 前端 Demo           | Gradio / Streamlit           |

------

## 八、创新点

1. **多模态交互**：语音输入 + LLM理解 + TTS输出
2. **任务拆解**：自然语言 → JSON → 系统API调用
3. **历史记忆**：上下文管理 + RAG式回溯
4. **无训练即可智能理解**：直接利用 Whisper + LLM
5. **可复用前两个项目**：Paper-RAG / Multi-LoRA
6. **模块化设计**：易扩展新功能 / 新系统接口

------

## 九、可扩展升级方向

1. **增加对话长记忆**：向量化历史，检索Top-K
2. **加入多轮任务链**：一个语音命令拆解成多步操作
3. **增加多模态输出**：文本 + 图片 + 图表
4. **自动LoRA增强**：根据任务生成小型LoRA专家

------

## 十、总结

本项目**不是训练模型**，而是**搭建多模态、任务智能化系统**：

- 利用现有能力（Whisper + LLM）完成理解
- 动态拆解任务，调用前两个项目API
- 输出语音 + 文本，保留历史
- 模块化 + 可扩展，工业可用

✅ 本质：**AI系统工程**，不是训练工程



需要用到的模型：1️⃣ 语音识别（ASR）

任务：把用户语音转成文本（文字输入给 LLM）

**推荐模型：**

| 模型            | 特点                   | 用途                         |
| --------------- | ---------------------- | ---------------------------- |
| `paraformer-v2` | 高精度中文 ASR，低延迟 | 日常语音输入转文字           |
| `fun-asr-mtl`   | 多任务训练，支持多语言 | 如果考虑混合英文和中文，可选 |

✅ **建议**：首选 `paraformer-v2`，因为精度和延迟平衡好，适合实时对话。

调用方式：

```
import requests, os

API_KEY = os.getenv("DASHSCOPE_API_KEY")
model_name = "paraformer-v2"
audio_file = "user_audio.wav"

# 调用阿里云 API (示意)
response = requests.post(
    f"https://api.aliyun.com/models/{model_name}/predict",
    headers={"Authorization": f"Bearer {API_KEY}"},
    files={"file": open(audio_file, "rb")}
)
text_input = response.json()["result"]
```

------

## 2️⃣ 文本理解 / LLM

任务：解析用户问题 → 拆解任务 → 调度调用系统

**推荐模型：**

| 模型        | 特点                   | 用途                           |
| ----------- | ---------------------- | ------------------------------ |
| `qwen-plus` | 大模型，多轮对话强     | 拆解复杂任务，生成调用 JSON    |
| `qwen3-max` | 旗舰模型，推理能力更强 | 可选用于复杂长上下文或论文问答 |

✅ **建议**：

- 常规任务使用 `qwen-plus`
- 对于长文本、多轮问答或调用 Paper-RAG 时，可升级到 `qwen3-max`

------

## 3️⃣ 文本到语音（TTS）

任务：把 LLM 输出回答 → 合成语音播放给用户

**推荐模型：**

| 模型                                                  | 特点               | 用途             |
| ----------------------------------------------------- | ------------------ | ---------------- |
| `qwen3-tts-vd-2026-01-26`                             | 高质量 TTS，多音色 | 用户听到自然语音 |
| 可选增强：自定义音色或方言，可在平台选择其他 TTS 模型 |                    |                  |

✅ **建议**：直接使用 `qwen3-tts-vd-2026-01-26`，无需训练

调用示意（Python）：

```
import requests, os

API_KEY = os.getenv("DASHSCOPE_API_KEY")
model_name = "qwen3-tts-vd-2026-01-26"
text = "你好，这是你的回答"

response = requests.post(
    f"https://api.aliyun.com/models/{model_name}/predict",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={"text": text}
)
with open("output.wav", "wb") as f:
    f.write(response.content)
```

------

## 4️⃣ 向量检索（Memory / Router / RAG）

任务：历史记忆检索 + LLM Router（选择 Multi-LoRA / Paper-RAG）

**推荐模型：**

| 模型                      | 特点             | 用途                                            |
| ------------------------- | ---------------- | ----------------------------------------------- |
| `bge-small` / `bge-large` | 高质量 embedding | 用于历史对话向量化、Router 相似度计算、RAG 检索 |
| `HuggingFaceEmbeddings`   | 本地部署可选     | 可替代向量 API，低延迟                          |

✅ **建议**：

- Router 路由可使用 `bge-small`
- 历史记忆向量化可使用 `bge-small` 或 `bge-large`
- 如果历史量大，可考虑向量数据库 + `bge-large`

------

## 5️⃣ Multi-LoRA / Paper-RAG 系统调用

- 这些不需要新的模型，直接调用你前两个项目的 API
- 对接方式：HTTP 或 Python requests
- 结合 Router 任务拆解即可

------

## 🔹 总结：推荐调用组合

| 模块            | 模型                      | 建议                            |
| --------------- | ------------------------- | ------------------------------- |
| ASR             | `paraformer-v2`           | 用户语音转文字                  |
| LLM             | `qwen3-max`               | 解析任务、生成回答              |
| TTS             | `qwen3-tts-vd-2026-01-26` | 文本回答 → 语音                 |
| Memory / Router | `bge-small`               | 历史记忆检索、LoRA路由          |
| 调用 API        | -                         | 直接调用 Paper-RAG / Multi-LoRA |