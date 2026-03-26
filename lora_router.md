## 项目目标

构建一个 **多领域专家大模型系统**，通过 **Router 自动选择 LoRA 专家**，提升大模型在不同领域任务中的专业能力。

本项目提出一种 **Multi-LoRA 专家系统**：

```
Base LLM
+
多个 LoRA 专家
+
Router 自动选择专家
```

实现：

```
User Query
↓
Router
↓
选择最合适的 LoRA
↓
Base LLM + LoRA
↓
Answer
```

本质类似：

**Mixture-of-Experts (MoE)**

但专家不是完整模型，而是：

**LoRA Adapter**

优势：

- 参数量小
- 易扩展
- 推理成本低

# 二、系统整体架构

系统由四个核心模块组成：

1️⃣ Router 模块
 2️⃣ LoRA 训练模块
 3️⃣ 推理模块
 4️⃣ Demo 展示模块

整体架构：

```
                    ┌─────────────┐
User Query ───────► │   Router     │
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
    Coding LoRA        Paper LoRA        Speech LoRA
        │                  │                  │
        └──────────────► Base LLM ◄───────────┘
                           │
                        Response
```

系统流程：

```
用户问题
↓
Router 判断领域
↓
选择最合适的 LoRA 专家
↓
加载 LoRA
↓
LLM 推理
↓
返回回答
```

------

# 三、Base Model

仓库默认基础模型：

**TinyLlama-1.1B-Chat**

选择原因：

1. 参数规模适中
2. 推理成本低
3. LoRA 微调稳定
4. 中文能力优秀
5. 对消费级显卡友好

默认路径：

```
models/TinyLlama-1.1B-Chat-v1.0
```

可在命令行参数或环境变量中替换为其他模型（如 Qwen2.5-1.5B）。

Base Model 只负责：

- 通用语言能力
- 推理能力
- 语言生成能力

领域知识由 **LoRA 专家提供**。

------

# 快速开始

安装依赖：

```
pip install torch transformers peft sentence-transformers bitsandbytes
```

命令行自动路由：

```
python inference/inference_pipeline.py --query "解释 VITS 的整体结构"
```

命令行手动指定领域：

```
python inference/inference_pipeline.py --query "实现快速排序" --domain coding
```

启动后端 API（供前端调用）：

```
python api/server.py
```

HTTP 调用示例：

```
POST /generate
{
  "query": "Explain VITS",
  "domain": "auto",
  "max_new_tokens": 256
}
```

响应包含选中的领域、分数与答案。

# 四、Router 模块

## Router 的作用

Router 负责：

**判断用户问题属于哪个领域**

从而选择最合适的 LoRA。

------

## Router 工作流程

```
User Query
↓
Embedding
↓
与领域向量计算相似度
↓
选择最相关领域
↓
加载对应 LoRA
```

示例：

用户问题：

```
Explain VITS
```

Router 计算相似度：

```
coding   0.31
paper    0.52
speech   0.89
```

最终选择：

```
Speech LoRA
```

------

## Router实现

Embedding模型：

推荐：

**bge-small**

Router算法：

```
1 定义领域描述
2 计算 embedding
3 cosine similarity
4 argmax
```

伪代码：

```
query_embedding = embed(query)

for domain in domains:
    score = cosine(query_embedding, domain_embedding)

best_domain = argmax(score)
```

可通过环境变量切换路由用的嵌入模型（中文推荐）：

```
ROUTER_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
```

------

# 五、LoRA 专家设计

系统设计 **三个领域专家**：

| 专家        | 能力                     |
| ----------- | ------------------------ |
| Coding LoRA | 代码生成、代码解释、算法 |
| Paper LoRA  | 论文理解、论文总结       |
| Speech LoRA | 语音技术知识             |

这种设计具有良好的扩展性：

未来可以增加：

- CV LoRA
- NLP LoRA
- Finance LoRA
- Medical LoRA

------

# 六、数据集设计

本项目构建了 **三个领域的数据集**。

总规模：

```
≈ 4100 instruction samples
```

适合 LoRA 微调。

# 1 Coding Dataset

来源：

| 数据集                                       | 数量 |
| -------------------------------------------- | ---- |
| iamketan25/python-qa-instructions-dataset    | 531  |
| iamtarun/python_code_instructions_18k_alpaca | 300  |
| cassanof/leetcode-solutions                  | 200  |

合并后：

```
1028 samples
```

任务类型：

- Python 问答
- 代码生成
- 算法题解释
- 代码解析

数据处理：

```
去重
统一 instruction 格式
```

最终文件：

```
merged_coding_all.json
```

------

# 2 Paper Dataset

来源：

| 数据集           | 数量 |
| ---------------- | ---- |
| Chinese_Paper_QA | 600  |
| SciRIFF          | 600  |
| SciDQA           | 600  |

合计：

```
1800 samples
```

任务：

- 论文解释
- 方法总结
- 技术对比
- 论文问答

数据文件：

```
merged_paper_lora_all.json
```

------

# 3 Speech Dataset

Speech 数据采用：

**Self-Instruct 数据生成**

方法：

1. 收集语音技术知识点
2. 让 LLM 自动生成 QA

知识类别：

| 类别              | 数量 |
| ----------------- | ---- |
| speech datasets   | 50   |
| speech models     | 250  |
| speech concepts   | 350  |
| speech pipeline   | 200  |
| speech tasks      | 150  |
| speech tools      | 100  |
| speech evaluation | 100  |

总计：

```
1300 QA
```

知识覆盖：

- ASR
- TTS
- Whisper
- Tacotron
- FastSpeech
- VITS
- CTC
- Speech datasets

最终数据：

```
speech_instruction_dataset.json
```

------

# 七、LoRA 训练模块

LoRA 训练流程：

```
dataset
↓
instruction format
↓
tokenization
↓
LoRA fine-tuning
↓
adapter
```

每个领域训练一个 LoRA：

```
coding_lora
paper_lora
speech_lora
```

训练方法：

PEFT LoRA

关键参数：

```
r = 8
alpha = 16
dropout = 0.05
```

训练输出：

```
adapters/
    coding_lora
    paper_lora
    speech_lora
```

每个 LoRA 仅几 MB。

------

# 八、推理系统

推理流程：

```
User Query
↓
Router
↓
选择 LoRA
↓
加载 LoRA
↓
LLM 生成回答
```

推理逻辑：

```
domain = router(query)

adapter = load_lora(domain)

model = base_model + adapter

answer = model.generate(query)
```

技术：

PEFT：

```
model.load_adapter()
```

------

# 九、系统升级设计（重要）

为了提升系统能力，可以做四个升级。

------

# 升级1：Top-K Router

当前：

```
选择1个 LoRA
```

升级：

```
选择 Top-K LoRA
```

例如：

```
coding 0.63
paper 0.58
speech 0.21
```

加载：

```
coding + paper
```

进行 **LoRA 融合推理**。

------

# 升级2：Router 训练

当前 Router：

```
embedding similarity
```

升级：

训练 **Router classifier**

```
query → domain
```

模型：

```
bge-small + linear layer
```

效果更稳定。

------

# 升级3：Dynamic LoRA Loading

推理时：

```
按需加载 LoRA
```

避免：

```
一次加载所有 LoRA
```

优点：

- 节省显存
- 扩展更多专家

------

# 升级4：前端页面对接 API

对接已有前端页面，通过标准 HTTP 接口实现自动/手动路由与生成，并支持前端多轮对话与代码高亮显示。

```
POST /generate
{
  "query": "...",
  "domain": "auto | coding | paper | speech",
  "max_new_tokens": 256,
  "route_available_only": true,
  "code_language": "Python | C | C++ | Java | Go | JavaScript | Rust",
  "fast_mode": false
}
```

返回：

```
{
  "selected_domain": "...",
  "scores": {"coding": ..., "paper": ..., "speech": ...},
  "used_adapter": true,
  "available_domains": ["coding"],
  "base_model_path": "models/TinyLlama-1.1B-Chat-v1.0",
  "answer": "..."
}
```

前端功能清单：

- 对话历史本地保存（刷新不丢）
- 多会话管理与新建对话
- 代码块高亮显示
- 语言选择与快速模式开关

对话持久化接口：

```
GET  /api/conversations
POST /api/conversations
GET  /api/conversations/{id}
POST /api/conversations/{id}/messages
PUT  /api/conversations/{id}
```

数据库路径可配置：

```
CHAT_DB_PATH=./data/chat.db
```

------

# 十、项目技术栈

主要技术：

| 技术                  | 用途       |
| --------------------- | ---------- |
| Transformers          | 加载LLM    |
| PEFT                  | LoRA训练   |
| Sentence Transformers | Embedding  |
| PyTorch               | 模型训练   |
| bitsandbytes          | 量化推理   |
| 标准库 HTTPServer     | 后端接口   |

------

# 十一、项目创新点

本项目创新点主要体现在：

### 1 Multi-LoRA 专家系统

使用 **多个 LoRA 作为专家**。

相比：

```
单 LoRA
```

更加灵活。

------

### 2 Router 自动选择专家

系统根据：

```
query semantic
```

自动选择专家。

类似：

```
Mixture-of-Experts
```

------

### 3 领域数据构建

项目构建了：

- Coding dataset
- Paper dataset
- Speech dataset

实现 **多领域能力增强**。

------

### 4 低成本扩展

新增领域只需：

```
训练新的 LoRA
```

无需重新训练 Base Model。

------

# 十二、项目规模

最终系统：

```
Base Model: TinyLlama-1.1B-Chat（可配置）
LoRA Experts: 3
Datasets: 4100+
Domains: Coding / Paper / Speech
```

系统特点：

- 轻量
- 可扩展
- 模块化



项目结构：
Adaptive-Multi-LoRA-Routing-for-LLM
│
├── api
│   └── server.py
│
├── data
│   ├── coding
│   │   ├── train.json
│   │   ├── val.json
│   │   └── test.json
│   │
│   ├── paper
│   │   ├── train.json
│   │   ├── val.json
│   │   └── test.json
│   │
│   └── speech
│       ├── train.json
│       ├── val.json
│       └── test.json
│
├── models
│   ├── TinyLlama-1.1B-Chat-v1.0
│   └── lora
│       ├── coding_lora
│       ├── paper_lora
│       └── speech_lora
│
├── router
│   ├── router_embedding.py
│   ├── router_classifier.py
│   ├── domain_config.py
│   └── similarity.py
│
├── training
│   ├── train_lora.py
│   ├── train_coding_lora.py
│   └── dataset_loader.py
│
├── inference
│   ├── inference_pipeline.py
│   ├── compare_loras.py
│   └── test_coding_lora.py
│
└── 页面代码.md



# router模块

Router 是你项目最核心模块。

```
router
│
├── router_embedding.py
├── router_classifier.py
├── domain_config.py
└── similarity.py
```

### 1 router_embedding.py

最简单 Router：

```
query
↓
embedding
↓
cosine similarity
↓
选择 domain
```

------

### 2 router_classifier.py

升级 Router：

```
query → domain
```

小模型分类。

------

### 3 domain_config.py

定义领域描述：

```
DOMAINS = {
    "coding": "questions about programming, python, algorithms, code generation",
    "paper": "research papers, scientific methods, machine learning papers",
    "speech": "speech recognition, tts, asr, speech models like vits whisper"
}
```

------

### 4 similarity.py

实现：

```
cosine similarity
```

------

# training模块

训练 LoRA。

```
training
│
├── train_lora.py
├── train_coding_lora.py
├── dataset_loader.py
```

常用命令：

```
python training/train_lora.py --domain coding --data data/coding/train.json --output_dir models/lora/coding_lora
python training/train_lora.py --domain paper  --data data/paper/train.json  --output_dir models/lora/paper_lora
python training/train_lora.py --domain speech --data data/speech/train.json --output_dir models/lora/speech_lora
```

------

## dataset_loader.py

读取你的 json。

你的格式：

```
instruction
input
output
```

dataset_loader.py 会把它转换成：

```
prompt = instruction + input
target = output
```

------

## train_coding_lora.py

训练 coding LoRA：

```
data/coding/train.json
```

输出：

```
models/lora/coding_lora
```

------

## train_paper_lora.py

训练 paper LoRA。

------

## train_speech_lora.py

训练 speech LoRA。

------

# inference模块

推理系统。

```
inference
│
├── inference_pipeline.py
├── compare_loras.py
└── test_coding_lora.py
```

------

## inference_pipeline.py

完整流程：

```
query
↓
router
↓
select domain
↓
load lora
↓
generate answer
```

------

# API 模块

前端通过标准 HTTP 接口与后端交互：

- 启动：`python api/server.py`
- 健康检查：`GET /health`
- 生成接口：`POST /generate`

请求体：

```
{
  "query": "必填",
  "domain": "auto | coding | paper | speech，可选",
  "base_model_path": "可选，默认使用仓库内模型",
  "max_new_tokens": 256
}
```

响应体：

```
{
  "selected_domain": "实际使用领域",
  "scores": { ... 自动路由时的打分 ... },
  "used_adapter": true/false,
  "answer": "最终回答"
}
```

# 环境变量与配置

常用环境变量：

```
BASE_MODEL_PATH        # 基础模型路径，默认 models/TinyLlama-1.1B-Chat-v1.0
LOAD_IN_4BIT           # 是否使用 4bit 量化，默认 1
ROUTER_EMBEDDING_MODEL # 路由用 Embedding 模型名称，默认 BAAI/bge-small-en-v1.5
API_HOST               # API 绑定地址，默认 0.0.0.0
API_PORT               # API 端口，默认 8000
```

# 完整系统运行流程

完整流程：

```
# 训练
python training/train_lora.py --domain coding --data data/coding/train.json --output_dir models/lora/coding_lora
python training/train_lora.py --domain paper  --data data/paper/train.json  --output_dir models/lora/paper_lora
python training/train_lora.py --domain speech --data data/speech/train.json --output_dir models/lora/speech_lora

# 命令行推理（自动路由）
python inference/inference_pipeline.py --query "Explain VITS"

# 命令行推理（手动领域）
python inference/inference_pipeline.py --query "Write quicksort" --domain coding

# 启动后端（前端调用）
python api/server.py
```

------

# 这个结构的优点

这个目录结构：

优点非常明显：

1️⃣ **数据清晰**

```
data/domain/train.json
```

2️⃣ **模块分离**

```
router
training
inference
```

3️⃣ **扩展非常简单**

如果新增领域：

例如：

```
cv
```

只需要增加：

```
data/cv
training/train_cv_lora.py
models/lora/cv_lora
```

Router加一行：

```
cv
```

系统就升级了。
