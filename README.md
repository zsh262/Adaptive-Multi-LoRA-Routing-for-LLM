# Adaptive Multi-LoRA Routing

基于 Router 的多领域专家 LoRA 系统，根据用户问题自动选择最合适的 LoRA 专家。

## 架构

```
用户问题 → EmbeddingRouter → 领域分类 → 加载对应LoRA → 生成回答
```

## 领域专家

| 专家 | 能力 |
|------|------|
| Coding LoRA | 代码生成、代码解释、算法 |
| Paper LoRA | 论文总结、方法分析、学术问答 |
| Speech LoRA | 语音识别、TTS、语音模型 |

## 快速启动

```bash
# 启动后端
python api/server.py

# 调用示例
python api/multi_lora_client.py
```

## API

```python
from multi_lora_client import MultiLoraClient

client = MultiLoraClient()

# 自动路由
result = client.ask("BERT的核心贡献是什么？")

# 强制指定领域
result = client.ask_coding("写个快速排序")
result = client.ask_paper("总结Attention论文")
result = client.ask_speech("VITS模型原理")
```

## 训练

```bash
# 训练指定领域 LoRA
python training/train_lora.py --domain paper --data data/paper/train.json
```

## 项目结构

```
├── api/                    # API 服务
│   ├── server.py          # 后端服务
│   └── multi_lora_client.py  # 客户端
├── router/                # 路由器
│   └── router_embedding.py  # Embedding 路由
├── training/              # 训练脚本
│   └── train_lora.py
└── models/lora/          # LoRA 模型存储
    ├── coding_lora/
    ├── paper_lora/
    └── speech_lora/
```
