# NMT Trainer

中英翻译模型训练与推理工具，基于 PyTorch Transformer 架构。

## 快速开始

```bash
pip install -r requirements.txt
python app.py
```

浏览器访问 `http://localhost:5000`

## 项目结构

```
nmt_trainer/
├── app.py              # Flask Web 服务 + API
├── model.py            # TransformerNMT 模型定义
├── trainer.py          # 训练循环 + 评估 + 检查点管理
├── dataset.py          # 数据加载、词汇表构建
├── data/               # 数据集存放目录（数据集在huggingface.co/datasets/Helsinki-NLP/opus-100下载）
│   ├── train.json      # 训练集 (JSON，每行 {"en": "...", "zh": "..."})
│   ├── validation.json # 验证集
│   └── test.json       # 测试集（可选）
├── checkpoints/        # 模型检查点（.pt 文件）
├── static/             # 前端静态资源
│   ├── style.css
│   ├── train.js
│   └── translate.js
└── templates/          # HTML 模板
    ├── train.html      # 训练控制台
    └── translate.html  # 翻译页面
```

## 页面功能

### 训练页面 (`/train`)
- 配置模型架构和训练超参数
- 实时图表监控 Loss / BLEU
- 加载已有检查点继续训练

### 翻译页面 (`/translate`)
- 单句翻译 / 批量智能分句翻译
- 翻译历史记录
- 加载已有模型进行推理

## 数据格式

训练/验证数据为 JSON 文件，每行一个对象：

```json
{"en": "Hello, how are you?", "zh": "你好，你好吗？"}
```

也支持 TSV 格式（`en\tzh`，首行为表头）。

## 使用 OPUS-100 数据集

`data/` 目录下已包含 OPUS-100 en-zh 子集。如需更多数据，可从 [OPUS-100](https://opus.nlpl.eu/opus-100.php) 下载。
