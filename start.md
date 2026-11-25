# 🚀 启动指南（start.md）

> 适用仓库：`Qwen3-VL-MoeLORA`。本文按「环境 → 依赖 → 模型 → 数据 → 训练 → 推理 → 桌面端 → 验证 → 排障」顺序给出**可直接复制执行**的步骤，并标注每个环节的**成功标志**。
> 所有命令以 **Windows PowerShell** 为例；Linux/macOS 把 `.\\.venv\\Scripts\\Activate.ps1` 换成 `source .venv/bin/activate` 即可。

---

## 📌 先看这张总表（决定你要走哪条路径）

| 目标 | 是否需要 GPU | 需要训练吗 | 预计耗时 | 章节 |
| --- | --- | --- | --- | --- |
| 只想看桌面助手界面（复用已有 LoRA 权重） | 是（≥8GB） | 否 | 10–20 分钟（含下载） | 1 → 2 → 3 → 4 → 6 |
| 只验证基座能推理（不加载 LoRA） | 是（≥8GB） | 否 | 10–20 分钟 | 1 → 2 → 3 → 7 |
| 跑通完整微调 + 推理 + UI | 是（≥8GB） | 是 | 2–4 小时（训练约 1.5 小时/方案） | 全部章节 |

> ⚠️ **关键前提**：`multi_agent/model_client.py`、`main_app_ui.py`、`test.py` 都**硬编码加载** `output/Qwen3-VL-4Blora/checkpoint-310`。仓库不自带该权重，**没训练过就直接启动 UI 会报路径不存在**，需按「第 7 节」注释掉 LoRA 加载。

---

## 0️⃣ 启动前检查：这三样必须先有

```powershell
# 0.1 确认 Python 版本（建议 3.10 或 3.11）
python --version

# 0.2 确认 GPU 与驱动（能看到 CUDA 版本号即可）
nvidia-smi

# 0.3 确认当前目录是仓库根目录（应能看到 MoeLORA.py）
Get-ChildItem -Name
```

**成功标志**：Python ≥ 3.10；`nvidia-smi` 正常输出且显存 ≥ 8GB；目录列表包含 `MoeLORA.py`、`multi_agent`。

> 该仓库**没有 `requirements.lock`、没有安装脚本、没有 HTTP 服务**，也**不监听任何端口**，启动 = 运行本地 Python 脚本。

---

## 1️⃣ 创建虚拟环境并安装依赖

```powershell
# 1.1 建环境
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

```powershell
# 1.2 先装 GPU 版 torch（避免 pip 默认装成 CPU 版）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 1.3 装顶层依赖
pip install -r requirements.txt

# 1.4 补装 requirements 里漏掉、但代码实际 import 的包
pip install langchain-huggingface PyQt5 rank-bm25
```

```powershell
# 1.5 装多智能体子模块依赖
pip install -r multi_agent\requirements.txt
```

**成功标志**：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望：2.x.x True（若为 False，见第 10 节 T2）
```

> 📎 为什么需要 1.4：
> - `main_langchain_ui.py` 里 `from langchain_huggingface import HuggingFaceEmbeddings`，顶层 `requirements.txt` 未声明；
> - `multi_agent/requirements.txt` 写的是 `PySide6`，但三个 UI 文件实际 `from PyQt5 import ...`；
> - `rank_bm25` 只写在子模块依赖里，缺了会退化成纯向量检索（不报错但降级）。

---

## 2️⃣ 下载基座模型（Qwen3-VL-4B-Instruct）

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"   # 国内加速，不需要可去掉
python download_model.py
```

- 脚本内容就是 `snapshot_download("Qwen/Qwen3-VL-4B-Instruct", local_dir="./qwen3-vl-4b-instruct")`，**不接受任何命令行参数**。
- 权重约 8GB，耗时取决于带宽。

**成功标志**：

```powershell
Get-ChildItem .\qwen3-vl-4b-instruct | Select-Object -ExpandProperty Name
# 应看到 config.json、model*.safetensors、tokenizer.json、preprocessor_config.json 等
```

> 目录里原本只有 `readme.md`，下载后会被真实权重填充；若仍只有 readme，说明下载失败（见第 10 节 T4）。

---

## 3️⃣ 准备 COCO 数据（有一个"目录存在即跳过"的坑）

```powershell
# 3.1 【必做】先移除占位目录，否则脚本检测到目录存在会直接跳过，不生成 CSV
Remove-Item -Recurse -Force .\coco_2014_caption

# 3.2 下载 500 条图文对，落盘 jpg + CSV
python download_data2csv.py

# 3.3 CSV → Qwen3-VL 对话 JSON
python csv2json.py
```

**成功标志**：

```powershell
Test-Path .\coco-2024-dataset.csv        # True
Test-Path .\data_vl.json                 # True
(Get-Content .\data_vl.json -Raw | ConvertFrom-Json).Count   # 500（或 ≤500）
```

> 📎 注意 `download_data2csv.py` 的 CSV 写在**根目录** `./coco-2024-dataset.csv`，不在 `coco_2014_caption/` 下；脚本也没有 `--output/--top_k` 参数（`README.md` 里的写法与实际代码不符）。
> `csv2json.py` 读取固定路径 `./coco-2024-dataset.csv`，输出固定 `./data_vl.json`，同样无参数。

---

## 4️⃣ 微调训练（四选一）

四个脚本都是**顶层直接执行**（无 `if __name__ == "__main__"`、无 argparse），配置硬编码在文件里：

```powershell
# 任选其一执行
python MoeLORA.py          # MoE-LoRA  → output/Qwen3-VL-4Bmoelora
python MotLoRA.py          # MoT-LoRA  → output/Qwen3-VL-4Bmotlora
python AdapterTuning.py     # IA3       → output/Qwen3-VL-4Badapter
python lora.py             # LoRA      → output/Qwen3-VL-4Blora（见下方修复）
```

### 4.1 ⚠️ 运行 `lora.py` 前必须补一个逗号

当前仓库 [lora.py](file:///e:/Code/AI/Start/Web/interview/微调/Qwen3-VL-MoeLORA-main/lora.py#L172-L176) 中：

```python
target_modules=[
    "q_proj", "k_proj", "v_proj", "o_proj"
    "visual_q_proj", "visual_k_proj"],
```

`"o_proj"` 后缺逗号 → Python 字符串隐式拼接成 `"o_projvisual_q_proj"`，会报找不到目标模块。改为：

```python
target_modules=[
    "q_proj", "k_proj", "v_proj", "o_proj",
    "visual_q_proj", "visual_k_proj"],
```

（`MoeLORA.py`、`MotLoRA.py`、`AdapterTuning.py` 中该处逗号是完整的，无此问题。）

### 4.2 训练前建议确认的 3 个硬编码项

| 文件内变量 | 当前值 | 说明 |
| --- | --- | --- |
| `model_id` | `"./qwen3-vl-4b-instruct"` | 基座本地路径 |
| `tokenizer / processor` | `"qwen3-vl-4b-instruct"` | 相对名，若报 HF 仓库校验错误，改成 `"./qwen3-vl-4b-instruct"` |
| `train_json_path` | `"data_vl.json"` | 由第 3 步生成 |

脚本会自动切分：`train_data = data[:-4]`、`test_data = data[-4:]`，写出 `data_vl_train.json` / `data_vl_test.json`。

### 4.3 SwanLab（可选）

```powershell
pip install swanlab          # requirements 已含
swanlab login                # 或设置 $env:SWANLAB_API_KEY="你的Key"
```

无账号 / 无网络时，`SwanLabCallback` 可能初始化失败，可临时注释训练脚本里的 `callbacks=[swanlab_callback]`。

**成功标志**：

- 终端出现 `trainable params: 5,898,240 || all params: 4,443,714,048 || trainable%: 0.1327`（MoE-LoRA 为 `10,298,240` / `0.245%`）；
- 进度条约 310 steps、5 epoch 走完（单卡约 1.5 小时/方案）；
- 目录出现：

```powershell
Get-ChildItem .\output
# Qwen3-VL-4Blora / Qwen3-VL-4Bmoelora / Qwen3-VL-4Bmotlora / Qwen3-VL-4Badapter（取决于你跑的是哪个）
Get-ChildItem .\output\Qwen3-VL-4Blora | Select-Object -ExpandProperty Name
# 应看到 checkpoint-310、adapter_config.json、adapter_model.safetensors
```

---

## 5️⃣ 选择你的推理入口（两个脚本都会加载 LoRA）

| 脚本 | 用途 | 硬编码依赖 |
| --- | --- | --- |
| `test.py` | 终端一次性图文推理 | 基座 + `output/Qwen3-VL-4Blora/checkpoint-310` |
| `main_app_ui.py` | 桌面助手（纯 FAISS RAG） | 同上 + `rag_data/` |
| `main_langchain_ui.py` | 桌面助手（LangChain 链式 RAG） | 同上 + `rag_data/` |
| `multi_agent/main_app.py` | 桌面助手（5-Agent + 混合检索） | `../` 基座与 LoRA + `knowledge_base/` |

> 若你训练的是 MoE-LoRA / IA3，请把上述脚本中的 `LORA_CHECKPOINT` 改成对应目录（如 `output/Qwen3-VL-4Bmoelora`）。**PeftModel 加载时 adapter 配置必须与训练时一致**，不要混用。

---

## 6️⃣ 启动桌面助手（三选一，注意工作目录）

### 方案 A：纯 FAISS 版（根目录运行）

```powershell
# 6.1 先建知识库目录（缺失时脚本不会崩，但检索永远为空）
New-Item -ItemType Directory -Force .\rag_data | Out-Null
Copy-Item .\multi_agent\knowledge_base\sample.md .\rag_data\ -ErrorAction SilentlyContinue

# 6.2 启动（必须在仓库根目录执行）
python main_app_ui.py
```

### 方案 B：LangChain 链式 RAG 版（根目录运行）

```powershell
python main_langchain_ui.py
```

> 该脚本首次会自动 `mkdir rag_data`，但依赖 `langchain-huggingface`（见 1.4）。

### 方案 C：5-Agent 版（**必须切到 `multi_agent` 目录**）

```powershell
cd multi_agent
python main_app.py
```

> 因为 `model_client.py` 默认 `../qwen3-vl-4b-instruct` 与 `../output/Qwen3-VL-4Blora/checkpoint-310`，在根目录运行会找不到路径。

**首次启动会额外下载两个模型（需联网）**：

| 模型 | 用途 | 体积 |
| --- | --- | --- |
| `sentence-transformers/all-MiniLM-L6-v2` | 向量化 | 约 90MB |
| `BAAI/bge-reranker-base` | Cross-Encoder 精排 | 约 1GB |

**成功标志**：窗口出现「Multi‑Agent 小苔藓 / xiao taixian 生活助手」标题；输入问题后右侧「调度 + 任务计划」「RAG 检索参考」面板有内容；发送按钮在推理期间置灰，完成后恢复。

---

## 7️⃣ 没有训练权重怎么办（只想验证基座推理）

`test.py` 与 UI 都调用 `PeftModel.from_pretrained(...)`，未训练时会报目录不存在。两种做法：

**做法 1（推荐）：先跑第 4 步训练一个方案。**
**做法 2：临时注释 LoRA 加载。**

以 [test.py](file:///e:/Code/AI/Start/Web/interview/微调/Qwen3-VL-MoeLORA-main/test.py#L23-L26) 为例，把

```python
lora_checkpoint = "output/Qwen3-VL-4Blora/checkpoint-310"
model = PeftModel.from_pretrained(model, lora_checkpoint)
model = model.to(model.device)
```

改为

```python
# lora_checkpoint = "output/Qwen3-VL-4Blora/checkpoint-310"
# model = PeftModel.from_pretrained(model, lora_checkpoint)
model = model.eval()
```

`main_app_ui.py` 同样处理 `load_model()` 中的 `PeftModel` 两行；`multi_agent/model_client.py` 处理 `_build_model()` 中的 `PeftModel.from_pretrained`。

**成功标志**：`python test.py` 打印「模型回答：」，内容为对示例图片的中文描述。

> `test.py` 默认使用官方在线示例图（需联网）。要离线测试，把 `image` 字段改成 `"image/demo.jpeg"`（仓库已自带）。

---

## 8️⃣ 启动验证清单（逐项打勾）

| # | 检查项 | 命令 / 观察点 | 通过标准 |
| --- | --- | --- | --- |
| 1 | 环境 | `python -c "import torch;print(torch.cuda.is_available())"` | `True` |
| 2 | 基座 | `Get-ChildItem .\qwen3-vl-4b-instruct` | 有 `*.safetensors` |
| 3 | 数据 | `Test-Path .\data_vl.json` | `True`，条目 500 |
| 4 | 训练产物 | `Get-ChildItem .\output` | 有 `checkpoint-310` |
| 5 | 推理 | `python test.py` | 打印中文回答 |
| 6 | 桌面端 | 启动任一 UI | 窗口正常、可发送 |
| 7 | RAG | 问一个知识库内的问题 | 右侧出现 `[Doc n / source]` |
| 8 | 归档 | 输入「总结对话」 | `multi_agent/output/reports/*.md` 新增文件 |

---

## 9️⃣ 完整启动顺序（一条流水线）

```text
0 检查环境 → 1 建 venv + 装依赖 → 2 下载基座
        → 3 删占位目录 + 下载数据 + 转 JSON
        → 4 训练（四选一，lora.py 先补逗号）
        → 5/6 推理 or 桌面端 → 8 验证清单
```

---

## 🔟 故障排查速查表

| ID | 现象 | 根因 | 处理 |
| --- | --- | --- | --- |
| T1 | `torch.cuda.is_available()` 为 `False` | 装成了 CPU 版 torch | 重装：`pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| T2 | 启动即 `CUDA out of memory` | 显存不足（其他进程占用或未启用 4-bit） | 关闭占显存进程；确认 `USE_4BIT=True` / `load_in_4bit=True`；把 `max_new_tokens` 降到 128 |
| T3 | `ValueError: ... not found in target_modules` / 找不到 `o_projvisual_q_proj` | `lora.py` 第 174 行缺逗号 | 按 4.1 补逗号后重跑 |
| T4 | 下载完目录仍只有 `readme.md` | 网络/HF 端点失败 | 设 `HF_ENDPOINT=https://hf-mirror.com`；或手动从 HF 页面下载后放入 `qwen3-vl-4b-instruct/` |
| T5 | `download_data2csv.py` 打印「目录已存在,跳过数据处理步骤」 | `coco_2014_caption/` 已存在 | 删除该目录后重跑（见第 3 步） |
| T6 | `Can't load ... repo id` / 本地路径被当仓库名 | 以 `model_id=` 关键字传本地路径 | 本地路径改作**第一个位置参数**；tokenizer/processor 统一 `"./qwen3-vl-4b-instruct"` |
| T7 | `FileNotFoundError: output/Qwen3-VL-4Blora/checkpoint-310` | 未训练或训练的是别的方案 | 跑第 4 步，或改 `LORA_CHECKPOINT` / 按第 7 节注释 PeftModel |
| T8 | `ModuleNotFoundError: PyQt5` | 子模块 requirements 声明的是 PySide6 | `pip install PyQt5` |
| T9 | `ModuleNotFoundError: langchain_huggingface` | 顶层 requirements 未声明 | `pip install langchain-huggingface` |
| T10 | 训练 loss 不动 / 报 `use_cache` 警告 | 梯度检查点与 cache 冲突 | 确认三行：`use_cache=False`、`gradient_checkpointing_enable()`、`enable_input_require_grads()` |
| T11 | 右侧 RAG 一直「未检索到相关文档」 | 知识库为空 / 走了 `rag_data` 而文档放在 `knowledge_base` | 按对应目录放 `.txt/.md/.pdf`；确认勾选「启用知识库检索」 |
| T12 | 启动报 `../qwen3-vl-4b-instruct` 不存在 | 在根目录运行了 `multi_agent/main_app.py` | 先 `cd multi_agent` 再运行 |
| T13 | 精排模型下载卡住 | 首次拉取 `bge-reranker-base` 约 1GB | 设置镜像；或临时 `pip uninstall sentence-transformers` 走降级（会失去精排） |
| T14 | 中文 txt 读入报 `UnicodeDecodeError` | 编码非 utf-8 | 无需改代码，脚本会依次尝试 utf-16/gbk/gb2312/latin-1 并兜底 |
| T15 | SwanLab 回调报鉴权错误 | 未登录 | `swanlab login`，或注释 `callbacks=[swanlab_callback]` |

---

## 📎 一页速查（复制即用）

```powershell
# ① 环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install langchain-huggingface PyQt5 rank-bm25
pip install -r multi_agent\requirements.txt

# ② 模型
$env:HF_ENDPOINT = "https://hf-mirror.com"
python download_model.py

# ③ 数据
Remove-Item -Recurse -Force .\coco_2014_caption
python download_data2csv.py
python csv2json.py

# ④ 训练（lora.py 需先补 target_modules 逗号）
python MoeLORA.py

# ⑤ 推理
python test.py

# ⑥ 桌面端
python main_app_ui.py
# 或： python main_langchain_ui.py
# 或： cd multi_agent; python main_app.py
```

> 全部步骤为本地脚本执行，**不占用端口、不启动服务、不需要数据库**；终止方式为 `Ctrl + C`（终端）或直接关闭窗口。
