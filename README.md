# 🩺 循证临床顾问

基于本地大模型 + PubMed 实时检索的 RAG 系统。每条医学结论都标注文献来源，
模型不编造检索结果之外的医学知识。

提供两种模式：

| 模式 | 入口 | 用途 |
|---|---|---|
| **临床问诊** | 页面顶部问诊表单 | 填写症状 + 病史 → 鉴别诊断 → 循证治疗方案 |
| **文献速查** | 底部「只检索文献，不做诊断」折叠区 | 定向查证某个医学问题，只复述文献 |

---

## 从零搭建（克隆本项目的人看这节）

> 仓库里**只有源码**，一共 11 个文件 59KB。模型权重、虚拟环境、向量库都被
> `.gitignore` 排除了——否则仓库会有 6.7GB，而且 GitHub 不接受超过 100MB 的单文件。
> 所以 clone 下来之后必须补齐下面这些东西，**光有代码跑不起来**。

### 前置要求

| 项目 | 要求 |
|---|---|
| 操作系统 | Windows（启动脚本是 `.bat`；其他系统手动 `streamlit run app.py` 同样可用） |
| Python | 3.11 以上（本项目在 3.14.0 验证） |
| Ollama | 任意较新版本（本项目在 0.35.0 验证） |
| 内存 | 16GB（7B Q4 模型加载后约占 5~6GB） |
| 磁盘 | 约 12GB（模型 5.5GB + 依赖 1.6GB + 余量） |

### 需要额外下载的东西

| 内容 | 大小 | 来源 |
|---|---|---|
| Ollama 本体 | ~700MB | [ollama.com/download](https://ollama.com/download) |
| Qwen2.5-7B-Instruct Q4_K_M（GGUF 单文件） | ~4.7GB | 步骤 3 |
| PubMedBERT 医学嵌入模型 | 837MB | 步骤 4 |

### 步骤

**1. 克隆并装依赖**

```bash
git clone https://github.com/<你的用户名>/evidence-based-clinical-advisor.git
cd evidence-based-clinical-advisor

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

`pip` 这一步会拉 PyTorch 等一批包（几百 MB），需要几分钟。

**2. 安装 Ollama 并让它保持运行**（托盘出现羊驼图标即为运行中）

**3. 下载模型并导入 Ollama**

HuggingFace / ModelScope 上超过约 4GB 的 GGUF 通常被切成多个分片，
**分片版无法直接导入**（原因见「踩坑 7」）。必须用**单文件**版本：
ModelScope 上 `bartowski/Qwen2.5-7B-Instruct-GGUF` 仓库里的
`Qwen2.5-7B-Instruct-Q4_K_M.gguf`。

下好后**放进 `models/` 目录，文件名保持原样**：

```
my-health-agent/
└── models/
    └── Qwen2.5-7B-Instruct-Q4_K_M.gguf
```

路径和文件名一个字都不能改——`Modelfile` 第一行是相对路径
`FROM ./models/Qwen2.5-7B-Instruct-Q4_K_M.gguf`。

然后导入：

```bash
ollama create qwen2.5:7b -f Modelfile
```

> 这一步只是把 GGUF 拷进 Ollama 的 blob 存储并登记生成参数，
> 不会产生第二份大文件。之后改提示词**不需要**重新执行。

**4. 下载医学嵌入模型**

```bash
hf download NeuML/pubmedbert-base-embeddings --local-dir ./pubmedbert-embeddings
```

必须下到 **`./pubmedbert-embeddings`** 这个目录（对应 `rag_engine.py` 里的
`EMBED_MODEL_LOCAL_DIR`）。国内直连 HuggingFace 会超时，
但 `rag_engine.py` 开头已自动设置 `HF_ENDPOINT=https://hf-mirror.com`，不用手动配。

**5. 建 Streamlit 凭据文件**

在**用户主目录**（不是项目目录）建 `.streamlit\credentials.toml`：

```toml
[general]
email = ""
```

不建的话第一次启动会卡在 `Email:` 提示，服务器根本起不来（见「踩坑 9」）。
这个文件不在仓库里，**每个使用者各自建**（里面是个人邮箱，不应入库）。

**6.（可选）换成自己的邮箱**

`pubmed_search.py` 顶部的 `CONTACT_EMAIL` 现在是占位符 `your_email@example.com`。
NCBI 的规定是调用方提供邮箱，填占位符照样能检索，但建议改成自己的——
出问题时 NCBI 会发邮件而不是直接封 IP。

**7. 启动**

双击 `启动助手.bat`。

---

## 快速启动（环境已搭好）

双击 `启动助手.bat`，浏览器会自动打开 http://localhost:8501。

或手动启动：

```bash
cd "桌面/medical AI/my-health-agent"
venv\Scripts\activate
streamlit run app.py
```

> 使用前请确保 **Ollama 正在运行**（开机自启，右下角托盘有羊驼图标）。

---

## 项目结构

| 文件 | 作用 |
|---|---|
| `pubmed_search.py` | 调用 PubMed E-utilities API 检索文献，自动过滤出系统评价 / Meta 分析 / RCT |
| `rag_engine.py` | RAG 核心：向量化 → 存入 Qdrant → 检索 → 生成回答。含 `clinical_consultation()`（临床问诊）与 `ask_health_question()`（文献速查）两条管线 |
| `app.py` | Streamlit 界面：顶部问诊表单 + 底部文献速查入口 |
| `Modelfile` | 定义本地模型的生成参数和系统提示 |
| `启动助手.bat` | 一键启动脚本（**GBK 编码 + CRLF 换行**，勿用 UTF-8 保存） |
| `.streamlit/` | Streamlit 配置，用于跳过首次运行的邮箱询问 |
| `models/` | 本地大模型文件（GGUF，4.7GB） |
| `pubmedbert-embeddings/` | PubMedBERT 医学嵌入模型（837MB） |
| `local_qdrant/` | 本地向量数据库（自动生成） |
| `venv/` | Python 虚拟环境 |

---

## 环境组成（实际安装版本）

| 组件 | 版本 / 说明 |
|---|---|
| Ollama | 0.35.0 |
| 本地大模型 | Qwen2.5-7B-Instruct (Q4_K_M 量化) |
| 嵌入模型 | NeuML/pubmedbert-base-embeddings（768 维） |
| Python | 3.14.0 |
| PyTorch | 2.14.0 (CPU 版) |

### 关于模型的说明

**BioMistral 无法使用。** 教程里写的 `ollama pull biomistral` 会返回 404 —
Ollama 官方模型库里没有这个模型。

本项目改用 **Qwen2.5-7B-Instruct**：中英双语，处理「中文提问 + 英文文献摘要」
这种混合场景效果更好，指令跟随能力也明显强于 BioMistral 所用的 Llama-2 基座。

想换模型只需修改 `rag_engine.py` 顶部的 `LLM_MODEL` 常量。

---

## 搭建过程中踩到的坑（教程已过时的地方）

这份教程有几处内容会导致直接失败，记录在此备查：

**1. `ollama pull biomistral` → 404**
官方库无此模型。已改用 Qwen2.5-7B-Instruct。

**2. `qdrant-client` 已删除 `search()` 方法**
教程代码用的是 `client.search(query_vector=...)`。qdrant-client 1.19.1 中该方法
已被移除，现在必须用 `query_points(query=...)`。已修正。

**3. `huggingface-cli` 命令已改名**
新版 `huggingface_hub` 中 `huggingface-cli` 已改为 `hf`。
实际命令：`hf download NeuML/pubmedbert-base-embeddings --local-dir ./pubmedbert-embeddings`

**4. HuggingFace 主站无法直连**
本机访问 `huggingface.co` 超时。所有 HF 操作需加环境变量走镜像：

```bash
HF_ENDPOINT=https://hf-mirror.com
```

该变量已写入 `rag_engine.py`，代码中会自动生效。

**5. Ollama 下载模型卡死（IPv6 问题）**
`ollama pull` 会卡在 `pulling manifest` 无限重试。原因是模型文件托管在
Cloudflare R2 CDN，本机 IPv6 不通但 Windows 优先走 IPv6，导致连接超时。

绕过方式：不通过 Ollama registry 下载，改为从 **ModelScope**（国内源，12MB/s）
下载 GGUF 文件，再用 `ollama create` 导入：

```bash
ollama create qwen2.5:7b -f Modelfile
```

> 如果希望以后 `ollama pull` 能正常工作，需要禁用 IPv6（管理员权限）：
> `netsh interface ipv6 set global randomizeidentifiers=disabled`
> 或在网卡属性中取消勾选「Internet 协议版本 6 (TCP/IPv6)」。

**6. 中文问题不能直接拿去检索 PubMed（重要）**

PubMed 是英文文献库。实测拿中文提问，返回的是**完全无关**的结果：

| 查询 | 返回的第一条 |
|---|---|
| `二甲双胍对心血管有什么保护作用？` | *Wings of access: 无人机医疗物资配送集群 RCT* |
| `metformin cardiovascular protective effects` | *Genetic Evidence for the Benefits and Risks of Glucose-Lowering Drugs on Cardiovascular...* |

本项目在 `rag_engine.py` 中加了 `translate_to_pubmed_query()`：
检索前先用本地模型把中文问题转成英文关键词。这一步不能省。

**7. GGUF 分片格式无法直接导入 Ollama**

HuggingFace / ModelScope 上超过约 4GB 的 GGUF 通常被切成 `-00001-of-00002.gguf`
这样的分片。把两片 `cat` 合并**不行**（文件头仍标记为分片，报
`invalid split GGUF`）；只指向第一片也不行（Ollama 只把该片拷进 blob 存储，
报 `has 1 shards, expected 2`）。

解决办法：直接找**单文件**版本。本项目用的是 ModelScope 上
`bartowski/Qwen2.5-7B-Instruct-GGUF` 仓库的 `Qwen2.5-7B-Instruct-Q4_K_M.gguf`。

**8. `.bat` 文件必须用 GBK 编码 + CRLF 换行（启动失败的元凶）**

如果 `启动助手.bat` 存成 UTF-8 + LF 换行，双击后会报一串：

```
'仴搴锋枃鐚姪鎵?echo' 不是内部或外部命令
'氭嫙鐜鍕?' 不是内部或外部命令
'streamlit' 不是内部或外部命令
```

原因有两层，缺一不可：

- **编码**：cmd.exe 在中文系统上按 GBK 解码脚本。UTF-8 的中文字节被误读成
  GBK 乱码（`个人健康文献助手` → `仴搴锋枃鐚姪鎵`）。
- **换行**：这才是致命的。`手` 的 UTF-8 编码是 `E6 89 8B`，
  cmd 读走 `E6 89` 后，剩下的 `8B` 会**吞掉后面的 `0A` 换行符**，
  于是下一行的 `echo` 被粘到上一行末尾，整份脚本逐行解析失败——
  `call venv\Scripts\activate.bat` 根本没执行，所以提示找不到 streamlit。

修复（本项目已处理）：

```bash
sed -i 's/$/\r/' 启动助手.bat                      # LF → CRLF
iconv -f UTF-8 -t GBK 启动助手.bat > tmp && mv tmp 启动助手.bat   # UTF-8 → GBK
```

> 用编辑器打开这个文件时，请选 **ANSI / GBK** 编码保存，不要选 UTF-8。

**9. Streamlit 首次运行会卡在邮箱询问**

第一次启动时会弹出：

```
Welcome to Streamlit!
Email:      ← 卡死在这里，服务器根本没起来
```

必须建 `C:\Users\19408\.streamlit\credentials.toml`（用户主目录，**不是**项目目录）：

```toml
[general]
email = ""
```

本项目同时在 `.streamlit/config.toml` 里关掉了文件监视器，
以消除启动时那条来自 `transformers.models.aria` 的无害 Traceback。

**10. 首屏要 16.7 秒才出现输入框（已修复）**

原本 `app.py` 第 2 行就是：

```python
from rag_engine import ask_health_question
```

这会**在模块导入时**就加载 torch + sentence_transformers + qdrant
并读取 PubMedBERT 权重，实测让首屏渲染阻塞 **16.7 秒**。
这期间浏览器里只有灰色骨架屏，看不到任何输入框；
等太久还可能被浏览器判定为 WebSocket 超时，直接显示
`Connection error — streamlit run yourscript.py`。

改成延迟加载后，首屏 **0.70 秒**（快 24 倍）：

```python
@st.cache_resource(show_spinner="正在加载医学模型与向量库（首次约 15 秒，之后瞬时）...")
def load_rag_engine():
    from rag_engine import ask_health_question   # 挪到函数内部
    return ask_health_question
```

模型在**第一次提问时**加载一次，之后由 `cache_resource` 缓存复用。
提问时会看到明确的加载提示，不再是莫名其妙的空白。

> 结论：重型依赖（torch、transformers、模型权重）永远不要放在
> Streamlit 应用文件的顶层导入。

**11. `pymedx` 的 `publication_date` 有时是字符串**

原代码写的是 `pub_date.year if pub_date else None`。但 `pymedx` 返回的
`publication_date` 有时是 `date` 对象，有时直接是字符串（`"2023"`、
`"2023 May"`）。遇到字符串就抛 `AttributeError: 'str' object has no attribute 'year'`，
而且它是在**遍历结果的中途**抛的，整条检索直接中断。

已在 `pubmed_search.py` 加 `extract_year()` 兼容两种形态。

**12. 「回答太宽泛」的根因是单条检索式**

用户反馈「回答比较宽泛，没有详细介绍疗法，只概括了文献内容」。原因不在提示词，
而在检索环节：原来整个流程只用**一条**检索式，召回的全是泛泛的综述，
治疗方案类文献根本没进上下文——模型自然只能概括。

临床问诊模式改为按「诊断 / 治疗 / 检查」分别生成检索式再合并召回，
治疗方案那一条才会真正拉回治疗类文献。

**13. 引用会过度集中到同一篇文献**

实测第一次生成报告时，几乎所有结论都标成 `[文献 4]`——引用失去区分度。
已在系统提示词中加规则 6 要求引用有区分度，并在正文后**用 Python 追加
真实参考文献列表**（不经模型之手，避免标题/期刊/年份张冠李戴）。

---

## 工作原理

### 临床问诊模式（主流程）

```
填写问诊表单（症状 / 病程 / 既往史 / 用药 / 年龄性别）
   ↓
【1】拼装中文病例小结 —— 纯字符串拼接，结构化字段不让模型去"理解"
   ↓
【2】生成 3 条英文检索式 —— 分别面向：鉴别诊断 / 治疗方案 / 相关检查
   ↓
【3】逐条检索 PubMed 并按 pmid 去重 —— 单条失败不影响整体
   ↓
【4】PubMedBERT 向量化 → 存入 Qdrant 本地向量库（按 pmid 派生固定 ID，天然幂等）
   ↓
【5】多路召回：每条检索式分别匹配，按相似度合并去重
   ↓
【6】临床推理提示词 → 本地 Qwen2.5 生成结构化报告
   ↓
【7】Python 追加真实参考文献列表（带 PubMed 链接）
```

报告固定为六个章节：临床小结 / 鉴别诊断 / 建议检查 / 治疗方案方向 /
需警惕的警示信号 / 免责声明。

### 文献速查模式（旧流程）

```
用户提问（中文）
   ↓
转成单条英文检索式 → PubMed 检索 → 向量化入库 → 匹配最相关 5 篇 → 生成回答
```

> 注意：`ask_health_question()` 的提示词仍强制「只复述文献、不诊断」。
> 临床角色是通过 `clinical_consultation()` 里的 `system` 消息设定的，
> 因此**不需要改 `Modelfile`、也不需要重新 `ollama create`**。

### 防幻觉设计

`rag_engine.py` 的提示词中强制了 5 条规则：

1. 每条结论必须标注来源编号
2. 文献不足时必须明说「无法确定」
3. 文献间矛盾结论要如实呈现分歧
4. 禁止编造文献外的结论；与内部知识冲突时以文献为准
5. 固定附加免责声明

同时 `temperature=0.2` 降低随机性。

---

## 已知局限（务必了解）

**临床问诊模式的推理质量受限于 7B 模型。**

报告的结构是对的（六个章节齐全、引用可追溯），但**推理内容不能当真**。
实测一例「52 岁女性、上腹隐痛餐后加重、反酸嗳气」，模型给出的"反对点"里
出现了「消化性溃疡的治疗首选质子泵抑制剂，但患者目前未使用」——
这不是反对点，是把治疗方案错当成了否定证据。

**引用编号是真实的，结论不是。** 参考文献列表由 Python 从检索结果直接生成，
所以标题、期刊、年份、链接都可信；但正文里「文献 N 支持某结论」的对应关系
是模型自己判断的，可能标错。

7B 模型能做的是**整理和结构化**，不是**临床推理**。这个工具的定位应该是
「帮你快速找到相关文献并按临床框架整理」，而不是「替你看病」。

**模型会搞混药物实体。** 实测提问「二甲双胍对心血管有什么保护作用」时，
模型输出了「二甲双胍（Ertugliflozin）」——Ertugliflozin 是 SGLT2 抑制剂，
根本不是二甲双胍。7B 模型在并列罗列多种药物时容易张冠李戴。

现有的 5 条提示词规则**拦不住这类错误**：模型确实引用了文献，只是把文献里的
药物名对错了对象。使用时务必对着原文核对关键结论和药物名称。

可行的改进方向：

- 在提示词中增加「引用药物名称前必须逐字核对原文」
- 提问时把问题限定得更窄（一次只问一种药），减少模型混淆的机会
- 换更大的模型（如 `qwen2.5:14b`，8.99GB，16GB 内存可跑但会变慢）

~~另外，`ingest_articles()` 目前每次提问都会往向量库追加新记录，
同一个问题问多次会存入重复文献。~~ **已修复**：现在用 `pmid` 派生确定性
point ID，`qdrant.upsert` 遇到相同 ID 会覆盖，重复入库不再产生冗余数据。

---

## 后续可做的优化

- **换更大的模型（提升临床推理质量的最有效手段）**：7B 是当前报告质量的
  硬天花板。`ollama pull qwen2.5:14b`（8.99GB，16GB 内存可跑但会明显变慢），
  或改用 `qwen2.5:32b` 需要更大内存。换模型只需改 `rag_engine.py` 顶部的
  `LLM_MODEL` 常量。
- **收紧防幻觉规则**：若仍偶有"发挥"，把 `temperature` 降到 0.1
- **加入临床指南**：用 `PyPDFLoader` 把 ADA / AHA 等指南 PDF 存入 Qdrant，
  与 PubMed 文献一起检索（指南是比原始研究更直接的证据来源）
- **混合检索**：向量检索 + BM25 关键词检索，可用 LangChain 的
  `EnsembleRetriever` 组合，对药物名、基因位点这类精确术语提升明显
- **申请 NCBI API Key**：免费，可把 PubMed 调用频率从 3 次/秒提到 10 次/秒

---

## 免责声明

本工具基于医学文献检索生成回答，**不构成医疗建议**。
如有健康问题，请咨询专业医生。
