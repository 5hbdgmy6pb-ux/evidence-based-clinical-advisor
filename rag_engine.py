import os
import uuid

# HuggingFace 主站在国内常被墙，默认走镜像（已下载到本地后此项不影响运行）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import ollama
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from pubmed_search import search_medical_literature

# ========== 初始化组件 ==========

# 本地大模型。BioMistral 不在 Ollama 官方库中（拉取会 404），
# 这里用 qwen2.5:7b 替代：中英双语，处理「中文提问 + 英文文献」效果好。
LLM_MODEL = "qwen2.5:7b"

# 嵌入模型：将医学文本转为向量。
# 优先使用本地已下载的目录；否则自动从 HuggingFace 镜像下载并缓存。
EMBED_MODEL_LOCAL_DIR = "./pubmedbert-embeddings"
EMBED_MODEL_HF_ID = "NeuML/pubmedbert-base-embeddings"

if os.path.isdir(EMBED_MODEL_LOCAL_DIR) and os.listdir(EMBED_MODEL_LOCAL_DIR):
    embed_model = SentenceTransformer(EMBED_MODEL_LOCAL_DIR)
else:
    embed_model = SentenceTransformer(EMBED_MODEL_HF_ID)

# PubMedBERT-base 的向量维度为 768
EMBED_DIM = 768

# 本地向量数据库（嵌入式模式，无需额外启动服务）
qdrant = QdrantClient(path="./local_qdrant")

# 创建 collection（如果不存在）
COLLECTION_NAME = "medical_literature"
try:
    qdrant.get_collection(COLLECTION_NAME)
except Exception:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )


def _point_id(article):
    """
    用 pmid 生成确定性 ID。

    原先用 uuid.uuid4()，每次入库都是新记录，导致同一个问题问多次会在
    向量库里堆满重复文献（实测 15 篇文献变成 30 个点）。改用 pmid 派生
    的固定 ID 后，qdrant.upsert 遇到相同 ID 会覆盖，天然幂等。
    """
    key = article.get("pmid") or article.get("title") or ""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"pubmed:{key}"))


def ingest_articles(articles):
    """将检索到的文献摘要向量化并存入本地数据库"""
    points = []
    for article in articles:
        text = f"Title: {article['title']}\nAbstract: {article['abstract']}"
        vector = embed_model.encode(text).tolist()
        points.append(PointStruct(
            id=_point_id(article),
            vector=vector,
            payload={
                "text": text,
                "title": article["title"],
                "journal": article["journal"],
                "year": article["year"],
                "pmid": article.get("pmid"),
                "doi": article.get("doi"),
            },
        ))
    if points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)


def retrieve_relevant_articles(query, top_k=5):
    """从本地数据库中检索与问题最相关的文献片段"""
    query_vector = embed_model.encode(query).tolist()
    # 注意：qdrant-client 1.19 已移除 client.search()，改用 query_points()
    result = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    )
    hits = result.points
    return [
        {
            "text": hit.payload["text"],
            "title": hit.payload["title"],
            "journal": hit.payload["journal"],
            "year": hit.payload["year"],
            # pmid 用于生成参考文献的 PubMed 链接；
            # 用 .get 是因为早期入库的点（首次运行前）可能没有这个字段
            "pmid": hit.payload.get("pmid"),
            "score": hit.score,
        }
        for hit in hits
    ]


def translate_to_pubmed_query(user_question):
    """
    把用户的中文问题转成英文检索关键词。

    PubMed 是英文文献库，实测直接拿中文提问检索会返回完全无关的结果
    （例如问二甲双胍的心血管作用，返回的却是无人机送药、助产士支持之类）。
    所以检索前必须先转成英文。
    """
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{
            "role": "user",
            "content": (
                "把下面的医学问题转换成简洁的英文检索关键词，用于 PubMed 文献检索。"
                "只输出英文关键词本身，不要任何解释、引号或多余标点。\n\n"
                f"问题：{user_question}"
            ),
        }],
        options={"temperature": 0},  # 翻译不需要随机性
    )
    query = response["message"]["content"].strip().strip('"').strip()
    return query or user_question


def ask_health_question(user_question):
    """
    完整的 RAG 流程：
    1. 把中文问题转成英文检索式
    2. 从 PubMed 实时检索文献
    3. 存入本地向量库
    4. 检索最相关的片段
    5. 交给本地模型生成带引用的回答
    """
    # 第 1 步：转成英文检索式（PubMed 对英文查询的检索质量远高于中文）
    print("正在转换检索关键词...")
    search_query = translate_to_pubmed_query(user_question)

    # 第 2 步：实时检索 PubMed
    print(f"正在检索 PubMed 文献...（检索式：{search_query}）")
    articles = search_medical_literature(search_query, max_results=15)

    if not articles:
        return "抱歉，未能在 PubMed 中检索到相关文献。建议换一种问法，或咨询专业医生。"

    # 第 3 步：存入向量库
    ingest_articles(articles)

    # 第 4 步：从向量库中检索最相关的片段
    # 用英文检索式而不是中文原句：PubMedBERT 是英文模型，中文向量匹配不准
    relevant_docs = retrieve_relevant_articles(search_query, top_k=5)

    # 构建上下文
    context = "\n\n---\n\n".join([
        f"[文献 {i + 1}] {doc['title']} ({doc['journal']}, {doc['year']})\n{doc['text']}"
        for i, doc in enumerate(relevant_docs)
    ])

    # 第 5 步：构造提示词，强制引用
    prompt = f"""你是一个严格的医学文献助手。请根据以下检索到的文献回答用户问题。

【检索到的文献】
{context}

【用户问题】
{user_question}

【回答规则】
1. 每条医学结论必须标注来源，格式为“[文献编号]”。
2. 如果文献不足以回答，必须明确说“根据当前检索到的文献，无法确定”。
3. 如果文献之间存在矛盾结论，如实呈现分歧。
4. 禁止编造任何不在文献中的医学结论；如果某条文献与你内部知识冲突，以文献为准，不要引入文献之外的知识。
5. 回答末尾必须附加：“以上信息基于医学文献检索，不构成医疗建议，请咨询专业医生。”

【回答】"""

    # 调用本地模型
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2},  # 低温度，减少随机性
    )

    return response["message"]["content"]


# ==========================================================================
# 临床问诊模式：症状 + 病史 → 鉴别诊断 → 循证治疗方案
# ==========================================================================
#
# 与上面的 ask_health_question() 的区别：
#   ask_health_question  是「文献速查」——不诊断、不给方案，只复述文献。
#   clinical_consultation 是「循证临床顾问」——做临床推理，再用文献佐证。
#
# 关键差异在检索环节：单一检索式只能召回一类文献，问诊断就漏掉治疗方案。
# 这里改成「按诊断 / 治疗 / 检查分别生成检索式，再合并召回」。


CLINICAL_SYSTEM_PROMPT = """你是一位经验丰富的循证临床顾问。你会先做临床推理，再用检索到的医学文献佐证你的判断。

你必须遵守以下规则：
1. 只做鉴别诊断：给出多个可能疾病并按可能性从高到低排序，绝不给出单一确诊结论。
2. 每条判断尽量标注文献来源，格式为 [文献编号]。
3. 治疗方案只给方向：药物类别、生活方式干预、需做的检查、随访计划。
   严禁给出具体剂量、具体处方或给药方案。
4. 文献不足或你不确定时，必须明确说明，不要编造。
   如果某条文献与你内部知识冲突，以文献为准，不要引入文献之外的知识。
5. 必须完整输出全部规定章节，包括最后的免责声明，不得省略或截断。
6. 引用要有区分度：每条结论引用最相关的那一篇文献，
   不要把几乎所有结论都归到同一篇文献上。"""


def build_patient_profile(intake):
    """
    把问诊表单字段拼成一段紧凑的中文临床小结。

    纯字符串拼接，不调用模型——年龄、病程这类结构化字段不该让模型去"理解"。
    """
    fields = [
        ("年龄", intake.get("age")),
        ("性别", intake.get("sex")),
        ("主诉与症状", intake.get("complaint")),
        ("病程", intake.get("duration")),
        ("既往史", intake.get("history")),
        ("当前用药", intake.get("medications")),
        ("其他补充", intake.get("notes")),
    ]
    return "\n".join(f"{label}：{value}" for label, value in fields if value)


def generate_clinical_queries(profile, max_queries=3):
    """
    根据病例生成 2~3 条英文 PubMed 检索式。

    这是「回答太宽泛」的根因所在：原方案只用一条检索式，召回的全是泛泛的
    综述。拆成诊断 / 治疗 / 检查三条后，治疗方案那条才会真正召回治疗类文献。
    """
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{
            "role": "user",
            "content": (
                "下面是一份中文病例。请为 PubMed 文献检索生成英文检索式。\n"
                "要求：\n"
                "1. 每行一条，最多 3 条；不要编号、不要解释、不要引号。\n"
                "2. 第 1 条面向「鉴别诊断 / 疾病临床特征」。\n"
                "3. 第 2 条面向「最可能疾病的治疗与管理指南」。\n"
                "4. 第 3 条可选，补充相关检查或预后。\n"
                "5. 每条只用英文医学术语，不要写成整句。\n\n"
                f"病例：\n{profile}"
            ),
        }],
        options={"temperature": 0},  # 生成检索式不需要随机性
    )

    queries = []
    for line in response["message"]["content"].splitlines():
        # 模型可能不听劝加上 "1. " / "- " 前缀，这里统一剥掉
        q = line.strip().lstrip("0123456789.-)*） ").strip().strip('"').strip()
        if q and q not in queries:
            queries.append(q)

    queries = queries[:max_queries]
    if not queries:
        # 兜底：退回旧的单条转换，至少能检索
        queries = [translate_to_pubmed_query(profile)]
    return queries


def search_and_collect(queries, max_results=15):
    """
    逐条检索 PubMed，合并结果，按 pmid 去重。

    单条检索式失败（网络抖动 / NCBI 限流 / 某条检索式语法不被接受）不应让
    整个问诊中断——其余检索式的结果仍然可用，所以这里逐条 try。
    """
    seen = set()
    articles = []
    for q in queries:
        try:
            results = search_medical_literature(q, max_results=max_results)
        except Exception as e:
            print(f"检索式失败，已跳过：{q}（{e}）")
            continue
        for article in results:
            key = article.get("pmid") or article.get("title")
            if not key or key in seen:
                continue
            seen.add(key)
            articles.append(article)
    return articles


def retrieve_multi(queries, top_k=6):
    """
    对每条检索式分别召回，按标题去重、按相似度分数降序合并。

    复用单查询版的 retrieve_relevant_articles()，保持向后兼容。
    """
    best = {}
    for q in queries:
        for doc in retrieve_relevant_articles(q, top_k=top_k):
            title = doc["title"]
            if title not in best or doc["score"] > best[title]["score"]:
                best[title] = doc
    return sorted(best.values(), key=lambda d: d["score"], reverse=True)[:top_k]


def generate_clinical_report(profile, context):
    """生成结构化的鉴别诊断 + 治疗方案报告。"""
    prompt = f"""【病例】
{profile}

【检索到的医学文献】
{context}

请严格按以下六个章节输出，章节标题必须逐字照写，不要增删章节：

【临床小结】
用 2~3 句话概括这个病例的核心临床特征。

【鉴别诊断】
按可能性从高到低列出候选疾病。每一项写明：
- 疾病名称 + 可能性（高 / 中 / 低）
- 支持点与反对点
尽量标注 [文献编号]。

【建议检查】
为区分上述鉴别诊断，建议做哪些检查，并说明每项检查的目的。

【治疗方案方向】
分三部分写：药物类别、生活方式干预、随访计划。
只写方向，不写具体剂量与处方。尽量标注 [文献编号]。

【需警惕的警示信号】
列出哪些情况一旦出现必须立即就医。

【免责声明】
固定写上：以上分析基于医学文献检索，仅供参考，不构成医疗建议，请务必咨询专业医生。

【报告】"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            # 以 system 消息设定临床角色，覆盖 Modelfile 里「只复述文献」的约束，
            # 因此不需要改 Modelfile、也不需要重新 ollama create。
            {"role": "system", "content": CLINICAL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        # num_ctx 提到 16384：报告有六个章节，8192 会在末尾（免责声明）被截断
        options={"temperature": 0.2, "num_ctx": 16384},
    )
    return response["message"]["content"]


def format_references(docs):
    """
    生成参考文献列表。

    刻意用 Python 拼接而不是让模型输出：模型列的文献表很容易张冠李戴
    （标题、期刊、年份对不上）。这里直接来自检索结果，编号与报告里的
    [文献 N] 一一对应，用户可据此核对原文。
    """
    lines = ["\n\n---\n\n### 参考文献（与上文 [文献 N] 对应）\n"]
    for i, doc in enumerate(docs):
        year = doc["year"] or "年份不详"
        pmid = doc.get("pmid")
        link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
        entry = f"**[文献 {i + 1}]** {doc['title']} — *{doc['journal']}*, {year}"
        if link:
            entry += f" · [PubMed]({link})"
        lines.append(entry)
    return "\n\n".join(lines)


def clinical_consultation(intake):
    """
    完整问诊流程：
    1. 表单字段 → 中文临床小结
    2. 生成 2~3 条英文检索式（诊断 / 治疗 / 检查）
    3. 逐条检索 PubMed 并按 pmid 去重
    4. 入库 + 多路召回
    5. 生成结构化临床报告
    """
    profile = build_patient_profile(intake)
    if not profile.strip():
        return "请先填写症状与病史。"

    print("正在根据病例生成检索式...")
    queries = generate_clinical_queries(profile)
    print(f"检索式（{len(queries)} 条）：")
    for q in queries:
        print(f"  - {q}")

    print("正在检索 PubMed 文献...")
    articles = search_and_collect(queries, max_results=15)
    if not articles:
        return "抱歉，未能在 PubMed 中检索到相关文献。建议把症状描述得更具体，或咨询专业医生。"

    ingest_articles(articles)
    relevant_docs = retrieve_multi(queries, top_k=6)

    context = "\n\n---\n\n".join([
        f"[文献 {i + 1}] {doc['title']} ({doc['journal']}, {doc['year']})\n{doc['text']}"
        for i, doc in enumerate(relevant_docs)
    ])

    print("正在生成临床报告...")
    report = generate_clinical_report(profile, context)

    # 参考文献列表由 Python 从真实检索结果生成，不经模型之手，
    # 保证标题/期刊/年份不会张冠李戴
    return report + format_references(relevant_docs)
