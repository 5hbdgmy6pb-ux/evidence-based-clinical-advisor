import os
import re
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
# 临床问诊模式：症状 + 病史 → 分流 → 鉴别诊断 → 治疗方案
# ==========================================================================
#
# 与上面的 ask_health_question() 的区别：
#   ask_health_question  是「文献速查」——不诊断、不给方案，只复述文献。
#   clinical_consultation 是「循证临床顾问」——做临床推理，再用文献佐证。
#
# 两处关键设计：
#
# 1. 检索环节：单一检索式只能召回一类文献，问诊断就漏掉治疗方案。
#    这里改成「按诊断 / 治疗 / 检查分别生成检索式，再合并召回」。
#
# 2. 分流环节：无差别地跑文献管线，会让「受凉了肚子不舒服」这种常见轻症
#    被硬套上重症文献。原因有两层——(a) PubMed 的高证据过滤
#    （systematic / meta / RCT / review / guideline）天然偏向「值得做试验」
#    的疾病，轻症管理几乎不会被收录；(b) 系统提示词原先禁止引入文献之外的
#    知识，模型明知文献不对口也只能照着说，于是给出「查幽门螺杆菌、考虑 PPI」
#    这类不现实的建议。所以要先拦急症、再区分轻症与需循证（见下节）。


CLINICAL_SYSTEM_PROMPT = """你是一位经验丰富的循证临床顾问。你会先做临床推理，再用检索到的医学文献佐证你的判断。

你必须遵守以下规则：
1. 只做鉴别诊断：给出多个可能疾病并按可能性从高到低排序，绝不给出单一确诊结论。
2. 每条判断尽量标注文献来源，格式为 [文献编号]。
3. 治疗方案只给方向：药物类别、生活方式干预、需做的检查、随访计划。
   严禁给出具体剂量、具体处方或给药方案。
4. 文献是主要依据，但不是唯一依据：
   - 当检索到的文献与当前病例不相关、不足以支撑某个判断，或明显违背医学常识时，
     可以给出基于一般医学常识的建议，但必须在该条末尾标注「（一般建议，非文献）」，
     并简要说明文献为何不适用。
   - 绝不允许把常识性建议伪装成有文献依据，也不许给它编造一个 [文献编号]。
   - 文献确实相关时，仍须按要求标注 [文献编号]；文献不足或你不确定时，
     必须明确说明，不要编造。
   - **引用前先核对文献的研究人群与场景是否与本例相符。** 儿科、ICU / 危重患者、
     免疫抑制人群、罕见病的研究，不能用来支持普通成人门诊病例的判断。
     检索结果里这类研究很常见（高证据文献天然偏向重症），不要因为它们
     "看起来相关"（比如都涉及腹痛、腹泻）就拿来引用。
5. 必须完整输出全部规定章节，包括最后的免责声明，不得省略或截断。
6. 引用要有区分度：每条结论引用最相关的那一篇文献，
   不要把几乎所有结论都归到同一篇文献上。"""


# --------------------------------------------------------------------------
# 分流：先拦急症，再决定报告以常识为主还是以文献为主
# --------------------------------------------------------------------------

# 红旗警示词。命中即视为急症，直接提示立即就医，不再做任何分析。
#
# 匹配方式是「子串包含」，因此必然会有误报——无法穷尽所有表述。这里刻意偏向
# 误报而不是漏报：漏掉一个急症的代价，远大于多提示一次就医。
#
# 只做了一层很窄的否定识别（见 _is_negated）：否定词必须紧贴在关键词前面
# （如「无胸痛」「否认胸痛」）才算数。像「无发热、咳嗽、胸痛」这种用顿号
# 并列列举的写法，否定词与关键词之间隔着分隔符，仍然会触发——宁可误报。
RED_FLAG_KEYWORDS = [
    # 心肺
    "胸痛", "胸闷压榨", "呼吸困难", "喘不上气", "咯血", "咳血", "窒息",
    # 消化道出血
    "呕血", "吐血", "便血", "黑便", "柏油样便", "血便",
    # 神经系统（脑卒中、癫痫）
    "剧烈头痛", "炸裂样头痛", "言语不清", "说话不清", "口角歪斜", "嘴角歪",
    "肢体无力", "半身麻木", "一侧无力", "偏瘫", "意识不清", "昏迷",
    "晕厥", "昏倒", "抽搐", "惊厥",
    # 严重感染
    "高热不退", "持续高热", "寒战高热",
    # 腹部急症
    "剧烈腹痛", "腹痛难忍", "肚子剧痛", "板状腹",
    # 泌尿
    "血尿",
    # 严重过敏
    "喉头水肿", "喉咙肿", "过敏性休克",
    # 自伤风险
    "自杀", "轻生", "不想活", "自残",
    # 外伤与大出血
    "大出血", "严重外伤", "骨折", "车祸",
    # 孕产
    "阴道出血", "胎动减少", "临产",
]

NEGATION_PREFIXES = ("无", "否认", "没有", "未出现", "未见", "不伴", "并无")


def _is_negated(text, start):
    """
    判断某个关键词是否被紧贴其前的否定词否定（「无胸痛」「否认胸痛」）。

    只回看关键词前面很窄的一段，且**不做分隔符跳跃**，这是刻意的：
    如果跳过顿号去匹配更远处的否定词，「无高血压，有胸痛」会被误判成否定，
    从而漏掉真正的红旗。宁可误报，不可漏报。
    """
    return any(
        text[max(0, start - len(neg)):start] == neg for neg in NEGATION_PREFIXES
    )


def detect_red_flags(profile_text):
    """扫描病例文本，返回命中的红旗关键词（已排除被紧邻否定词否定的）。"""
    hits = []
    for keyword in RED_FLAG_KEYWORDS:
        start = profile_text.find(keyword)
        while start != -1:
            if not _is_negated(profile_text, start):
                hits.append(keyword)
                break
            start = profile_text.find(keyword, start + 1)
    return hits


def build_emergency_response(reasons, hint_rephrase=False):
    """
    急症提示。

    刻意用固定模板而不是让模型生成——模型很可能把「立即就医」稀释成一句
    轻描淡写的建议。这是整条流程里最不能交给模型的一步。
    """
    reason_text = "、".join(reasons)
    response = f"""# ⚠️ 请立即就医

你描述的情况中包含需要紧急处理的线索：**{reason_text}**。

**请立刻前往最近医院的急诊，或拨打 120。** 不要等待症状自行缓解，
也不要先自行做其他检查或用药。

本工具**不会**对这种情况做鉴别诊断或治疗建议——急症需要医生现场评估，
任何基于文字的分析都可能延误处理。

---

如果你填写的其实是**否定**表述被误判了（例如「无胸痛」被读成了「胸痛」），
请在「主诉与症状」里改用更明确的写法，例如「没有胸痛，主要是……」，然后重新提交。

---

以上信息仅供参考，不构成医疗建议。"""

    if hint_rephrase:
        response += """

> 说明：触发本次提示的是固定的关键词匹配。它宁可误报也不漏报，
> 所以看到这条提示不代表一定有问题——但如果症状确实存在，请务必就医。"""

    return response


TRIAGE_PROMPT = """你是一位急诊分诊护士，需要为下面的病例做快速分诊。

请**只**输出下面两行，不要任何解释、不要其他内容：

RED_FLAG=是 或 RED_FLAG=否
MODE=SELF_CARE 或 MODE=EVIDENCE

判断标准：

- RED_FLAG=是：存在需要立即就医的急症征象，例如急性心肌梗死、脑卒中、
  消化道大出血、急腹症、严重过敏、意识障碍、自伤风险等。**拿不准就填「是」。**
- MODE=SELF_CARE：常见、轻微、自限性的不适，以家庭护理和观察为主即可，
  例如受凉后胃肠不适、普通感冒、轻微肌肉酸痛、偶尔的轻微头痛。
- MODE=EVIDENCE：症状持续、反复、原因不明，或需要鉴别诊断，
  应当以医学文献为依据，例如长期上腹痛、不明原因的体重下降、反复发热。

病例：
{profile}"""


def triage_case(profile):
    """
    模型分流：判断是否为急症，以及报告应以常识为主还是以文献为主。

    这是关键词筛查之后的第二道，作用是兜住关键词没覆盖到的急症表述
    （关键词表无法穷尽），并区分「常见轻症」和「需要循证」。

    解析失败或调用出错时，兜底为 {"red_flag": False, "mode": "evidence"}——
    evidence 模式更保守（以文献为主要依据），比误判成轻症安全。
    """
    fallback = {"red_flag": False, "mode": "evidence"}

    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{
                "role": "user",
                "content": TRIAGE_PROMPT.format(profile=profile),
            }],
            options={"temperature": 0},  # 分诊不需要随机性
        )
    except Exception as e:
        print(f"分流调用失败，按默认（需循证）处理：{e}")
        return fallback

    # 模型可能用全角冒号或加空格，统一成一种写法再匹配
    compact = re.sub(r"[\s　]", "", response["message"]["content"])
    compact = compact.replace("：", "=").replace(":", "=").upper()

    if "RED_FLAG" not in compact:
        print(f"分流输出无法解析，按默认（需循证）处理：{compact[:80]}")
        return fallback

    return {
        "red_flag": "RED_FLAG=是" in compact,
        "mode": "self_care" if "MODE=SELF_CARE" in compact else "evidence",
    }


# 分流结论注入报告提示词：告诉模型本次该以常识还是以文献为主
MODE_GUIDANCE = {
    "self_care": (
        "【本次分流判断】常见轻症。以一般健康指导为主，文献仅作参考。\n"
        "本模式下必须遵守以下额外要求：\n"
        "1. 检索到的文献很可能聚焦于重症、特殊人群或住院患者，与本例不对口。\n"
        "   研究人群或疾病严重程度与本例明显不符的文献，**直接不要引用**，\n"
        "   也不要把它的结论套到本例上。宁可全篇没有 [文献编号]，也不要错引。\n"
        "2. 【鉴别诊断】可以只写一句：「本例为常见轻症，无需鉴别诊断\n"
        "   （一般建议，非文献）」。不要为了凑内容而罗列可能性很低的疾病。\n"
        "3. 【建议检查】如无必要，直接写「无需特殊检查（一般建议，非文献）」。\n"
        "4. 【随访计划】不要写「X 天后复诊」这类要求，改为说明\n"
        "   **什么情况下才需要就医**。\n"
        "5. 每一行建议都必须带标注：来自文献的标 [文献编号]，\n"
        "   其余一律标「（一般建议，非文献）」。不允许出现没有任何标注的建议行。"
    ),
    "evidence": (
        "【本次分流判断】需要循证。以检索到的文献为主要依据。\n"
        "只有在文献确实不适用时，才使用「（一般建议，非文献）」的标注。"
    ),
}


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


# 检查/操作类词汇。检索式一旦带上它们，PubMed 会整片召回顾内镜、影像、
# 外科手术类文献——实测「上腹隐痛 2 天」被翻成 gastrointestinal endoscopy 后，
# 36 篇里二十多篇是内镜/结肠镜/ERCP/ESD，跟「这个症状可能是什么病」无关。
# 提示词里已经写了禁令，但 7B 并不总听，所以这里再做一道确定性过滤。
_PROCEDURE_QUERY_RE = re.compile(
    r"\b("
    r"endoscop\w*|colonoscop\w*|gastroscop\w*|sigmoidoscop\w*|laparoscop\w*|"
    r"bronchoscop\w*|biops\w*|ultrasound|sonograph\w*|radiograph\w*|"
    r"tomograph\w*|imaging|mri|ct|screen\w*|surg\w*|operat\w*|resection"
    r")\b",
    re.IGNORECASE,
)

# 空壳检索式。禁掉检查类词汇后，模型会退化成只写 "management guidelines"、
# "prognosis" 这种不含任何疾病/症状名的检索式——它们同样召回随机文献
# （实测召回肾移植随访、卟啉病、心衰）。要求每条检索式里至少有一个
# 具体词，否则丢弃。
_GENERIC_QUERY_WORDS = {
    "management", "treatment", "therapy", "therapies", "guideline",
    "guidelines", "prognosis", "diagnosis", "diagnostic", "follow",
    "up", "screening", "care", "overview", "review", "approach",
    "clinical", "evidence", "current", "recent", "update", "practice",
    "of", "for", "and", "in", "on", "the", "a", "an", "to", "with",
}


def _is_generic_query(query):
    """检索式里若没有任何具体疾病/症状词，就只是个空壳，应予丢弃。"""
    tokens = re.findall(r"[a-z]+", query.lower())
    return not tokens or all(t in _GENERIC_QUERY_WORDS for t in tokens)


def generate_clinical_queries(profile, max_queries=3):
    """
    根据病例生成 2~3 条英文 PubMed 检索式。

    这是「回答太宽泛」的根因所在：原方案只用一条检索式，召回的全是泛泛的
    综述。拆成诊断 / 治疗 / 预后三条后，治疗方案那条才会真正召回治疗类文献。
    """
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{
            "role": "user",
            "content": (
                "下面是一份中文病例。请为 PubMed 文献检索生成英文检索式。\n"
                "要求：\n"
                "1. 每行一条，共 3 条；不要编号、不要解释、不要引号。\n"
                "2. 三条分别面向：疾病临床特征与鉴别、治疗与管理、预后与随访。\n"
                "3. 【每一条都必须包含具体的疾病名或症状名】。只写\n"
                "   management、guidelines、prognosis、treatment 这类空泛词\n"
                "   会被丢弃，因为那样的检索式召回的是随机文献。\n"
                "   反例：management guidelines、prognosis、treatment\n"
                "   正例：acute gastroenteritis management、dyspepsia prognosis\n"
                "   若还不能确定具体疾病，就用症状名，例如 epigastric pain。\n"
                "4. 每条只用英文医学术语，不要写成整句。\n"
                "5. 严禁检查、操作、手术类词汇：endoscopy、colonoscopy、CT、\n"
                "   MRI、ultrasound、biopsy、surgery、screening 等。\n\n"
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

    # 确定性过滤：剔除检查/操作类、以及不含具体疾病/症状名的空壳检索式。
    # 全被滤掉时保留原样——宁可检索质量差一点，也不要变成没得检索。
    kept = [
        q for q in queries
        if not _PROCEDURE_QUERY_RE.search(q) and not _is_generic_query(q)
    ]
    if kept:
        queries = kept

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


# 人群不符关键词。为什么会反复召回到这些文献：PubMed 的高证据过滤下，
# 「儿童腹痛」类文章标题里字面就有 abdominal pain，向量相似度天然排前，
# 于是 21 岁男性的上腹痛，top 6 里有 3 篇是儿童腹痛（实测）。
# 后果不是「引错文献」那么轻——模型会拿文献倒推病人，写出
# 「腹痛是儿童常见症状，且患者年龄符合」，把 21 岁改写成儿童。
# 这是病历事实被篡改，所以按人群做确定性剔除，不交给模型判断。
PEDIATRIC_KEYWORDS = (
    "pediatric", "paediatric", "children", "child", "neonat", "infant",
    "adolescen",
)
CRITICAL_CARE_KEYWORDS = (
    "intensive care", "critically ill", "critical care", "icu",
)


def _parse_age(age_text):
    """从「21 岁」「21」这类文本里取出年龄整数；取不到返回 None。"""
    if age_text is None:
        return None
    match = re.search(r"\d+", str(age_text))
    return int(match.group()) if match else None


def filter_population_mismatch(docs, age):
    """
    剔除研究人群与本例明显不符的文献。

    成人（>=18 岁）病例：儿科 / 新生儿类文献一律剔除——「儿童腹痛」的结论
    不能用于成人，这是硬规则，跟疾病本身像不像无关。ICU / 危重患者类文献
    同样剔除：这份表单是给普通人自查用的，住院危重病人的研究对他没有意义。

    年龄取不到时**不做任何过滤**——宁可保留几篇不对口的，也不要因为一个
    解析失败就把全部文献删光。返回空列表是允许的，调用方按「无适用文献」处理。
    """
    age_value = _parse_age(age)
    if age_value is None or age_value < 18:
        return docs
    drop = PEDIATRIC_KEYWORDS + CRITICAL_CARE_KEYWORDS
    return [
        doc for doc in docs
        if not any(k in (doc.get("title") or "").lower() for k in drop)
    ]


def generate_clinical_report(profile, context, mode="evidence"):
    """
    生成结构化的鉴别诊断 + 治疗方案报告。

    mode 来自 triage_case()，决定这一版报告以常识还是以文献为主：
      self_care —— 常见轻症，允许大量使用带标注的常识性建议
      evidence  —— 需要循证，以文献为主要依据
    """
    guidance = MODE_GUIDANCE.get(mode, MODE_GUIDANCE["evidence"])

    prompt = f"""【病例】
{profile}

【检索到的医学文献】
{context}

{guidance}

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
如果本例属于常见轻症、确实不需要任何检查，就直接说明「无需特殊检查」，
并标注「（一般建议，非文献）」——不要为了凑内容而列举不必要的检查。

【治疗方案方向】
分三部分写：药物类别、生活方式干预、随访计划。
只写方向，不写具体剂量与处方。
每条都要标注依据：来自文献的标 [文献编号]，属于常识的标「（一般建议，非文献）」。

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
    0. 红旗关键词筛查 —— 命中即直接提示立即就医，后面全部跳过
    1. 模型分流 —— 兜住关键词漏掉的急症，并决定报告以常识还是文献为主
    2. 表单字段 → 中文临床小结
    3. 生成 2~3 条英文检索式（诊断 / 治疗 / 预后），并剔除检查类检索式
    4. 逐条检索 PubMed 并按 pmid 去重
    5. 入库 + 多路召回，再按人群剔除不对口文献
    6. 生成结构化临床报告
    """
    profile = build_patient_profile(intake)
    if not profile.strip():
        return "请先填写症状与病史。"

    # 第 0 步：确定性红旗筛查。
    # 这一步不调模型、不联网，是最后一道防线——即便模型分流失灵，
    # 命中的急症表述也不会被漏掉。
    flags = detect_red_flags(profile)
    if flags:
        print(f"命中红旗警示词，直接提示就医：{flags}")
        return build_emergency_response(flags, hint_rephrase=True)

    # 第 1 步：模型分流。
    # 关键词表无法穷尽所有急症表述，这里再让模型看一遍语义；
    # 同时区分常见轻症与需要循证，决定报告该以常识还是以文献为主。
    print("正在做分诊判断...")
    triage = triage_case(profile)
    print(f"分诊结果：{triage}")
    if triage["red_flag"]:
        return build_emergency_response(["需要急诊评估的征象（由模型判断）"])

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

    # 按人群剔除不对口文献。必须在拼 context 之前做：context 里的 [文献 N]
    # 与末尾参考文献列表共用同一套编号，两边必须是同一个列表，
    # 否则正文引用和文末原文对不上。
    kept_docs = filter_population_mismatch(relevant_docs, intake.get("age"))
    if len(kept_docs) != len(relevant_docs):
        print(f"按人群剔除不对口文献 {len(relevant_docs) - len(kept_docs)} 篇，"
              f"保留 {len(kept_docs)} 篇")
    relevant_docs = kept_docs

    if relevant_docs:
        context = "\n\n---\n\n".join([
            f"[文献 {i + 1}] {doc['title']} ({doc['journal']}, {doc['year']})\n{doc['text']}"
            for i, doc in enumerate(relevant_docs)
        ])
    else:
        # 一篇都不剩时不能塞空字符串进提示词——模型会理解成「文献里没提到」，
        # 然后自由发挥。必须明确告诉它：没有可引用的文献，别编引用。
        context = (
            "（未检索到适用于本例的文献：检索到的文献其研究人群与本例不符，"
            "已全部剔除。）\n"
            "本例没有可引用的文献，请完全基于一般医学常识作答，"
            "不要输出任何 [文献编号]。若某个判断确实需要文献支持，"
            "直接说明「未检索到适用于本例的文献」。"
        )

    print(f"正在生成临床报告...（模式：{triage['mode']}）")
    report = generate_clinical_report(profile, context, mode=triage["mode"])

    # 参考文献列表由 Python 从真实检索结果生成，不经模型之手，
    # 保证标题/期刊/年份不会张冠李戴
    if not relevant_docs:
        return report
    return report + format_references(relevant_docs)
