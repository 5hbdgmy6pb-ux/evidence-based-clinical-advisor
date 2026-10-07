import streamlit as st

st.set_page_config(page_title="循证临床顾问", page_icon="🩺")
st.title("🩺 循证临床顾问")
st.caption("基于 Qwen2.5-7B + PubMed 实时检索 | 根据症状与病史做鉴别诊断，并给出循证治疗方案")


@st.cache_resource(show_spinner="正在加载医学模型与向量库（首次约 15 秒，之后瞬时）...")
def load_rag_engine():
    """
    延迟加载 RAG 引擎。

    注意：不要在文件顶部直接 `from rag_engine import ...`。
    那会在模块导入时就加载 torch / sentence_transformers / qdrant
    并读取 PubMedBERT 权重，实测让首屏渲染卡住 16.7 秒——
    这段时间页面只有骨架屏，看不到输入框，浏览器还可能因等待过久
    断开 WebSocket，显示 "Connection error"。

    放进 cache_resource 后：首屏瞬间渲染出表单，
    模型改为第一次提交时加载一次，之后走缓存。
    """
    from rag_engine import ask_health_question, clinical_consultation

    return ask_health_question, clinical_consultation


def summarize_intake(intake):
    """把表单内容整理成一段用户可读的病例摘要，作为对话里的「用户消息」显示。"""
    parts = []
    for label in ("age", "sex", "complaint", "duration", "history", "medications", "notes"):
        value = intake.get(label)
        if value:
            parts.append(f"**{label}**：{value}")
    return "\n\n".join(parts)


# 初始化对话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ========== 问诊表单 ==========
with st.form("问诊表"):
    st.markdown("#### 填写病历")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("年龄", min_value=0, max_value=120, value=30, step=1)
    with col2:
        sex = st.selectbox("性别", ["男", "女", "其他 / 不愿透露"])

    complaint = st.text_area(
        "主诉与症状（必填）",
        placeholder="例如：近两周反复上腹隐痛，餐后加重，伴反酸、嗳气",
    )
    duration = st.text_input("病程", placeholder="例如：2 周")

    col3, col4 = st.columns(2)
    with col3:
        history = st.text_area(
            "既往史",
            placeholder="例如：高血压 5 年，无糖尿病，无手术史",
        )
    with col4:
        medications = st.text_area(
            "当前用药",
            placeholder="例如：氨氯地平 5mg 每日一次",
        )

    notes = st.text_area(
        "其他补充",
        placeholder="例如：家族史、过敏史、近期的化验或影像结果",
    )

    submitted = st.form_submit_button("生成诊断与治疗方案")

# ========== 生成报告 ==========
if submitted:
    intake = {
        "年龄": f"{int(age)} 岁",
        "性别": sex,
        "主诉与症状": complaint.strip(),
        "病程": duration.strip(),
        "既往史": history.strip(),
        "当前用药": medications.strip(),
        "其他补充": notes.strip(),
    }

    if not intake["主诉与症状"]:
        st.warning("请至少填写「主诉与症状」，否则无法分析。")
    else:
        # 表单字段名是中文，而 rag_engine 期望英文键，这里做一次映射
        engine_intake = {
            "age": intake["年龄"],
            "sex": intake["性别"],
            "complaint": intake["主诉与症状"],
            "duration": intake["病程"],
            "history": intake["既往史"],
            "medications": intake["当前用药"],
            "notes": intake["其他补充"],
        }

        st.session_state.messages.append(
            {"role": "user", "content": summarize_intake(engine_intake)}
        )
        with st.chat_message("user"):
            st.markdown(summarize_intake(engine_intake))

        with st.chat_message("assistant"):
            try:
                _, clinical_consultation = load_rag_engine()
                with st.spinner("正在评估病情并检索分析（首次约 20 秒）..."):
                    report = clinical_consultation(engine_intake)
                st.markdown(report)
                st.session_state.messages.append({"role": "assistant", "content": report})
            except Exception as e:
                st.error(f"出错了：{e}")

# ========== 降级入口：只查文献，不做诊断 ==========
with st.expander("只检索文献，不做诊断（旧模式）"):
    st.caption("定向查证某个具体医学问题时使用。只复述文献内容，不给出诊断与治疗方案。")
    if prompt := st.chat_input("描述你想查证的医学问题..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                ask_health_question, _ = load_rag_engine()
                with st.spinner("正在检索 PubMed 文献并生成回答..."):
                    answer = ask_health_question(prompt)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"出错了：{e}")
