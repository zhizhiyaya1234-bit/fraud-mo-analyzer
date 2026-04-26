import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ================= 1. 页面全局配置 =================
st.set_page_config(page_title="诈骗行为链路自动化研判系统", layout="wide", page_icon="🛡️")

# 自定义 CSS 提升页面质感
st.markdown("""
    <style>
    .main-header {font-size: 34px !important; font-weight: bold; color: #1E3A8A; margin-bottom: 0;}
    .sub-header {font-size: 18px !important; color: #4B5563; margin-top: 5px;}
    .stMetric {background-color: #F3F4F6; padding: 15px; border-radius: 10px;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🛡️ 诈骗行为链路 (MO) 自动化研判与预警系统</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">基于大语言模型提取与 HDBSCAN 密度聚类的底层犯罪模式发现引擎</p>', unsafe_allow_html=True)
st.divider()

# ================= 2. 加载数据与模型 (缓存加速) =================
@st.cache_data
def load_data():
    try:
        df_profiles = pd.read_excel("cluster_profiles_named.xlsx")
        return df_profiles
    except:
        return pd.DataFrame()

@st.cache_resource
def load_model():
    try:
        model = joblib.load('fraud_rf_model.pkl')
        features = joblib.load('model_features.pkl')
        return model, features
    except:
        return None, None

df_profiles = load_data()
model, model_features = load_model()

# ================= 3. 构建顶部核心指标看板 (KPIs) =================
col1, col2, col3, col4 = st.columns(4)
col1.metric(label="📥 接入涉案卷宗样本", value="1,069 宗", delta="已清洗清洗噪声")
col2.metric(label="🧬 提炼标准犯罪链路", value="12 类", delta="无监督自底向上提取")
col3.metric(label="⚙️ 决策树白盒准确率", value="91.0%", delta="警务规则可解释")
col4.metric(label="🚀 随机森林极限准确率", value="96.0%", delta="AI预测引擎加持")

st.write("") # 留白

# ================= 4. 构建三大核心功能模块 (多标签页) =================
tab1, tab2, tab3 = st.tabs(["📊 宏观态势地图", "🔍 诈骗模式图谱", "🚨 智能预警沙盒 (实战)"])

# ----------------- Tab 1: 宏观态势地图 -----------------
with tab1:
    st.markdown("### 🗺️ 诈骗宇宙空间分布拓扑图")
    st.info("💡 算法通过计算不同案件在【准备-接触-信任-操纵-榨取】全链路的相似度，自动将手法一致的案件聚集在相邻的拓扑空间中。")
    
    try:
        st.image("01_Named_UMAP_Scatter.png", use_container_width=True)
    except:
        st.warning("未检测到 01_Named_UMAP_Scatter.png 图片。")
        
    st.divider()
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.markdown("#### 🎯 智能化诈骗类型判别路径")
        try:
            st.image("06_Named_Decision_Tree_Clean.png", use_container_width=True)
        except:
            st.caption("暂无决策树图片展示")
    with col_img2:
        st.markdown("#### ⚡ 判定诈骗类型的核心关键动作")
        try:
            st.image("07_Named_Feature_Importance.png", use_container_width=True)
        except:
            st.caption("暂无特征重要性图片展示")

# ----------------- Tab 2: 诈骗模式图谱 (单类下钻) -----------------
with tab2:
    st.markdown("### 🕵️‍♂️ 犯罪团伙标准作业程序 (SOP) 拆解")
    if not df_profiles.empty:
        fraud_types = df_profiles['cluster_name'].dropna().unique().tolist()
        selected_type = st.selectbox("👉 请选择要下钻研判的诈骗类型：", fraud_types)
        
        if selected_type:
            current_data = df_profiles[df_profiles['cluster_name'] == selected_type].iloc[0]
            
            c1, c2 = st.columns([2, 1])
            with c1:
                st.success(f"**黑产/警方俗称**：{current_data.get('police_jargon', '无')}")
                st.error(f"**受害者心理弱点利用**：{current_data.get('top_psychological_vulnerability', '未知')}")
                st.warning(f"**核心致案机理**：{current_data.get('mechanism_analysis', '无分析')}")
                
                st.markdown("#### 📋 机器提取的判别规则")
                st.code(current_data.get('decision_rule', '无提取规则'), language="sql")
            
            with c2:
                st.markdown("#### ⛓️ 典型作案流转链路")
                script = str(current_data.get('canonical_script', ''))
                if script and script != 'nan':
                    steps = script.split(" → ")
                    for step in steps:
                        st.markdown(f"⬇️ `{step}`")
            
            with st.expander("📂 调阅底层支撑卷宗 (代表性新闻摘要)"):
                summaries = str(current_data.get('sample_summaries', ''))
                if summaries and summaries != "nan":
                    summary_list = summaries.split(" /// ")
                    for i, text in enumerate(summary_list):
                        if text.strip():
                            st.write(f"**卷宗 {i+1}**: {text}")
    else:
        st.error("未能加载 cluster_profiles_named.xlsx 文件，请检查路径。")

# ----------------- Tab 3: 智能预警沙盒 (实战互动) -----------------
with tab3:
    st.markdown("### 🚨 警务实战研判沙盒：全链路智能定性")
    st.write("请根据案情描述，录入该案件在关键阶段的行为特征。天网引擎将通过底层决策林瞬间给出案件定性。")
    
    if model is None or model_features is None:
        st.error("❌ 未能加载模型文件 (fraud_rf_model.pkl) 或特征文件，无法启用预测沙盒。")
    else:
        # 国标字典：包含全部可能的特征，增强系统的实战鲁棒性
        LABEL_DEFS = {
            "prep": ["PREP1_人设身份伪造", "PREP2_平台网站搭建", "PREP3_数据名单获取", "PREP4_无明显前期准备"],
            "contact": ["CON1_盲发广撒触达", "CON2_社交平台搭讪", "CON3_需求场景切入", "CON4_冒名定向联络", "CON5_线下物理接触", "CON6_受害者主动上门"],
            "trust": ["TRU1_公权身份伪装", "TRU2_机构品牌冒用", "TRU3_熟人关系利用", "TRU4_专业人设包装", "TRU5_群体氛围伪造", "TRU6_小额返利验证", "TRU7_伪造凭证文件"],
            "manipulation": ["MAN1_恐吓威胁施压", "MAN2_高收益利诱", "MAN3_情感绑架操控", "MAN4_制造紧急时限", "MAN5_隔离保密要求", "MAN6_沉没成本追加"],
            "operation": ["OPR1_下载安装应用", "OPR2_共享屏幕远程控制", "OPR3_点击链接填写信息", "OPR4_注册账户加入群组", "OPR5_执行刷单任务", "OPR6_上传证件人脸识别"],
            "extraction": ["EXT1_银行转账", "EXT2_第三方数字支付", "EXT3_加密货币转移", "EXT4_礼品卡充值卡", "EXT5_线下现金交割", "EXT6_账户权限接管"],
            "aftermath": ["AFT1_立即失联消失", "AFT2_设障拖延拒付", "AFT3_编造新由追骗", "AFT4_转化身份复害", "AFT5_转为勒索威胁", "AFT6_发展为工具人"]
        }
        
        stage_names = {
            "prep": "1. 准备阶段", "contact": "2. 接触阶段", "trust": "3. 信任阶段", 
            "manipulation": "4. 操纵阶段", "operation": "5. 操作阶段", 
            "extraction": "6. 榨取阶段", "aftermath": "7. 善后阶段"
        }

        with st.form("prediction_sandbox"):
            st.markdown("#### 🕵️ 录入案件链路行为特征")
            row1, row2 = st.columns(4), st.columns(3)
            user_inputs = {}
            for i, (key, label) in enumerate(stage_names.items()):
                col = row1[i] if i < 4 else row2[i-4]
                with col:
                    options = ["[无明显特征]"] + LABEL_DEFS[key]
                    user_inputs[key] = st.selectbox(label, options)

            submitted = st.form_submit_button("⚡ 立即启动 AI 智能研判引擎", use_container_width=True)
            
        if submitted:
            # 防呆机制：如果全空则拦截
            all_empty = all(val == "[无明显特征]" for val in user_inputs.values())
            if all_empty:
                st.warning("⚠️ 警报：案情特征提取过少！请至少录入一个明确的涉案行为特征进行锚定。")
            else:
                with st.spinner('天网引擎并发计算中...'):
                    # 1. 构造模型特征向量
                    input_vector = {feat: 0 for feat in model_features}
                    for stage_key, selected_val in user_inputs.items():
                        if selected_val != "[无明显特征]":
                            feature_name = f"{stage_key}_primary_{selected_val}"
                            if feature_name in input_vector:
                                input_vector[feature_name] = 1
                    
                    # 2. 进行预测
                    input_df = pd.DataFrame([input_vector])[model_features]
                    prediction = model.predict(input_df)[0]
                    proba = np.max(model.predict_proba(input_df)) * 100
                
                # 3. 结果还原机制：消除预测结果中的 "..."，精准匹配原始数据表
                clean_pred = prediction.replace("...", "")
                matched_data = df_profiles[df_profiles['cluster_name'].str.startswith(clean_pred, na=False)]
                full_name = matched_data.iloc[0]['cluster_name'] if not matched_data.empty else prediction
                
                # 4. 展示炫酷结果
                st.divider()
                r1, r2 = st.columns([1, 3])
                with r1:
                    st.metric("模型研判置信度", f"{proba:.1f}%")
                with r2:
                    st.success(f"### 🔴 案件定性匹配：\n**{full_name}**")
                
                # 5. 联动展示业务建议
                if not matched_data.empty:
                    rule = matched_data.iloc[0].get('decision_rule', '无提取规则')
                    st.info(f"**📖 机器判别规则依据**：\n{rule}")
                    analysis = matched_data.iloc[0].get('mechanism_analysis', '立即开展预警止付')
                    st.error(f"**👮 案件机理与防范建议**：\n{analysis}")
