import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

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
        
        try:
            df_full = pd.read_excel("clustered_scam_only.xlsx")
            df_clean = df_full[df_full['cluster'] >= 0].copy()
            name_map = dict(zip(df_profiles['cluster_id'], df_profiles['cluster_name']))
            df_clean['cluster_name'] = df_clean['cluster'].map(name_map)
        except Exception as e:
            df_clean = pd.DataFrame()
            
        return df_profiles, df_clean
    except:
        return pd.DataFrame(), pd.DataFrame()

@st.cache_resource
def load_model():
    try:
        model = joblib.load('fraud_rf_model.pkl')
        features = joblib.load('model_features.pkl')
        return model, features
    except:
        return None, None

df_profiles, df_clean = load_data()
model, model_features = load_model()

# ================= 3. 构建顶部核心指标看板 (KPIs) =================
col1, col2, col3, col4 = st.columns(4)
col1.metric(label="📥 接入涉案卷宗样本", value="1,069 宗", delta="已清洗清洗噪声")
col2.metric(label="🧬 提炼标准犯罪链路", value="12 类", delta="无监督自底向上提取")
col3.metric(label="⚙️ 决策树白盒准确率", value="91.0%", delta="警务规则可解释")
col4.metric(label="🚀 随机森林泛化性能", value="94.0%", delta="5-Fold 交叉验证加持")

st.write("") # 留白

# ================= 4. 构建四大核心功能模块 =================
tab1, tab2, tab3, tab4 = st.tabs(["📊 宏观态势地图", "🔍 诈骗模式图谱", "🧠 犯罪心理交叉透视", "🚨 智能预警沙盒 (实战)"])

# ----------------- Tab 1: 宏观态势地图 (重新排版升级版) -----------------
with tab1:
    st.markdown("### 🗺️ 诈骗宇宙空间分布拓扑图")
    st.info("💡 算法通过计算不同案件在【准备-接触-信任-操纵-榨取】全链路的相似度，自动将手法一致的案件聚集在相邻的拓扑空间中。")
    
    try:
        st.image("01_Named_UMAP_Scatter.png", use_container_width=True)
    except:
        st.caption("暂无 UMAP 图片展示")
        
    st.markdown("#### 🌊 犯罪行为全链路流转图 (MO Flow)")
    if not df_clean.empty:
        # 选择核心的4个阶段画流转图
        flow_cols = ['contact_primary', 'trust_primary', 'manipulation_primary', 'extraction_primary']
        df_flow = df_clean[flow_cols].copy()
        # 清洗前缀，让桑基图更清爽
        for col in flow_cols:
            df_flow[col] = df_flow[col].apply(lambda x: str(x).split('_')[-1] if '_' in str(x) else str(x))
            
        fig_flow = px.parallel_categories(
            df_flow, 
            dimensions=flow_cols,
            labels={'contact_primary': '接触', 'trust_primary': '信任', 'manipulation_primary': '操纵', 'extraction_primary': '榨取'},
            color_continuous_scale=px.colors.sequential.Agsunset
        )
        fig_flow.update_layout(height=450, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_flow, use_container_width=True)
        
    st.divider()
    
    # --- 第二排：占比 vs 特征重要性 ---
    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        st.markdown("#### 🍩 诈骗大类规模占比基线")
        if not df_clean.empty:
            type_counts = df_clean['cluster_name'].value_counts().reset_index()
            type_counts.columns = ['诈骗类型', '案件宗数']
            type_counts['简称'] = type_counts['诈骗类型'].apply(lambda x: str(x)[:12] + '...' if len(str(x))>12 else str(x))
            
            fig_pie = px.pie(
                type_counts, values='案件宗数', names='简称', hole=0.4,
                hover_data=['诈骗类型'], color_discrete_sequence=px.colors.qualitative.Prism
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(height=450, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)
            
    with row2_col2:
        st.markdown("#### ⚡ 判定诈骗类型的核心关键动作")
        try:
            st.image("07_Named_Feature_Importance.png", use_container_width=True)
        except:
            st.caption("暂无特征重要性图片展示")
            
    st.divider()
    
    # --- 第三排：决策树 vs 标签热力矩阵 ---
    row3_col1, row3_col2 = st.columns(2)
    with row3_col1:
        st.markdown("#### 🎯 智能化诈骗类型判别路径")
        try:
            st.image("06_Named_Decision_Tree_Clean.png", use_container_width=True)
        except:
            st.caption("暂无决策树图片展示")
            
    with row3_col2:
        st.markdown("#### 🧮 犯罪团伙作案标签热力矩阵")
        st.info("颜色越深，代表该类别犯罪团伙在行骗过程中触发对应标签动作的频率越高。")
        try:
            # 兼容 jpg 格式
            if os.path.exists("02_Named_Feature_Heatmap.jpg"):
                st.image("02_Named_Feature_Heatmap.jpg", use_container_width=True)
            else:
                st.image("02_Named_Feature_Heatmap.png", use_container_width=True)
        except:
            st.caption("暂无热力矩阵图片展示")

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

# ----------------- Tab 3: 🧠 犯罪心理交叉透视 -----------------
with tab3:
    st.markdown("### 🧠 犯罪心理画像：案件类型与受害者心理交叉分析")
    st.info("💡 洞察解析：通过交叉矩阵，揭示不同诈骗套路底层“收割”的是受害者的哪种心理弱点与行为驱动力。")

    if not df_clean.empty and 'psychological_vulnerability' in df_clean.columns and 'compliance_driver' in df_clean.columns:
        df_clean['short_name'] = df_clean['cluster_name'].apply(lambda x: str(x)[:12] + '...' if len(str(x))>12 else str(x))
        
        col_p, col_d = st.columns(2)
        
        with col_p:
            st.markdown("#### 💔 诈骗类型 VS 心理弱点")
            cross_psych = pd.crosstab(df_clean['short_name'], df_clean['psychological_vulnerability'])
            fig_psych = px.imshow(
                cross_psych,
                labels=dict(x="受害者心理弱点", y="诈骗判定类型", color="案件数量"),
                x=cross_psych.columns,
                y=cross_psych.index,
                color_continuous_scale="Blues", 
                text_auto=True,
                aspect="auto"
            )
            fig_psych.update_layout(xaxis_tickangle=-30, height=600)
            st.plotly_chart(fig_psych, use_container_width=True)

        with col_d:
            st.markdown("#### 🪝 诈骗类型 VS 顺从驱动力")
            cross_driver = pd.crosstab(df_clean['short_name'], df_clean['compliance_driver'])
            fig_driver = px.imshow(
                cross_driver,
                labels=dict(x="受害者顺从驱动力", y="诈骗判定类型", color="案件数量"),
                x=cross_driver.columns,
                y=cross_driver.index,
                color_continuous_scale="Oranges", 
                text_auto=True,
                aspect="auto"
            )
            fig_driver.update_layout(xaxis_tickangle=-30, height=600)
            st.plotly_chart(fig_driver, use_container_width=True)
            
        st.markdown("#### 🕵️ 警务实战建议 (反诈精准宣发)")
        st.caption("基于上述交叉图谱，公安机关的反诈宣传应转向“精准滴灌”。例如，深色区域显示“情感孤独”类受害者极易落入特定诈骗陷阱，反诈中心应联合社区进行针对性的情感疏导宣发；而对于受“避免惩罚（恐吓）”驱动的受害者，宣传核心应放在普及“公检法绝对不会在线办案”的常识上。")
    else:
        st.warning("⚠️ 缺失全量案卷数据或心理特征列，无法生成交叉分析矩阵。请确保 `clustered_full_data.xlsx` 在同级目录下。")

# ----------------- Tab 4: 智能预警沙盒 (实战互动) -----------------
with tab4:
    st.markdown("### 🚨 警务实战研判沙盒：全链路智能定性")
    st.write("请根据案情描述，录入该案件在关键阶段的行为特征。天网引擎将通过底层决策林瞬间给出案件定性。")
    
    if model is None or model_features is None:
        st.error("❌ 未能加载模型文件 (fraud_rf_model.pkl) 或特征文件，无法启用预测沙盒。")
    else:
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
            all_empty = all(val == "[无明显特征]" for val in user_inputs.values())
            if all_empty:
                st.warning("⚠️ 警报：案情特征提取过少！请至少录入一个明确的涉案行为特征进行锚定。")
            else:
                with st.spinner('天网引擎并发计算中...'):
                    input_vector = {feat: 0 for feat in model_features}
                    for stage_key, selected_val in user_inputs.items():
                        if selected_val != "[无明显特征]":
                            feature_name = f"{stage_key}_primary_{selected_val}"
                            if feature_name in input_vector:
                                input_vector[feature_name] = 1
                    
                    input_df = pd.DataFrame([input_vector])[model_features]
                    prediction = model.predict(input_df)[0]
                    proba = np.max(model.predict_proba(input_df)) * 100
                
                clean_pred = prediction.replace("...", "")
                matched_data = df_profiles[df_profiles['cluster_name'].str.startswith(clean_pred, na=False)]
                full_name = matched_data.iloc[0]['cluster_name'] if not matched_data.empty else prediction
                
                st.divider()
                r1, r2 = st.columns([1, 3])
                with r1:
                    st.metric("模型研判置信度", f"{proba:.1f}%")
                with r2:
                    st.success(f"### 🔴 案件定性匹配：\n**{full_name}**")
                
                if not matched_data.empty:
                    rule = matched_data.iloc[0].get('decision_rule', '无提取规则')
                    st.info(f"**📖 机器判别规则依据**：\n{rule}")
                    analysis = matched_data.iloc[0].get('mechanism_analysis', '立即开展预警止付')
                    st.error(f"**👮 案件机理与防范建议**：\n{analysis}")
