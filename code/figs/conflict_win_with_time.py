# --*-- conding:utf-8 --*--
# @Time : 2026/4/17 14:42
# @Author : Jacob Xu
# Personal email omitted from publication copy; original author attribution retained.
# @File : conflict_win_with_time.py
# @Software : PyCharm

import os
import glob
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import defaultdict
import matplotlib.patheffects as path_effects

# 如果你没有安装 adjustText，程序会正常运行并提示
try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False

# ==========================================
# 全局路径配置
# ==========================================
DATA_DIR = os.environ.get('HCC_EXPLANATION_DIR', 'outputs/model_outputs')
CSV_PATH = os.environ.get('HCC_PREDICTIONS_CSV', 'outputs/model_outputs/ground_truth_with_predictions.csv')
OUTPUT_BASE_DIR = os.environ.get('HCC_OUTPUT_DIR', 'outputs/conflict_win_with_time')

os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# ==========================================
# 核心数据提取与解析模块
# ==========================================
def extract_agent_json(text, agent_name):
    search_str = f"[{agent_name}]"
    agent_idx = text.find(search_str)
    if agent_idx == -1: return None
    start_idx = text.find('{', agent_idx)
    if start_idx == -1: return None
    stack = []
    end_idx = -1
    for i in range(start_idx, len(text)):
        if text[i] == '{': stack.append('{')
        elif text[i] == '}':
            stack.pop()
            if not stack:
                end_idx = i + 1
                break
    if end_idx != -1:
        try: return json.loads(text[start_idx:end_idx])
        except json.JSONDecodeError: return None
    return None

def parse_atomic_features(raw_text):
    """
    将原始混合特征文本解析为原子特征列表 (V3 终极增强覆盖版)
    输入: raw_text (str)
    输出: list of dicts [{'Atomic_Feature': '...', 'Value/State': '...'}]
    """
    if pd.isna(raw_text):
        return []

    text = str(raw_text).lower()
    features = []

    # --- 1. 微血管侵犯 (MVI) ---
    if re.search(r'mvi|微血管|microvascular|MVI', text):
        state = 'Negative' if re.search(r'阴|neg|无|absence|no|0级|m0', text) else 'Positive' if re.search(r'阳|pos', text) else 'Unknown'
        features.append({'Atomic_Feature': 'MVI', 'Value': state})

    # --- 2. 卫星结节 (Satellite Nodules) ---
    if re.search(r'卫星|satellite', text):
        state = 'Negative' if re.search(r'阴|neg|无|absence|no', text) else 'Positive' if re.search(r'阳|pos', text) else 'Unknown'
        features.append({'Atomic_Feature': 'Satellite_Nodule', 'Value': state})

    # --- 3. 肿瘤数目与分布 (Tumor Number & Distribution) ---
    if re.search(r'单发|孤立|单一|solitary|single', text):
        features.append({'Atomic_Feature': 'Tumor_Number', 'Value': 'Single'})
    elif re.search(r'多发|多灶|multiple|multifocal', text) or re.search(r'非单发', text):
        features.append({'Atomic_Feature': 'Tumor_Number', 'Value': 'Multiple'})

    if re.search(r'寡|oligo', text):
        features.append({'Atomic_Feature': 'Tumor_Burden', 'Value': 'Oligo-nodular'})

    if re.search(r'双叶', text):
        features.append({'Atomic_Feature': 'Tumor_Distribution', 'Value': 'Bilobar'})
    elif re.search(r'单叶|局限在.*段', text):
        features.append({'Atomic_Feature': 'Tumor_Distribution', 'Value': 'Unilobar'})

    # --- 4. 肝脏背景疾病与纤维化 (Liver Background) ---
    if re.search(r'硬化|cirrhosis', text):
        state = 'Negative' if re.search(r'无.*硬化|non-cirrhotic', text) else 'Present'
        features.append({'Atomic_Feature': 'Liver_Cirrhosis', 'Value': state})
    if re.search(r'肝炎|hepatitis|乙肝|hbv|hbsag', text):
        features.append({'Atomic_Feature': 'Hepatitis_B_Background', 'Value': 'Positive'})
    if re.search(r'门脉高压|门静脉高压|portal hypertension', text):
        features.append({'Atomic_Feature': 'Portal_Hypertension', 'Value': 'Present'})
    if re.search(r'脂肪变性|steatosis', text):
        features.append({'Atomic_Feature': 'Hepatic_Steatosis', 'Value': 'Present'})
    if re.search(r's(\d)\s*纤维化|fibrosis\s*\(?s(\d)\)?', text):
        match = re.search(r's(\d)\s*纤维化|fibrosis\s*\(?s(\d)\)?', text)
        val = match.group(1) or match.group(2)
        features.append({'Atomic_Feature': 'Liver_Fibrosis_Stage', 'Value': f'S{val}'})
    elif re.search(r'背景慢性肝病', text):
        features.append({'Atomic_Feature': 'Chronic_Liver_Disease', 'Value': 'Chronic Liver Disease'})

    if re.search(r'正常背景肝|背景肝脏质地|non-cirrhotic', text):
        state = 'Normal/Non-cirrhotic' if re.search(r'正常|non-cirrhotic', text) else 'Poor'
        features.append({'Atomic_Feature': 'Liver_Background_Texture', 'Value': state})

    # --- 5. 肿瘤分化与特殊组织学 (Histology & Differentiation) ---
    if re.search(r'分化|differentiation|differentiated|histology', text):
        if re.search(r'中至低|moderately to poorly', text):
            state = 'Moderate-Poor'
        elif re.search(r'高至中|well-differentiated to moderately', text):
            state = 'Well-Moderate'
        elif re.search(r'低|poor', text):
            state = 'Poor'
        elif re.search(r'中|moderate', text):
            state = 'Moderate'
        elif re.search(r'高|well', text):
            state = 'Well (High-grade)' if 'high-grade' in text else 'Well'
        else:
            state = 'Recorded'
        features.append({'Atomic_Feature': 'Tumor_Differentiation', 'Value': state})

    if re.search(r'混合|chcc-cca|hcc-cca', text):
        features.append({'Atomic_Feature': 'Histologic_Type', 'Value': 'cHCC-CCA (Mixed)'})
    elif re.search(r'腺瘤|adenoma', text):
        features.append({'Atomic_Feature': 'Histologic_Type', 'Value': 'Hepatocellular Adenoma'})
    elif re.search(r'纤维肉瘤|fibrosarcoma', text):
        features.append({'Atomic_Feature': 'Histologic_Type', 'Value': 'Primary Fibrosarcoma'})

    # --- 6. 肝包膜与肿瘤包膜 (Capsule Status) ---
    if re.search(r'肝.*包膜|肝被膜|肝脏包膜|liver capsul|浆膜', text):
        state = 'Positive' if re.search(r'累及|侵犯|浸润|invasion', text) else 'Unknown'
        features.append({'Atomic_Feature': 'Liver_Capsule_Invasion', 'Value': state})
    if re.search(r'被膜|capsule', text):
        state = 'Positive' if re.search(r'累及|侵犯|invasion', text) else 'Unknown'
        features.append({'Atomic_Feature': 'Liver_Capsule_Invasion', 'Value': state})

    if re.search(r'肿瘤.*包膜|假包膜|tumor capsule|胶囊', text) or (re.search(r'包膜|capsule', text) and not re.search(r'肝', text)):
        if re.search(r'无|缺乏|缺失|absent|non-encapsulated', text):
            state = 'Absent'
        elif re.search(r'不完整|不明确|未明确|未知|不明|不确定|indistinct|incomplete|unknown|未见', text):
            state = 'Incomplete/Unknown'
        elif re.search(r'假包膜|pseudocapsule', text) or '假' in text:
            state = 'Present (Pseudocapsule)'
        elif re.search(r'完整|complete|fortress', text):
            state = 'Complete'
        elif re.search(r'部分', text):
            state = 'Partial'
        else:
            state = 'Recorded'
        features.append({'Atomic_Feature': 'Tumor_Capsule_Status', 'Value': state})

    # --- 7. 肝功能储备 (Liver Function Reserve) ---
    if re.search(r'child-pugh', text):
        match = re.search(r'child-pugh\s*([a-c])', text)
        state = match.group(1).upper() if match else 'Good'
        features.append({'Atomic_Feature': 'Child_Pugh_Grade', 'Value': state})

    if re.search(r'albi', text):
        match = re.search(r'albi\s*grade\s*(\d)', text)
        state = match.group(1) if match else 'Unknown'
        features.append({'Atomic_Feature': 'ALBI_Grade', 'Value': state})

    if re.search(r'肝功能.*良好|肝功能代偿良好|极佳.*功能|优异.*肝功能|excellent liver|good liver|肝脏.*储备.*good|肝.*储备.*极佳|肝功能正常|preserved function|肝功能储备', text):
        features.append({'Atomic_Feature': 'Liver_Function_Assessment', 'Value': 'Good/Excellent'})

    # --- 8. 肿瘤标志物 (Tumor Markers) ---
    if re.search(r'afp|alpha-fetoprotein|甲胎蛋白', text):
        value_match = re.search(r'(\d+\.?\d*)\s*ng/ml', text)
        if value_match:
            # features.append({'Atomic_Feature': 'Preoperative_AFP', 'Value': float(value_match.group(1))})
            state = float(value_match.group(1))
        elif re.search(r'正常|normal', text):
            state = 'Normal'
        elif re.search(r'>400|显著升高', text):
            state = 'Significantly Elevated (>400)'
        elif re.search(r'升高|高水平|elevated', text):
            state = 'Elevated'
        elif re.search(r'低水平|low', text):
            state = 'Low'
        else:
            state = 'Recorded'
        features.append({'Atomic_Feature': 'Preoperative_AFP', 'Value': state})

    if re.search(r'ca19-9', text):
        val = re.search(r'ca19-9.*?(\d+\.?\d*)', text)
        features.append({'Atomic_Feature': 'Tumor_Marker_CA19_9', 'Value': float(val.group(1)) if val else 'Elevated'})
    if re.search(r'cea', text):
        val = re.search(r'cea.*?(\d+\.?\d*)', text)
        features.append({'Atomic_Feature': 'Tumor_Marker_CEA', 'Value': float(val.group(1)) if val else 'Elevated'})
    if re.search(r'ggt', text):
        val = re.search(r'ggt.*?(\d+\.?\d*)', text)
        features.append({'Atomic_Feature': 'Tumor_Marker_GGT', 'Value': float(val.group(1)) if val else 'Elevated'})

    # --- 9. 年龄与体能状态 (Age & Performance Status) ---
    if re.search(r'年龄|岁|年轻|高龄|老年|age|years', text):
        age_match = re.search(r'(\d+)\s*(岁|years|yrs)|age\s*(\d+)', text)
        if age_match:
            val = age_match.group(1) or age_match.group(3)
            features.append({'Atomic_Feature': 'Age', 'Value': int(val)})
        if re.search(r'年轻|young', text): features.append({'Atomic_Feature': 'Age', 'Value': 'Young'})
        elif re.search(r'高龄|老年|old', text) or re.search(r'>70', text): features.append({'Atomic_Feature': 'Age', 'Value': 'Elderly'})

    if re.search(r'体能|体力|一般状况|全身状况|生理储备|身体状况|performance status|ps|ecog|host status|患者条件', text):
        state = 'Poor' if re.search(r'差|低|poor', text) else 'Good'
        features.append({'Atomic_Feature': 'Performance_Status', 'Value': state})

    if re.search(r'年龄.*全身状态|年龄.*肝功能', text):
        features.append({'Atomic_Feature': 'Performance_Status', 'Value': 'Recorded'})

    # --- 10. 营养与血液学状态 (Nutritional & Hematological) ---
    if re.search(r'营养|nutritional|白蛋白|alb', text):
        state = 'Poor' if re.search(r'差|poor|低', text) else 'Good'
        features.append({'Atomic_Feature': 'Nutritional_Status', 'Value': state})
        alb_val = re.search(r'alb\s*(\d+\.?\d*)', text)
        if alb_val: features.append({'Atomic_Feature': 'Albumin_Value', 'Value': float(alb_val.group(1))})

    if re.search(r'贫血|anemia|hb\s*\d+', text):
        state = 'Negative' if re.search(r'无贫血|no anemia', text) else 'Present'
        features.append({'Atomic_Feature': 'Anemia', 'Value': state})
        hb_val = re.search(r'hb\s*(\d+\.?\d*)', text)
        if hb_val: features.append({'Atomic_Feature': 'Hemoglobin_Value', 'Value': float(hb_val.group(1))})

    if re.search(r'血小板|platelet', text):
        state = 'Present' if re.search(r'减少|thrombocytopenia', text) else 'Recorded'
        features.append({'Atomic_Feature': 'Platelet_Count', 'Value': state})
        plt_val = re.search(r'count\s*(\d+\.?\d*)', text)
        if plt_val: features.append({'Atomic_Feature': 'Platelet_Count', 'Value': float(plt_val.group(1))})

    # --- 11. 手术方式、切缘与治疗反应 (Surgical Method & Response) ---
    if re.search(r'微创|腹腔镜|minimally invasive|laparoscopic|minimal surgical trauma', text):
        features.append({'Atomic_Feature': 'Surgical_Method', 'Value': 'Minimally Invasive (Laparoscopic)'})
    elif re.search(r'开腹|复杂|高创伤|大范围|双叶切除|major anatomical resection', text):
        features.append({'Atomic_Feature': 'Surgical_Method', 'Value': 'Open/Major Resection'})
    elif re.search(r'解剖性切除|anatomical resection', text):
        features.append({'Atomic_Feature': 'Surgical_Method', 'Value': 'Anatomical Resection'})

    if re.search(r'切缘|margin|r0|r1|r2', text):
        if re.search(r'阴性|neg|r0|未见癌', text): features.append({'Atomic_Feature': 'Surgical_Margin', 'Value': 'Negative (R0)'})
        elif re.search(r'阳性|pos|r1|r2', text): features.append({'Atomic_Feature': 'Surgical_Margin', 'Value': 'Positive (R1/R2)'})
        elif re.search(r'unknown|未明确|未知', text): features.append({'Atomic_Feature': 'Surgical_Margin', 'Value': 'Unknown'})
    elif re.search(r'根治性切除|根治性处理', text):
        features.append({'Atomic_Feature': 'Surgical_Margin', 'Value': 'Negative (R0) - Radical Treatment'})

    if re.search(r'pcr|完全缓解', text):
        features.append({'Atomic_Feature': 'Pathological_Response', 'Value': 'Complete (pCR)'})
    elif re.search(r'活性病灶|viable tumor', text):
        features.append({'Atomic_Feature': 'Treatment_Response', 'Value': 'Residual Viable Tumor'})

    # --- 12. 肿瘤大小与阶段 (Tumor Size & Stage) ---
    if re.search(r'肿瘤|tumou?r|病灶|mass', text, re.I):
        size_matches = re.findall(r'(\d+\.?\d*)\s*(cm|厘米|mm|毫米)', text, re.I)
        if size_matches:
            vals = []
            for val, unit in size_matches:
                val = float(val)
                if unit.lower() in ['mm', '毫米']:
                    val = val / 10  # mm → cm
                vals.append(val)

            # 👉 取最大值（关键！）
            features.append({
                'Atomic_Feature': 'Tumor_Size',
                'Value': max(vals)
            })

        elif re.search(r'[>≥]\s*\d+\.?\d*\s*(cm|厘米)', text):
            val = re.search(r'[>≥]\s*(\d+\.?\d*)', text).group(1),
            features.append({'Atomic_Feature': 'Tumor_Size','Value': f'> {val}cm'})
        elif re.search(r'巨大', text):features.append({'Atomic_Feature': 'Tumor_Size', 'Value': '> 10cm'})
        elif re.search(r'较大|大', text):features.append({'Atomic_Feature': 'Tumor_Size', 'Value': '> 5cm'})
        elif '>2' in text:features.append({'Atomic_Feature': 'Tumor_Size', 'Value': '> 2cm'})

    t_stage = re.search(r'\b(p?t[1-4][a-c]?)\b', text, re.I)
    if t_stage:features.append({'Atomic_Feature': 'TNM_T_Stage', 'Value': t_stage.group(1).upper()})
    if re.search(r'\bstage\s*i\b', text, re.I):features.append({'Atomic_Feature': 'TNM_Stage', 'Value': 'Stage I'})

    # --- 13. 大血管与神经侵犯 (Macrovascular & Neural Invasion) ---
    if re.search(r'大血管|宏观|大体血管|(?<!微)血管侵犯|门静脉(?!高压)|门脉(?!高压)|静脉.*(?:癌栓|侵犯)|脉管.*癌栓|pvtt|macrovascular|macroscopic vascular|gross vascular|(?<!micro)vascular invasion', text):
        if re.search(r'无|no|未见|absence', text):
            features.append({'Atomic_Feature': 'Macroscopic_Vascular_Invasion', 'Value': 'Negative'})
        else:
            state = 'Positive'
            if re.search(r'门静脉.*分支|门脉分支|branch pvtt', text): state = 'Positive (Branch PVTT)'
            elif re.search(r'门静脉|门脉|pvtt', text): state = 'Positive (PVTT)'
            elif re.search(r'肝.*静脉', text): state = 'Positive (Hepatic Vein)'
            features.append({'Atomic_Feature': 'Macroscopic_Vascular_Invasion', 'Value': state})

    if re.search(r'胆管癌栓', text): features.append({'Atomic_Feature': 'Bile_Duct_Tumor_Thrombus', 'Value': 'Positive'})
    if re.search(r'神经|perineural', text):
        state = 'Negative' if re.search(r'无|no', text) else 'Positive'
        features.append({'Atomic_Feature': 'Perineural_Invasion', 'Value': state})

    # --- 14. 淋巴结与远处转移 (Lymph Node & Distant Metastasis) ---
    if re.search(r'淋巴结|lymph node|n0|n1', text):
        if re.search(r'pn0|阴性|无|no', text) and not re.search(r'可疑|suspect|随访', text):
            features.append({'Atomic_Feature': 'Lymph_Node_Metastasis', 'Value': 'Negative (N0)'})
        elif re.search(r'cn1|n1|转移|可疑|肿大|suspect|随访', text):
            features.append({'Atomic_Feature': 'Lymph_Node_Metastasis', 'Value': 'Positive/Suspected (N1)'})

    if re.search(r'肺|lung|骨|bone|肋骨|椎体|远处转移|肝外|大网膜|cm1|cm0|pmx|metastases|metastasis|extralepatic|extrahepatic|pulmonary', text):
        if re.search(r'无|no|未见', text) and not re.search(r'结节|可疑|怀疑|疑似', text):
            features.append({'Atomic_Feature': 'Distant_Metastasis', 'Value': 'Negative'})
        elif re.search(r'结节|可疑|待定|待查|suspect|疑似|怀疑|indeterminate', text) or 'c' in text.replace('cm', 'c'):
            loc = 'Lung' if re.search(r'肺|lung|pulmonary', text) else ('Bone' if re.search(r'骨|椎|肋骨', text) else 'Extrahepatic/Unspecified')
            features.append({'Atomic_Feature': 'Distant_Metastasis', 'Value': f'Suspected ({loc})'})
        else:
            loc = 'Lung' if re.search(r'肺|lung|pulmonary', text) else ('Omentum' if re.search(r'大网膜', text) else 'Extrahepatic/Unspecified')
            features.append({'Atomic_Feature': 'Distant_Metastasis', 'Value': f'Confirmed ({loc})'})

    # --- 15. 结节性质(Nodules) ---
    if re.search(r'包膜下.*结节|s5.*结节|s8.*结节|肝内.*结节', text) and re.search(r'可疑|不明确|需随访', text):
        features.append({'Atomic_Feature': 'Intrahepatic_Nodule', 'Value': 'Suspected/Indeterminate'})

    # --- 16. 肿瘤生长模式与边缘 (Growth Pattern & Border) ---
    if re.search(r'浸润|侵袭|infiltrative|aggressive|边缘不清|边界欠清|ill-defined', text):
        features.append({'Atomic_Feature': 'Growth_Pattern', 'Value': 'Infiltrative/Aggressive'})
        features.append({'Atomic_Feature': 'Tumor_Boundary', 'Value': 'Ill-defined'})
    if re.search(r'多结节.*生长|融合|multinodular', text):
        features.append({'Atomic_Feature': 'Growth_Pattern', 'Value': 'Multinodular'})
    if re.search(r'局限|well-defined', text):
        features.append({'Atomic_Feature': 'Tumor_Boundary', 'Value': 'Well-defined'})
    if re.search(r'限制性界面', text):
        features.append({'Atomic_Feature': 'Tumor_Boundary', 'Value': 'Restrictive Interface (Possible Capsule)'})
    if re.search(r'边界|boundary', text):
        state = 'Ill-defined' if re.search(r'欠清|ill-defined', text) else 'Unknown'
        features.append({'Atomic_Feature': 'Tumor_Boundary', 'Value': state})


    # --- 17. 影像学特殊指征 (Radiological Abnormalities) ---
    if re.search(r's[1-8].*强化灶|动脉期异常强化', text):
        features.append({'Atomic_Feature': 'Radiological_Abnormality', 'Value': 'Arterial Phase Hyperenhancement'})
    if re.search(r's[1-8]\s*nodules|s[1-8]\s*结节', text):
        features.append({'Atomic_Feature': 'Intrahepatic_Nodule', 'Value': 'Suspected (Imaging)'})

    # --- 18. 并发症与其他属性 (Comorbidities & Others) ---
    if re.search(r'ki-67|ki67', text):
        ki_match = re.search(r'(\d+)\s*%', text)
        if ki_match: features.append({'Atomic_Feature': 'Ki_67', 'Value': int(ki_match.group(1))})
        else: features.append({'Atomic_Feature': 'Ki_67', 'Value': 'Recorded'})
    if re.search(r'坏死|necrosis', text):
        state = 'Present' if re.search(r'presence|有|伴', text) else 'Unknown'
        features.append({'Atomic_Feature': 'Tumor_Necrosis', 'Value': state})
    if re.search(r'破裂|出血|rupture', text):
        features.append({'Atomic_Feature': 'Tumor_Rupture', 'Value': 'Positive/Suspected'})
    if re.search(r'膈肌|邻近器官', text):
        features.append({'Atomic_Feature': 'Adjacent_Organ_Invasion', 'Value': 'Positive (Diaphragm/Others)'})
    if re.search(r'紧邻|推压', text):
        features.append({'Atomic_Feature': 'Tumor_Location_Risk', 'Value': 'Close to Major Vessels/Organs'})
    if re.search(r'食管癌|esophageal', text):
        features.append({'Atomic_Feature': 'Comorbidity', 'Value': 'Esophageal Cancer'})


    # 如果没有匹配到任何规则，保留原值并标记为未分类
    if not features:
        features.append({'Atomic_Feature': 'Unclassified_Raw_Feature', 'Value': text})

    return features


# ==========================================
# 分析逻辑：静态权重 + 博弈胜率
# ==========================================
def method_1_semantic_weight_scoring(file_paths):
    weight_map = {"High": 3, "Medium": 2, "Low": 1}
    feature_scores = defaultdict(list)
    feature_origins = defaultdict(set)

    for path in file_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
        except: continue

        risk_data = extract_agent_json(content, "Recurrence_Risk_Agent_1")
        if not risk_data or "prognostic_factor_analysis" not in risk_data:
            continue

        analysis = risk_data["prognostic_factor_analysis"]
        all_factors = analysis.get("favorable_factors", []) + analysis.get("unfavorable_factors", [])

        for item in all_factors:
            raw_factor = item.get("factor", "").strip()
            weight_str = item.get("weight", "")

            if raw_factor and weight_str in weight_map:
                score = weight_map[weight_str]
                atomic_list = parse_atomic_features(raw_factor)

                for atomic in atomic_list:
                    atomic_name = atomic['Atomic_Feature']
                    atomic_val = atomic['Value']
                    key = (atomic_name, atomic_val)
                    feature_scores[key].append(score)
                    feature_origins[key].add(raw_factor)

    results = []
    for key, scores in feature_scores.items():
        atomic_name, atomic_val = key
        freq = len(scores)
        avg_score = np.mean(scores)
        variance = np.var(scores)
        global_importance = freq * avg_score
        results.append({
            "Atomic_Feature_Name": atomic_name,
            "Feature_Value": atomic_val,
            "Frequency": freq,
            "Mean_Weight": round(avg_score, 2),
            "Variance": round(variance, 2),
            "Global_Importance": round(global_importance, 2),
            "Original_Text_Sources": " | ".join(feature_origins[key])
        })
    return pd.DataFrame(results).sort_values(by="Global_Importance", ascending=False) if results else pd.DataFrame()

def aggregate_by_atomic_feature_name(df):
    if df.empty: return pd.DataFrame()
    grouped = df.groupby('Atomic_Feature_Name')
    results = []

    for name, group in grouped:
        freq_total = group['Frequency'].sum()
        global_imp_total = group['Global_Importance'].sum()
        mean_weight = global_imp_total / freq_total if freq_total > 0 else 0
        sum_sq = (group['Frequency'] * (group['Variance'] + group['Mean_Weight']**2)).sum()
        variance = (sum_sq / freq_total) - (mean_weight**2) if freq_total > 0 else 0

        all_sources = set()
        for sources in group['Original_Text_Sources']:
            if pd.notna(sources) and sources.strip():
                all_sources.update([s.strip() for s in sources.split(' | ')])
        merged_sources = " | ".join(sorted(all_sources))

        results.append({
            "Atomic_Feature_Name": name,
            "Frequency": freq_total,
            "Mean_Weight": round(mean_weight, 2),
            "Variance": round(variance, 2),
            "Global_Importance": round(global_imp_total, 2),
            "Original_Text_Sources": merged_sources
        })
    return pd.DataFrame(results).sort_values(by="Global_Importance", ascending=False).reset_index(drop=True)

def enhanced_rule_based_conflict_parser(risk_data):
    analysis = risk_data.get("prognostic_factor_analysis", {})
    conflict_synth = risk_data.get("core_conflict_synthesis", {})
    resolution_text = conflict_synth.get("balanced_resolution", "")

    if not resolution_text or "Explain how you weighed them" in resolution_text or "Name of factor" in str(analysis):
        return []

    fav_list = [re.sub(r'\(.*?\)|（.*?）', '', f.get("factor", "")).strip() for f in analysis.get("favorable_factors", []) if f]
    unfav_list = [re.sub(r'\(.*?\)|（.*?）', '', f.get("factor", "")).strip() for f in analysis.get("unfavorable_factors", []) if f]

    alias_dict = {
        "MVI": ["mvi", "微血管"],
        "大血管侵犯": ["大血管", "门静脉", "pvtt", "静脉癌栓"],
        "卫星结节": ["卫星"],
        "肿瘤数目": ["单发", "多发", "多灶", "多结节"],
        "肿瘤大小": ["大小", "最大径", "体积"],
        "分化程度": ["分化"],
        "AFP": ["afp", "甲胎蛋白"],
        "包膜与被膜": ["被膜", "包膜", "假包膜", "浆膜"],
        "肝功能储备": ["肝功能", "child-pugh", "albi", "储备", "全身状态", "ecog"],
        "背景肝脏": ["肝硬化", "肝炎", "hbv", "慢性肝病", "纤维化"],
        "切缘": ["切缘", "r0", "r1", "pcr"],
        "淋巴结与远处转移": ["淋巴结", "转移", "m1", "n1"]
    }

    def check_feature_mentioned(formal_name, text):
        formal_lower = formal_name.lower()
        text_lower = text.lower()
        if formal_lower in text_lower: return True
        for key, aliases in alias_dict.items():
            if key in formal_lower or any(a in formal_lower for a in aliases):
                if any(a in text_lower for a in aliases): return True
        return False

    mentioned_fav = [f for f in fav_list if check_feature_mentioned(f, resolution_text)]
    mentioned_unfav = [f for f in unfav_list if check_feature_mentioned(f, resolution_text)]

    if not mentioned_fav or not mentioned_unfav: return []

    fav_win_markers = ["倾向有利", "大幅削弱", "保护力远超", "保护屏障", "被...削弱", "抵消", "权重远高于", "重大利好", "风险缓解", "缓冲"]
    unfav_win_markers = ["倾向不利", "负面影响远超", "占据主导", "压制", "掩盖", "无法抵消", "极高风险", "放大", "主要风险驱动", "严重向极高", "根本性风险"]

    cl_text = resolution_text.replace("'", "").replace("‘", "").replace("’", "")
    fav_score = sum(1 for m in fav_win_markers if m in cl_text)
    unfav_score = sum(1 for m in unfav_win_markers if m in cl_text)

    battles = []
    if fav_score > unfav_score:
        for winner in mentioned_fav:
            for loser in mentioned_unfav: battles.append((winner, loser))
    elif unfav_score > fav_score:
        for winner in mentioned_unfav:
            for loser in mentioned_fav: battles.append((winner, loser))
    return battles

def method_2_core_conflict_win_rate(file_paths, df_agg=None):
    win_counts = defaultdict(int)
    total_battles = defaultdict(int)

    for path in file_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            risk_data = extract_agent_json(content, "Recurrence_Risk_Agent_1")
            if not risk_data: continue

            battles = enhanced_rule_based_conflict_parser(risk_data)

            for raw_winner, raw_loser in battles:
                winner_atomics = parse_atomic_features(raw_winner)
                loser_atomics = parse_atomic_features(raw_loser)

                w_names = {a['Atomic_Feature'] for a in winner_atomics if a['Atomic_Feature'] != 'Unclassified_Raw_Feature'}
                l_names = {a['Atomic_Feature'] for a in loser_atomics if a['Atomic_Feature'] != 'Unclassified_Raw_Feature'}

                for w in w_names:
                    for l in l_names:
                        if w == l: continue
                        win_counts[w] += 1
                        total_battles[w] += 1
                        total_battles[l] += 1
        except: continue

    results = []
    for feature, total in total_battles.items():
        wins = win_counts.get(feature, 0)
        losses = total - wins
        win_rate = wins / total if total > 0 else 0
        results.append({
            "Atomic_Feature_Name": feature,
            "Total_Battles": total,
            "Wins": wins,
            "Losses": losses,
            "Net_Wins": wins - losses,
            "Win_Rate": round(win_rate, 3)
        })

    df_win_rate = pd.DataFrame(results)

    if df_agg is not None and not df_win_rate.empty and not df_agg.empty:
        agg_subset = df_agg[['Atomic_Feature_Name', 'Global_Importance', 'Mean_Weight', 'Frequency']]
        merged_df = pd.merge(df_win_rate, agg_subset, on='Atomic_Feature_Name', how='left')
        merged_df = merged_df.fillna({'Global_Importance': 0, 'Mean_Weight': 0, 'Frequency': 0})
        merged_df['Dominance_Score'] = (
                merged_df['Win_Rate'] * np.log2(merged_df['Total_Battles'] + 1) * merged_df['Global_Importance']
        ).round(2)
        merged_df = merged_df.sort_values(by=["Dominance_Score", "Net_Wins"], ascending=[False, False])
        return merged_df

    elif not df_win_rate.empty:
        df_win_rate = df_win_rate.sort_values(by=["Net_Wins", "Win_Rate"], ascending=[False, False])
        return df_win_rate

    return pd.DataFrame()

# ==========================================
# 可视化模块 (移除 plt.show 防阻塞)
# ==========================================
def plot_dominance_score(df, output_name):
    if df.empty: return
    plot_df = df.head(15).copy().sort_values('Dominance_Score', ascending=True)
    plot_df['Display_Name'] = plot_df['Atomic_Feature_Name'].str.replace('_', ' ')

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))
    palette = sns.color_palette("rocket", len(plot_df))
    bars = plt.barh(plot_df['Display_Name'], plot_df['Dominance_Score'], color=palette)

    max_score = plot_df['Dominance_Score'].max()
    offset = max_score * 0.02
    for bar in bars:
        width = bar.get_width()
        plt.text(width + offset, bar.get_y() + bar.get_height()/2,
                 f'{width:.1f}', va='center', fontsize=14, color='#333333')

    plt.grid(False)
    # plt.title('Conflict Dominance Score of Prognostic Factors', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Dominance Score', fontsize=16)
    plt.xlim(0, max_score * 1.15)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(output_name, format="pdf", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def plot_win_rate_vs_battles_bubble(df, output_name):
    if df.empty or 'Global_Importance' not in df.columns: return

    plot_df = df[df['Total_Battles'] >= 2].copy().reset_index(drop=True)
    if plot_df.empty: return
    plot_df['Display_Name'] = plot_df['Atomic_Feature_Name'].str.replace('_', ' ')

    plot_df['X_plot'] = plot_df['Total_Battles'].astype(float)
    plot_df['Y_plot'] = plot_df['Win_Rate'].astype(float)

    grouped = plot_df.groupby(['Total_Battles', 'Win_Rate'])
    for (x_val, y_val), group in grouped:
        n = len(group)
        if n > 1:
            radius_x = 0.15 + (0.02 * n)
            radius_y = 0.015 + (0.002 * n)
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            plot_df.loc[group.index, 'X_plot'] = x_val + radius_x * np.cos(angles)
            plot_df.loc[group.index, 'Y_plot'] = y_val + radius_y * np.sin(angles)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 10))

    x = plot_df['X_plot']
    y = plot_df['Y_plot']
    c = plot_df['Dominance_Score']
    sizes = plot_df['Global_Importance'] * 2.5 + 40

    scatter = plt.scatter(x, y, s=sizes, c=c, cmap='viridis', alpha=0.6, edgecolors="white", linewidth=1.2, zorder=2)

    # texts = []
    # top_features = plot_df.nlargest(15, 'Dominance_Score')
    # for _, row in top_features.iterrows():
    #     t = plt.text(row['X_plot'], row['Y_plot'], row['Display_Name'],
    #                  fontsize=16, fontweight='semibold',
    #                  bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="gray", alpha=0.7), zorder=4)
    #     texts.append(t)

    # texts = []
    # top_features = plot_df.nlargest(15, 'Dominance_Score')
    # for _, row in top_features.iterrows():
    #     t = plt.text(
    #         row['X_plot']+ 0.1,
    #         row['Y_plot']+ 0.02,
    #         row['Display_Name'],
    #         fontsize=17, color='black',fontweight='semibold',
    #         bbox=dict(
    #             boxstyle="round,pad=0.25",
    #             fc="white",
    #             ec="black",
    #             linewidth=0.8,
    #             alpha=0.7
    #         ),
    #         zorder=5
    #     )
    #     t.set_path_effects([
    #         path_effects.Stroke(linewidth=2, foreground='white'),
    #         path_effects.Normal()
    #     ])
    #
    #     texts.append(t)


    texts = []
    top_features = plot_df.nlargest(15, 'Dominance_Score')

    for _, row in top_features.iterrows():
        t = plt.text(
            row['X_plot'],
            row['Y_plot'],
            row['Display_Name'],
            fontsize=20,
            # fontweight='semibold',
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.9),
            zorder=6
        )
        texts.append(t)


    if HAS_ADJUST_TEXT:
        adjust_text(
            texts,
            x=plot_df['X_plot'],
            y=plot_df['Y_plot'],
            arrowprops=dict(
                arrowstyle='-',
                color='black',
                lw=1.0,
                alpha=0.7
            ),
            expand_points=(2, 2),
            expand_text=(1.5, 1.5),
            force_text=2,
            force_points=1,
            lim=300   # 👈 很关键：增加迭代次数
        )

    cbar = plt.colorbar(scatter, pad=0.01, fraction=0.05)
    cbar.set_label('Conflict Dominance Score', rotation=90, labelpad=10, fontsize=20)
    cbar.ax.tick_params(labelsize=20)
    # plt.title('Win Rate vs Total Conflict Battles', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Frequency of Involvement in Prognostic Conflicts', fontsize=20)
    plt.ylabel('Conflict Win Rate', fontsize=20)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.grid(False)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    # plt.figtext(0.99, 0.01, '* Note: Overlapping features are dispersed to reveal data density.',
    #             ha='right', fontsize=9, color='dimgray', style='italic')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    # plt.tight_layout()
    plt.savefig(output_name, format="pdf", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def plot_dominance_wordcloud(df, output_name):
    if df.empty or 'Dominance_Score' not in df.columns: return
    plot_df = df[df['Dominance_Score'] > 0].copy()
    if plot_df.empty: return

    plot_df['Display_Name'] = plot_df['Atomic_Feature_Name'].str.replace('_', ' ')
    word_freq_dict = dict(zip(plot_df['Display_Name'], plot_df['Dominance_Score']))
    if not word_freq_dict: return

    wordcloud = WordCloud(
        width=1600, height=1000, background_color='white', colormap='inferno',
        max_words=40, contour_width=0, prefer_horizontal=0.85
    ).generate_from_frequencies(word_freq_dict)

    plt.figure(figsize=(16, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    # plt.title('Word Cloud of Features by Conflict Dominance Score', fontsize=20, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.savefig(output_name, format="pdf", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

# ==========================================
# 辅助函数：从文件路径提取 patient_sn
# ==========================================
def get_patient_sn_from_path(filepath):
    base_name = os.path.basename(filepath)
    return base_name.replace("_output.txt", "").replace(".txt", "")

# ==========================================
# 主程序：时间队列划分与批量执行
# ==========================================
if __name__ == "__main__":
    print(f"加载标签数据: {CSV_PATH}")
    time_file = pd.read_csv(CSV_PATH)
    # 确保 sn 被当作字符串匹配
    time_file['patient_sn'] = time_file['patient_sn'].astype(str)

    files = glob.glob(os.path.join(DATA_DIR, "*.txt"))
    if not files:
        print(f"错误：未在 {DATA_DIR} 找到测试文件！")
        exit()

    print(f"发现 {len(files)} 个测试文件，开始进行时间分层的人群划分...")
    sn_to_path = {get_patient_sn_from_path(path): path for path in files}

    # 按模型预测结果分组
    cohorts = {
        "Pred_6m_HighRisk": time_file[time_file['p_0_6'] == 1]['patient_sn'].tolist(),
        "Pred_6m_LowRisk": time_file[time_file['p_0_6'] == 0]['patient_sn'].tolist(),
        "Pred_24m_HighRisk": time_file[time_file['p_0_24'] == 1]['patient_sn'].tolist(),
        "Pred_24m_LowRisk": time_file[time_file['p_0_24'] == 0]['patient_sn'].tolist(),
        "Pred_24_60m_HighRisk": time_file[time_file['p_24_60'] == 1]['patient_sn'].tolist(),
        "Pred_24_60m_LowRisk": time_file[time_file['p_24_60'] == 0]['patient_sn'].tolist()
    }

    # 遍历处理各个预测分组
    for cohort_name, sn_list in cohorts.items():
        cohort_files = [sn_to_path[sn] for sn in sn_list if sn in sn_to_path]
        print(f"\n=========================================")
        print(f"开始处理分组: [{cohort_name}], 共匹配样本: {len(cohort_files)}")

        if not cohort_files:
            print(f"警告：该分组无可用文件，跳过。")
            continue

        # 第一步：获取全局聚合数据 (提供 Global Importance)
        df_m1 = method_1_semantic_weight_scoring(cohort_files)
        df_agg = aggregate_by_atomic_feature_name(df_m1)

        # 第二步：计算博弈胜率并合成统治力指数
        df_m2_combined = method_2_core_conflict_win_rate(cohort_files, df_agg=df_agg)

        if df_m2_combined.empty:
            print(f"警告：该分组未能提取到有效博弈数据，跳过保存与绘图。")
            continue

        # 动态生成输出路径
        out_csv = os.path.join(OUTPUT_BASE_DIR, f"{cohort_name}_conflict_win_analysis.csv")
        out_bar = os.path.join(OUTPUT_BASE_DIR, f"{cohort_name}_conflict_win_dominance_score.pdf")
        out_scatter = os.path.join(OUTPUT_BASE_DIR, f"{cohort_name}_conflict_win_battles_scatter.pdf")
        out_wc = os.path.join(OUTPUT_BASE_DIR, f"{cohort_name}_conflict_win_dominance_wordcloud.pdf")

        # 打印 Top 数据供即时查看
        display_cols = ["Atomic_Feature_Name", "Dominance_Score", "Win_Rate", "Total_Battles"]
        print(df_m2_combined[display_cols].head(10).to_string(index=False))

        # 保存结果
        df_m2_combined.to_csv(out_csv, index=False, encoding='utf-8-sig')

        plot_dominance_score(df_m2_combined, out_bar)
        plot_win_rate_vs_battles_bubble(df_m2_combined, out_scatter)
        plot_dominance_wordcloud(df_m2_combined, out_wc)

        print(f"[{cohort_name}] 处理完毕。数据已存入: {OUTPUT_BASE_DIR}")