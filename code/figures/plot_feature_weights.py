# --*-- conding:utf-8 --*--
# @Author : Jacob Xu
# Personal email omitted from publication copy; original author attribution retained.
# @File : plot_feature_weights.py
# @Software : PyCharm

import os
import glob
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from wordcloud import WordCloud

# ==========================================
# 全局路径与配置
# ==========================================
DATA_DIR = os.environ.get('HCC_EXPLANATION_DIR', 'outputs/model_outputs')
CSV_PATH = os.environ.get('HCC_PREDICTIONS_CSV', 'outputs/model_outputs/ground_truth_with_predictions.csv')
OUTPUT_BASE_DIR = os.environ.get('HCC_OUTPUT_DIR', 'outputs/weight_score_with_time')

# 确保输出目录存在
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# ==========================================
# 核心数据提取函数
# ==========================================
def extract_agent_json(text, agent_name):
    """鲁棒的 JSON 提取器"""
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
# 打分与聚合函数
# ==========================================
def method_1_semantic_weight_scoring(file_paths):
    weight_map = {"High": 3, "Medium": 2, "Low": 1}
    feature_scores = defaultdict(list)
    feature_origins = defaultdict(set)

    for path in file_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"读取文件失败: {path}, 错误: {e}")
            continue

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
    if df.empty:
        return pd.DataFrame()

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

# ==========================================
# 可视化函数
# ==========================================
def plot_top_features(df_agg, output_name):
    if df_agg is None or df_agg.empty:
        print(f"数据为空，跳过绘制: {output_name}")
        return
    df_filtered = df_agg[df_agg['Atomic_Feature_Name'] != 'Unclassified_Raw_Feature'].copy()
    if df_filtered.empty: return
    plot_df = df_filtered.sort_values('Global_Importance', ascending=False).head(15).sort_values('Mean_Weight', ascending=True)
    plot_df['Display_Name'] = plot_df['Atomic_Feature_Name'].str.replace('_', ' ')

    # sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))
    palette = sns.color_palette("viridis", len(plot_df))
    bars = plt.barh(plot_df['Display_Name'], plot_df['Mean_Weight'], color=palette)

    for i, bar in enumerate(bars):
        width = bar.get_width()
        freq = plot_df.iloc[i]['Frequency']
        # plt.text(width + 0.05, bar.get_y() + bar.get_height()/2,
        #          f'Score:{width:.2f} (n={int(freq)})',
        #          va='center', fontsize=16, color='#333333')
        plt.text(width + 0.05, bar.get_y() + bar.get_height()/2,
                 f'{width:.2f} ',
                 va='center', fontsize=14, color='#333333')

    # plt.title('Explicit Weight of Prognostic Factors', fontsize=16, fontweight='bold', pad=20)
    # plt.xlabel('Mean Assigned Weight Score (1: Low, 2: Medium, 3: High)', fontsize=16)
    plt.xlabel('Mean Assigned Weight Score', fontsize=16)
    # plt.ylabel('Clinical & Pathological Features', fontsize=16)
    plt.xlim(0, max(3.8, plot_df['Mean_Weight'].max() + 0.8))
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(output_name, format="pdf", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def plot_top_global_importance(df_agg, output_name):
    if df_agg is None or df_agg.empty:
        print(f"数据为空，跳过绘制: {output_name}")
        return
    df_filtered = df_agg[df_agg['Atomic_Feature_Name'] != 'Unclassified_Raw_Feature'].copy()
    if df_filtered.empty: return
    plot_df = df_filtered.sort_values('Global_Importance', ascending=False).head(15).sort_values('Global_Importance', ascending=True)
    plot_df['Display_Name'] = plot_df['Atomic_Feature_Name'].str.replace('_', ' ')

    # sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))
    palette = sns.color_palette("magma", len(plot_df))
    bars = plt.barh(plot_df['Display_Name'], plot_df['Global_Importance'], color=palette)

    max_imp = plot_df['Global_Importance'].max()
    offset = max_imp * 0.015

    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + offset, bar.get_y() + bar.get_height()/2,
                 f'{width:.1f}', va='center', fontsize=14, color='#333333')

    # plt.title('Global Importance of Prognostic Factors', fontsize=16, fontweight='bold', pad=20)
    # plt.xlabel('Global Importance Score (Frequency × Mean Weight)', fontsize=16)
    plt.xlabel('Global Importance Score', fontsize=16)
    # plt.ylabel('Clinical/Pathological Features', fontsize=16)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.xlim(0, max_imp * 1.15)
    plt.tight_layout()
    plt.savefig(output_name, format="pdf", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def plot_feature_wordcloud(df_agg, output_name):
    if df_agg is None or df_agg.empty:
        print(f"数据为空，跳过绘制词云: {output_name}")
        return
    df_filtered = df_agg[df_agg['Atomic_Feature_Name'] != 'Unclassified_Raw_Feature'].copy()
    if df_filtered.empty: return
    df_filtered['Display_Name'] = df_filtered['Atomic_Feature_Name'].str.replace('_', ' ')
    word_freq_dict = dict(zip(df_filtered['Display_Name'], df_filtered['Global_Importance']))

    if not word_freq_dict: return

    wordcloud = WordCloud(
        width=1600, height=1000, background_color='white', colormap='ocean',
        max_words=50, contour_width=0, prefer_horizontal=0.8
    ).generate_from_frequencies(word_freq_dict)

    plt.figure(figsize=(16, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_name, format="pdf", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

# ==========================================
# 辅助函数与主执行流程
# ==========================================
def get_patient_sn_from_path(filepath):
    """从路径中提取 patient_sn"""
    base_name = os.path.basename(filepath)
    return base_name.replace("_output.txt", "").replace(".txt", "")

if __name__ == "__main__":
    # 1. 加载包含预测结果的 CSV
    print(f"加载标签数据: {CSV_PATH}")
    time_file = pd.read_csv(CSV_PATH)
    time_file = time_file[["patient_sn", "p_0_6", "p_6_24", "p_24_60", "p_no", "p_0_24", "p_0_60",
                           "Target_0_to_6m", "Target_6_to_24m", "Target_24_to_60m", "Target_0_to_24m", "Target_0_to_60m"]]
    # 确保 patient_sn 为字符串以便精确匹配
    time_file['patient_sn'] = time_file['patient_sn'].astype(str)

    # 2. 匹配本地文件
    files = glob.glob(os.path.join(DATA_DIR, "*.txt"))
    if not files:
        print(f"错误：未在 {DATA_DIR} 找到测试文件！")
        exit()

    print(f"发现 {len(files)} 个测试文件，开始进行预测人群划分...")
    sn_to_path = {get_patient_sn_from_path(path): path for path in files}

    # 3. 按模型预测结果分组 (预测2年内复发/不复发，预测5年内复发/不复发)
    cohorts = {
        "Pred_6m_HighRisk": time_file[time_file['p_0_6'] == 1]['patient_sn'].tolist(),
        "Pred_6m_LowRisk": time_file[time_file['p_0_6'] == 0]['patient_sn'].tolist(),
        "Pred_24m_HighRisk": time_file[time_file['p_0_24'] == 1]['patient_sn'].tolist(),
        "Pred_24m_LowRisk": time_file[time_file['p_0_24'] == 0]['patient_sn'].tolist(),
        "Pred_24_60m_HighRisk": time_file[time_file['p_24_60'] == 1]['patient_sn'].tolist(),
        "Pred_24_60m_LowRisk": time_file[time_file['p_24_60'] == 0]['patient_sn'].tolist()
    }

    # 4. 遍历处理每个分组
    for cohort_name, sn_list in cohorts.items():
        cohort_files = [sn_to_path[sn] for sn in sn_list if sn in sn_to_path]
        print(f"\n=========================================")
        print(f"开始处理分组: [{cohort_name}], 共匹配样本: {len(cohort_files)}")

        if not cohort_files:
            print(f"警告：该分组无可用文件，跳过。")
            continue

        # 执行提取与打分
        df_m1 = method_1_semantic_weight_scoring(cohort_files)
        df_agg = aggregate_by_atomic_feature_name(df_m1)

        if df_m1.empty or df_agg.empty:
            print(f"警告：该分组未能提取到有效特征得分，跳过保存与绘图。")
            continue

        # 保存 CSV 文件
        csv_m1_path = os.path.join(OUTPUT_BASE_DIR, f"{cohort_name}_Global_Importance.csv")
        csv_agg_path = os.path.join(OUTPUT_BASE_DIR, f"{cohort_name}_Global_Importance_agg.csv")
        df_m1.to_csv(csv_m1_path, index=False, encoding='utf-8-sig')
        df_agg.to_csv(csv_agg_path, index=False, encoding='utf-8-sig')

        # 绘制并保存可视化图表
        plot_top_features(df_agg, os.path.join(OUTPUT_BASE_DIR, f"{cohort_name}_weight_rank.pdf"))
        plot_top_global_importance(df_agg, os.path.join(OUTPUT_BASE_DIR, f"{cohort_name}_importance_rank.pdf"))
        plot_feature_wordcloud(df_agg, os.path.join(OUTPUT_BASE_DIR, f"{cohort_name}_WordCloud.pdf"))

        print(f"[{cohort_name}] 处理完成。结果已输出至: {OUTPUT_BASE_DIR}")