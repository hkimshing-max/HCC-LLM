# --*-- conding:utf-8 --*--
# @Author : Jacob Xu
# Personal email omitted from publication copy; original author attribution retained.
# @File : plot_conditional_blocking.py
# @Software : PyCharm
import os
import glob
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# ==========================================
# 全局路径配置
# ==========================================
DATA_DIR = os.environ.get('HCC_EXPLANATION_DIR', 'outputs/model_outputs')
OUTPUT_BASE_DIR = os.environ.get('HCC_OUTPUT_DIR', 'outputs/node_blocking')

os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

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

def extract_block_reasons(criteria_text):
    if not criteria_text: return ["Unknown_Criteria"]
    atomics = parse_atomic_features(criteria_text)
    risk_features = set()
    safe_keywords = ['negative', 'single', 'unilobar', 'normal', 'good', 'complete', 'well-defined', 'small']

    for a in atomics:
        name = a['Atomic_Feature']
        val = str(a['Value']).lower()
        if name == 'Unclassified_Raw_Feature': continue
        is_safe = any(safe_word in val for safe_word in safe_keywords)
        if 'absent' in val and 'capsule' not in name.lower():
            is_safe = True
        if not is_safe:
            risk_features.add(name)


    text_lower = criteria_text.lower()

    # 对应 Node 1B: 手术创伤/播散与生理代偿低下的复合风险
    if "双重打击" in text_lower or "double hit" in text_lower:
        risk_features.add("Compound_Perioperative_Risk")

    # 对应 Node 2C: 肿瘤负荷(种子)与恶劣肝背景(土壤)的叠加
    if "协同" in text_lower or "synergy" in text_lower:
        risk_features.add("Seed_and_Soil_Synergy")

    # 对应 Node 3B: 背景肝硬化/肝炎导致的远期新生肿瘤机制 (区域癌化)
    if "de novo" in text_lower or "新生" in text_lower:
        risk_features.add("Carcinogenic_Field_Effect")

    return list(risk_features) if risk_features else ["Unclassified_Risk_Factor"]

# ==========================================
# 分析逻辑：动态提取条件阻断率
# ==========================================
# def method_3_conditional_blocking_merged(file_paths):
#     """生成合并时间窗的分析数据"""
#     node_reach_counts = {
#         "Early Recurrence (0-24m)": 0,
#         "Late Recurrence (24-60m)": 0,
#         "Very Late Recurrence (>60m)": 0
#     }
#     node_block_reasons = {
#         "Early Recurrence (0-24m)": defaultdict(int),
#         "Late Recurrence (24-60m)": defaultdict(int),
#         "Very Late Recurrence (>60m)": defaultdict(int)
#     }
#
#     for path in file_paths:
#         try:
#             with open(path, 'r', encoding='utf-8') as f: content = f.read()
#         except: continue
#
#         final_data = extract_agent_json(content, "Final_Decision_Agent")
#         if not final_data or "sequential_recurrence_tree" not in final_data: continue
#
#         execution_log = final_data["sequential_recurrence_tree"].get("execution_log", [])
#         reached_0_24 = reached_24_60 = reached_60_plus = False
#
#         for step_info in execution_log:
#             step_name = step_info.get("step", "")
#             outcome = step_info.get("outcome", "").upper()
#             criteria = step_info.get("criteria_check", "")
#             is_stop = "STOP" in outcome or ("MET" in outcome and "NOT" not in outcome)
#
#             if "Node 1" in step_name or "Node 2" in step_name:
#                 if not reached_0_24:
#                     node_reach_counts["Early Recurrence (0-24m)"] += 1
#                     reached_0_24 = True
#                 if is_stop:
#                     reasons = extract_block_reasons(criteria)
#                     for r in reasons: node_block_reasons["Early Recurrence (0-24m)"][r] += 1
#                     break
#
#             elif "Node 3" in step_name:
#                 if not reached_24_60:
#                     node_reach_counts["Late Recurrence (24-60m)"] += 1
#                     reached_24_60 = True
#                 if is_stop:
#                     reasons = extract_block_reasons(criteria)
#                     for r in reasons: node_block_reasons["Late Recurrence (24-60m)"][r] += 1
#                     break
#
#             elif "Node 4" in step_name:
#                 if not reached_60_plus:
#                     node_reach_counts["Very Late Recurrence (>60m)"] += 1
#                     reached_60_plus = True
#                 if is_stop:
#                     reasons = extract_block_reasons(criteria)
#                     for r in reasons: node_block_reasons["Very Late Recurrence (>60m)"][r] += 1
#                     break
#
#     results = []
#     for node in ["Early Recurrence (0-24m)", "Late Recurrence (24-60m)", "Very Late Recurrence (>60m)"]:
#         reach_count = node_reach_counts[node]
#         if reach_count == 0: continue
#         for reason, block_count in node_block_reasons[node].items():
#             cond_rate = block_count / reach_count
#             results.append({
#                 "Evaluation_Window": node,
#                 "Atomic_Feature_Name": reason,
#                 "Reached_Count": reach_count,
#                 "Blocked_Count": block_count,
#                 "Conditional_Block_Rate": round(cond_rate, 3)
#             })
#
#     df = pd.DataFrame(results)
#     if not df.empty:
#         df = df.sort_values(by=["Evaluation_Window", "Conditional_Block_Rate"], ascending=[True, False])
#     return df


def method_3_conditional_blocking_merged(file_paths):
    """生成合并时间窗的分析数据 (精确支持 0-6m, 0-24m累计, 24-60m)"""
    node_reach_counts = {
        "Very Early Recurrence (0-6m)": 0,
        "Early Recurrence (0-24m)": 0,
        "Late Recurrence (24-60m)": 0
    }
    node_block_reasons = {
        "Very Early Recurrence (0-6m)": defaultdict(int),
        "Early Recurrence (0-24m)": defaultdict(int),
        "Late Recurrence (24-60m)": defaultdict(int)
    }

    for path in file_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f: content = f.read()
        except: continue

        final_data = extract_agent_json(content, "Final_Decision_Agent")
        if not final_data or "sequential_recurrence_tree" not in final_data: continue

        execution_log = final_data["sequential_recurrence_tree"].get("execution_log", [])
        reached_0_6 = reached_0_24 = reached_24_60 = False

        for step_info in execution_log:
            step_name = step_info.get("step", "")
            outcome = step_info.get("outcome", "").upper()
            criteria = step_info.get("criteria_check", "")

            # 严格判定阻断
            is_stop = "STOP" in outcome or ("MET" in outcome and "NOT" not in outcome)

            if "Node 1" in step_name:
                # 只要到达 Node 1，即代表进入了 0-6m 和 0-24m 两个评价窗
                if not reached_0_6:
                    node_reach_counts["Very Early Recurrence (0-6m)"] += 1
                    reached_0_6 = True
                if not reached_0_24:
                    node_reach_counts["Early Recurrence (0-24m)"] += 1
                    reached_0_24 = True

                if is_stop:
                    reasons = extract_block_reasons(criteria)
                    for r in reasons:
                        # Node 1 阻断，同时计入 0-6m 和 0-24m
                        node_block_reasons["Very Early Recurrence (0-6m)"][r] += 1
                        node_block_reasons["Early Recurrence (0-24m)"][r] += 1
                    break

            elif "Node 2" in step_name:
                # 只有通过了 Node 1 才会到达 Node 2，仍属于 0-24m 评价窗
                if not reached_0_24:
                    node_reach_counts["Early Recurrence (0-24m)"] += 1
                    reached_0_24 = True

                if is_stop:
                    reasons = extract_block_reasons(criteria)
                    for r in reasons:
                        # Node 2 阻断，仅计入 0-24m 累加窗
                        node_block_reasons["Early Recurrence (0-24m)"][r] += 1
                    break

            elif "Node 3" in step_name:
                if not reached_24_60:
                    node_reach_counts["Late Recurrence (24-60m)"] += 1
                    reached_24_60 = True

                if is_stop:
                    reasons = extract_block_reasons(criteria)
                    for r in reasons:
                        node_block_reasons["Late Recurrence (24-60m)"][r] += 1
                    break

    results = []
    # 遍历我们指定需要的三个时间窗
    for node in ["Very Early Recurrence (0-6m)", "Early Recurrence (0-24m)", "Late Recurrence (24-60m)"]:
        reach_count = node_reach_counts[node]
        if reach_count == 0: continue
        for reason, block_count in node_block_reasons[node].items():
            cond_rate = block_count / reach_count
            results.append({
                "Evaluation_Window": node,
                "Atomic_Feature_Name": reason,
                "Reached_Count": reach_count,
                "Blocked_Count": block_count,
                "Conditional_Block_Rate": round(cond_rate, 3)
            })

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by=["Evaluation_Window", "Conditional_Block_Rate"], ascending=[True, False])
    return df

# ==========================================
# 可视化模块：各时间窗独立绘图
# ==========================================
def plot_method3_bar_charts(csv_path, base_output_png):
    """
    智能动态可视化方案：
    为 CSV 中存在的每一个 Evaluation_Window 独立绘制并保存一张图表。
    不再进行强制并排合并，保证数据清晰和自适应。
    """
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    if df.empty: return

    # 过滤掉无法归类或杂音特征，并美化名称
    df = df[~df['Atomic_Feature_Name'].isin(["Unclassified_Risk_Factor", "Unknown_Criteria"])]
    df['Display_Name'] = df['Atomic_Feature_Name'].str.replace('_', ' ')

    sns.set_theme(style="whitegrid")

    def add_labels_to_ax(ax, data_df):
        """辅助函数：在柱状图条形内外智能添加标签"""
        ax.set_xlim(0, 1)
        for i, p in enumerate(ax.patches):
            width = p.get_width()
            if np.isnan(width): continue
            count = data_df.iloc[i]['Blocked_Count']

            text_x = width + 0.02
            ha_align = 'left'
            text_color = '#333333'

            # 智能排版防溢出
            if width > 0.85:
                text_x = width - 0.02
                ha_align = 'right'
                text_color = 'white'

            # ax.annotate(f'{width:.1%} (n={int(count)})',
            ax.annotate(f'{width:.1%}',
                        (text_x, p.get_y() + p.get_height() / 2),
                        ha=ha_align, va='center', fontsize=14, fontweight='medium', color=text_color)

    # 遍历所有存在的数据窗 (Early, Late 等) 单独出图
    windows = df['Evaluation_Window'].unique()

    for window in windows:
        # 获取当前时间窗的 Top 15 阻断因素
        df_window = df[df['Evaluation_Window'] == window].sort_values('Conditional_Block_Rate', ascending=False).head(15).reset_index(drop=True)

        if df_window.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 8))

        # 智能匹配主色调与标题
        if "Early" in window and "Very" not in window:
            palette = 'Reds_r'
            title_color = 'darkred'
        elif "Late" in window and "Very" not in window:
            palette = 'Greens_r'
            title_color = 'darkgreen'
        else:
            palette = 'Blues_r'
            title_color = 'darkblue'

        sns.barplot(x='Conditional_Block_Rate', y='Display_Name', data=df_window, ax=ax, palette=palette)

        # ax.set_title(f'Top Drivers of {window}', fontsize=15, fontweight='bold', color=title_color, pad=15)
        add_labels_to_ax(ax, df_window)
        ax.set_xlabel('Conditional Blocking Rate', fontsize=16)
        ax.set_ylabel('')
        ax.tick_params(axis='both', labelsize=16)


        # 基于传入的 base_output_png，动态生成特异性的文件名
        # 将空格、括号等替换为下划线，以适应不同的操作系统文件路径
        safe_window_name = window.replace(' ', '_').replace('(', '').replace(')', '').replace('>', 'gt_').replace('-', '_')
        output_path = base_output_png.replace('.pdf', f'_{safe_window_name}.pdf')

        plt.tight_layout()
        plt.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

        print(f"成功: 独立图表已保存至 {output_path}")

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    files = glob.glob(os.path.join(DATA_DIR, "*.txt"))
    if not files:
        print("未找到测试文件，请检查路径。")
    else:
        CSV_OUTPUT = os.path.join(OUTPUT_BASE_DIR, "method3_blocking_merged.csv")
        # 这是基础图表名，实际产出的图表会被自动加上后缀 (如 _Early_Recurrence_0_24m.pdf)
        PNG_OUTPUT_BASE = os.path.join(OUTPUT_BASE_DIR, "Fig3_Temporal_Analysis_Atomic.pdf")

        # 1. 运行数据合并提取（全集处理，不需要切分人群）
        df_m3_merged = method_3_conditional_blocking_merged(files)
        print("=== 方法三：统一原子特征的时间窗阻断率榜单 ===")
        print(df_m3_merged.to_string(index=False))
        df_m3_merged.to_csv(CSV_OUTPUT, index=False, encoding='utf-8-sig')

        # 2. 执行独立图片渲染
        if not df_m3_merged.empty:
            print("\n正在为每个时间窗独立生成阻断率条形图...")
            plot_method3_bar_charts(CSV_OUTPUT, PNG_OUTPUT_BASE)