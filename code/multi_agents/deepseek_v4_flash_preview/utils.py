# --*-- conding:utf-8 --*--
# @Time : 2025/12/30 10:21
# @Author : Jacob Xu
# Personal email omitted from publication copy; original author attribution retained.
# @File : utils.py
# @Software : PyCharm


import json
import pandas as pd
import os
from typing import List, Dict, Any
import re
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, classification_report, roc_auc_score
)

def load_patient_data(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"rror: Data file ‘{filename}’ not found. Please make sure it is in the same directory as the script.")
        return None
    except json.JSONDecodeError:
        print(f"Error：file '{filename}' is not a valid JSON format.")
        return None

def split_patient_json(full_patient_data: dict) -> dict:

    patient_info = full_patient_data.get('patient_info', {})
    pathology_data = full_patient_data.get('postoperative_pathology', {})
    pre_op_exams = full_patient_data.get('preoperative_examinations', {})
    imaging_data = pre_op_exams.get('imaging', {})
    lab_data_main = pre_op_exams.get('laboratory', {})
    lab_data_viral = pre_op_exams.get('viral_and_other', {})
    combined_lab_data = lab_data_main.copy()
    combined_lab_data['viral_and_other'] = lab_data_viral

    split_data = {
        "patient_info": patient_info,
        "imaging_data": imaging_data,
        "pathology_data": pathology_data,
        "lab_data": combined_lab_data
    }

    return split_data

# train data need pfs labeled info
def split_train_patient_json(full_patient_data: dict) -> dict:

    patient_info = full_patient_data.get('patient_info', {})
    pathology_data = full_patient_data.get('postoperative_pathology', {})
    recurrence_info = full_patient_data.get('recurrence_info', {})
    pre_op_exams = full_patient_data.get('preoperative_examinations', {})
    imaging_data = pre_op_exams.get('imaging', {})
    lab_data_main = pre_op_exams.get('laboratory', {})
    lab_data_viral = pre_op_exams.get('viral_and_other', {})
    combined_lab_data = lab_data_main.copy()
    combined_lab_data['viral_and_other'] = lab_data_viral

    split_data = {
        "patient_info": patient_info,
        "recurrence_info": recurrence_info,
        "imaging_data": imaging_data,
        "pathology_data": pathology_data,
        "lab_data": combined_lab_data
    }

    return split_data

def save_messages_to_txt(messages, filepath, file_name):
    with open(filepath, "w", encoding="utf-8") as f:
        for i, msg in enumerate(messages, 1):
            f.write(f"Paragraph {i} - 来自 [{msg.source}] ：\n")
            f.write(msg.content.strip() + "\n\n")
    folder = os.path.dirname(filepath)
    csv_path = os.path.join(folder, "finished_records.csv")
    new_row = pd.DataFrame({"file_name": [file_name]})
    if os.path.exists(csv_path):
        new_row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        new_row.to_csv(csv_path, mode="w", header=True, index=False)


# performance_evaluate
def extract_bool(json_str, key):

    pattern = rf'"{key}"\s*:\s*"?([A-Za-z0-9]+)"?'
    match = re.search(pattern, json_str, re.IGNORECASE)
    if not match:
        return None
    value = match.group(1).strip().lower()
    if value in {"true", "1", "yes"}:
        return 1
    if value in {"false", "0", "no"}:
        return 0

    return None

def extract_number(json_str, key):
    pattern = rf'"{key}"\s*:\s*"?([0-9]+(?:\.[0-9]+)?)"?'
    match = re.search(pattern, json_str)
    return float(match.group(1)) if match else None

def extract_str(json_str, key):
    # 1. 标准带引号字符串
    pattern_quoted = rf'"{key}"\s*:\s*"([^"]+)"'
    match = re.search(pattern_quoted, json_str, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # 2. 非引号字符串（如 Node 1）
    pattern_unquoted = rf'"{key}"\s*:\s*([A-Za-z0-9_\-\s]+)'
    match = re.search(pattern_unquoted, json_str, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def normalize_final_decision_node(x):
    """
    将 LLM 生成的 final_decision_node 统一规范为：
    Node 1 / Node 2 / Node 3 / Node 4
    """
    if pd.isna(x):
        return None

    x = str(x).strip().lower()

    # 中文：节点1 / 节点 1
    if re.search(r'节点\s*1', x):
        return 'Node 1'
    if re.search(r'节点\s*2', x):
        return 'Node 2'
    if re.search(r'节点\s*3', x):
        return 'Node 3'
    if re.search(r'节点\s*4', x):
        return 'Node 4'

    # 英文：node1 / node 1 / node_1
    if re.search(r'node[\s\-_]*1', x):
        return 'Node 1'
    if re.search(r'node[\s\-_]*2', x):
        return 'Node 2'
    if re.search(r'node[\s\-_]*3', x):
        return 'Node 3'
    if re.search(r'node[\s\-_]*4', x):
        return 'Node 4'

    return None

def extract_patient_info_and_prediction_text(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    result = {}

    # 1. 提取 patient_id
    pid_match = re.search(r"'patient_id'\s*:\s*'([^']+)'", content)
    if not pid_match:
        pid_match = re.search(r'"patient_id"\s*:\s*"([^"]+)"', content)
    if not pid_match:
        raise ValueError(f"【{os.path.basename(file_path)}】无法提取 patient_id")
    result["patient_id"] = pid_match.group(1)

    # 2. 提取 Final_Decision_Agent JSON
    final_pattern = r'Final_Decision_Agent[\s\S]*?(\{[\s\S]*\})\s*$'
    match = re.search(final_pattern, content, re.MULTILINE | re.DOTALL)

    if not match:
        all_jsons = re.findall(r'\{[\s\S]*?\}(?=\s*$|\s*\n)', content)
        if not all_jsons:
            raise ValueError(f"【{os.path.basename(file_path)}】未找到 Final_Decision_Agent")
        json_str = all_jsons[-1].strip().strip('`')
    else:
        json_str = match.group(1).strip().strip('`')


    result.update({
        "recurrence": extract_bool(json_str, "recurrence"),
        "death": extract_bool(json_str, "death"),
        "predicted_interval": extract_str(json_str, "predicted_interval"),
        "final_decision_node": extract_str(json_str, "final_decision_node"),
    })

    if result["recurrence"] == None:
        print(result["patient_id"])
        print(json_str)


    risk_score_for_cindex = extract_number(json_str, "risk_score_for_cindex")
    predicted_pfs_months_estimate = extract_number(json_str, "predicted_pfs_months_estimate")


    result["risk_score_for_cindex"] = risk_score_for_cindex
    result["predicted_pfs_months"] = predicted_pfs_months_estimate

    return result

def evaluate_binary_classification(df, y_true_col,y_pred_col,verbose=True):
    # === 1. 数据预处理 ===
    df_clean = df[[y_true_col, y_pred_col]].dropna().copy()

    df_clean[y_true_col] = df_clean[y_true_col].astype(int)
    df_clean[y_pred_col] = df_clean[y_pred_col].astype(int)

    print(df_clean)

    y_true = df_clean[y_true_col]
    y_pred = df_clean[y_pred_col]

    # === 2. 混淆矩阵 ===
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # === 3. sklearn 指标 ===
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # === 4. 医学指标手动计算 ===
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = recall

    # === 5. AUC 计算（0/1 预测） ===
    # 若将来有预测概率，可在这里直接替换 y_pred 为 y_prob
    auc = roc_auc_score(y_true, y_pred)

    # === 6. 结果整理为字典 ===
    metrics_dict = {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Accuracy": accuracy,
        "Precision_PPV": ppv,
        "Sensitivity_Recall": sensitivity,
        "Specificity": specificity,
        "NPV": npv,
        "F1": f1,
        "AUC": auc
    }

    # === 7. 打印结果 ===
    if verbose:
        print("=== Confusion Matrix (混淆矩阵) ===")
        print(cm)
        print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

        print("\n=== Classification Metrics (分类指标) ===")
        print(f"Accuracy (准确率):          {accuracy:.4f}")
        print(f"Precision (精确率/PPV):     {ppv:.4f}")
        print(f"Sensitivity (敏感度/召回率): {sensitivity:.4f}")
        print(f"Specificity (特异度):        {specificity:.4f}")
        print(f"NPV (阴性预测值):            {npv:.4f}")
        print(f"F1-score:                    {f1:.4f}")
        print(f"AUC:                         {auc:.4f}")

        print("\n=== Classification Report ===")
        print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    return metrics_dict, cm
