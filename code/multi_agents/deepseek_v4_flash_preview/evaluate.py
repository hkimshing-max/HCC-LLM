# --*-- conding:utf-8 --*--
# @Author : Jacob Xu
# Personal email omitted from publication copy; original author attribution retained.
# @File : evaluate.py
# @Software : PyCharm

import os
import pandas as pd
from typing import List, Dict, Any
import json
import re
from utils import extract_patient_info_and_prediction_text,evaluate_binary_classification, normalize_final_decision_node
import numpy as np
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,recall_score, f1_score, classification_report, roc_auc_score)
from lifelines.utils import concordance_index


def extract_patient_info(file_path: str):
    # 解析文件夹下的文件
    txt_files = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.lower().endswith('.txt')]
    print(f"共发现 {len(txt_files)} 个 txt 文件，开始批量解析...")

    results: List[Dict[str, Any]] = []
    for i, fp in enumerate(txt_files, 1):
        try:
            info = extract_patient_info_and_prediction_text(fp)
            results.append(info)
        except Exception as e:
            print(f"   × 失败: {e}")
            results.append({
                "patient_id": os.path.basename(fp).split("_")[0],
                "error": str(e)
            })

    df = pd.DataFrame(results)
    return df

def merge_with_ground_truth(pred_df: pd.DataFrame, gt_filepath: str) -> pd.DataFrame:
    pfs_info = pd.read_csv(gt_filepath)
    pred_df['patient_id'] = pred_df['patient_id'].astype(str)
    pfs_info['patient_sn'] = pfs_info['patient_sn'].astype(str)

    # 0-24 months ground truth logic
    cond_event_24 = (pfs_info['Target_0_to_6m'] == 1) | (pfs_info['Target_6_to_24m'] == 1)
    cond_safe_24 = (pfs_info['Target_0_to_6m'] == 0) & (pfs_info['Target_6_to_24m'] == 0)
    pfs_info['Target_0_to_24m'] = np.select(
        [cond_event_24, cond_safe_24],
        [1, 0],
        default=np.nan
    )

    # 0-60 months ground truth logic (NEW)
    cond_event_60 = (pfs_info['Target_0_to_6m'] == 1) | (pfs_info['Target_6_to_24m'] == 1) | (pfs_info['Target_24_to_60m'] == 1)
    cond_safe_60 = (pfs_info['Target_0_to_6m'] == 0) & (pfs_info['Target_6_to_24m'] == 0) & (pfs_info['Target_24_to_60m'] == 0)
    pfs_info['Target_0_to_60m'] = np.select(
        [cond_event_60, cond_safe_60],
        [1, 0],
        default=np.nan
    )

    merged_df = pfs_info.merge(
        pred_df,
        how='inner',
        left_on='patient_sn',
        right_on='patient_id',
        suffixes=('', '_pred')
    )

    merged_df = merged_df.drop(columns=['patient_id'])

    # Assuming normalize_final_decision_node is defined elsewhere
    merged_df['final_decision_node_norm'] = merged_df['final_decision_node'].apply(normalize_final_decision_node)

    merged_df['p_0_6'] = (merged_df['final_decision_node'] == 'Node 1').astype(int)
    merged_df['p_6_24'] = (merged_df['final_decision_node'] == 'Node 2').astype(int)
    merged_df['p_24_60'] = (merged_df['final_decision_node'] == 'Node 3').astype(int)
    merged_df['p_no'] = (merged_df['final_decision_node'] == 'Node 4').astype(int)

    # Cumulative prediction flags
    merged_df['p_0_24'] = np.where((merged_df['p_0_6'] == 1) | (merged_df['p_6_24'] == 1), 1, 0)
    merged_df['p_0_60'] = np.where((merged_df['p_0_6'] == 1) | (merged_df['p_6_24'] == 1) | (merged_df['p_24_60'] == 1), 1, 0)

    return merged_df


def evaluate_performance(eva_df: pd.DataFrame, threshold: float = 0.5):
    results = {}
    time_windows = {
        "0_6m": ("Target_0_to_6m", "p_0_6"),
        "6_24m": ("Target_6_to_24m", "p_6_24"),
        "0_24m": ("Target_0_to_24m", "p_0_24"),
        "24_60m": ("Target_24_to_60m", "p_24_60"),
        "0_60m": ("Target_0_to_60m", "p_0_60"), # NEW
    }

    for window, (target_col, prob_col) in time_windows.items():
        # Ensure the column exists to avoid KeyError if ground truth wasn't generated properly
        if target_col not in eva_df.columns or prob_col not in eva_df.columns:
            results[window] = None
            continue

        df_sub = eva_df[[target_col, prob_col]].dropna()
        if df_sub.empty:
            results[window] = None
            continue

        y_true = df_sub[target_col].astype(int)
        y_prob = df_sub[prob_col]
        y_pred = (y_prob >= threshold).astype(int)

        # Calculate confusion matrix quadrants
        # Note: ravel() handles the case where only one class is present in y_true/y_pred,
        # but specifying labels=[0,1] ensures it always returns a 2x2 matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        results[window] = {
            "n_samples": len(df_sub),
            "accuracy": accuracy_score(y_true, y_pred),
            "ppv_precision": precision_score(y_true, y_pred, zero_division=0),
            "npv": tn / (tn + fn) if (tn + fn) > 0 else 0.0,
            "sensitivity": recall_score(y_true, y_pred, zero_division=0),
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_prob) if y_true.nunique() > 1 else np.nan
        }

    # C-index Calculation
    required_cindex_cols = ["new_pfs_months", "recurrence", "risk_score_for_cindex"]
    if all(col in eva_df.columns for col in required_cindex_cols):
        cindex_df = eva_df[required_cindex_cols].dropna()

        if not cindex_df.empty:
            c_index = concordance_index(
                event_times=cindex_df["new_pfs_months"],
                predicted_scores=-cindex_df["risk_score_for_cindex"],
                event_observed=cindex_df["recurrence"]
            )
        else:
            c_index = np.nan

        results["c_index"] = {
            "n_samples": len(cindex_df) if not cindex_df.empty else 0,
            "c_index": c_index
        }
    else:
        results["c_index"] = None

    return results

def convert_numpy(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError


def train_main():

# Configure input/output locations using the HCC_* environment variables.
    # train_result_path
    result_folder_path = os.environ.get('HCC_RESULT_DIR', 'outputs/development_outputs')
    mac_result = extract_patient_info(result_folder_path)
    pfs_info_path = os.environ.get('HCC_GROUND_TRUTH', 'data/development_outcomes.csv')
    ground_truth_df = merge_with_ground_truth(mac_result, pfs_info_path)
    ground_truth_output_path = result_folder_path + "/ground_truth_with_predictions.csv"
    ground_truth_df.to_csv(ground_truth_output_path,index=False, encoding='utf-8-sig')
    print("解析完成，已保存至：", ground_truth_output_path)
    performance_results = evaluate_performance(ground_truth_df, threshold=0.5)
    print("性能评估结果：")
    print(json.dumps(performance_results, indent=4, ensure_ascii=False))
    performance_results_path = result_folder_path + "/performance_results.json"
    with open(performance_results_path, "w") as f:
        json.dump(performance_results, f, indent=4, default=convert_numpy)

def test_main():

    result_folder_path = os.environ.get('HCC_RESULT_DIR', 'outputs/model_outputs')
# Configure input/output locations using the HCC_* environment variables.
# Configure input/output locations using the HCC_* environment variables.
    mac_result = extract_patient_info(result_folder_path)
# Configure input/output locations using the HCC_* environment variables.
    pfs_info_path = os.environ.get('HCC_GROUND_TRUTH', 'data/outcomes.csv')
    ground_truth_df = merge_with_ground_truth(mac_result, pfs_info_path)
    ground_truth_output_path = result_folder_path + "/ground_truth_with_predictions.csv"
    ground_truth_df.to_csv(ground_truth_output_path,index=False, encoding='utf-8-sig')
    print("解析完成，已保存至：", ground_truth_output_path)

    print(ground_truth_df["Target_0_to_60m"].value_counts(), ground_truth_df["p_0_60"].value_counts(),ground_truth_df["final_decision_node"].value_counts())

    performance_results = evaluate_performance(ground_truth_df, threshold=0.5)
    performance_results_path = result_folder_path + "/performance_results.json"
    with open(performance_results_path, "w") as f:
        json.dump(performance_results, f, indent=4, default=convert_numpy)

    print(json.dumps(performance_results, indent=4, ensure_ascii=False))
    # # test_result_path
# Configure input/output locations using the HCC_* environment variables.


if __name__ == '__main__':

    # train_main()

    test_main()


