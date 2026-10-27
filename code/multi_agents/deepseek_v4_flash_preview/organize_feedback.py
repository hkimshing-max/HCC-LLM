# --*-- conding:utf-8 --*--
# @Author : Jacob Xu
# Personal email omitted from publication copy; original author attribution retained.
# @File : organize_feedback.py
# @Software : PyCharm

import pandas as pd
import os

def extract_misclassified_samples(
        df: pd.DataFrame,
        target_col: str,
        prob_col: str,
        output_path: str,
        threshold: float = 0.5,
        patient_id_col: str = "patient_sn"
):

    sub_df = df[[patient_id_col, target_col, prob_col]].dropna(subset=[target_col])
    print(len(sub_df))
    y_true = sub_df[target_col].astype(int)
    y_pred = (sub_df[prob_col] >= threshold).astype(int)
    misclassified = sub_df[y_true != y_pred].copy()
    misclassified["error_type"] = misclassified.apply(
        lambda row: (
            "FN" if row[target_col] == 1 and row[prob_col] < threshold else
            "FP"
        ),
        axis=1
    )

    misclassified.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[✓] Saved {len(misclassified)} misclassified samples to: {output_path}")

    return misclassified


if __name__ == '__main__':

    file_path = os.environ.get('HCC_RESULT_DIR', 'outputs/development_outputs')
    ground_truth_path = file_path + "ground_truth_with_predictions.csv"
    optimization_path = file_path + "optimization_analysis.csv"

    ground_truth_df = pd.read_csv(ground_truth_path, encoding='utf-8-sig')
    optimization_df = pd.read_csv(optimization_path, encoding='utf-8-sig')
    output_dir = file_path + "error_analysis/"
    os.makedirs(output_dir, exist_ok=True)

    err_0_6 = extract_misclassified_samples(
                df=ground_truth_df,
                target_col="Target_0_to_6m",
                prob_col="p_0_6",
                output_path=os.path.join(output_dir, "misclassified_0_6m.csv"))
    opt_err_0_6 = optimization_df[
        optimization_df["Patient_ID"].isin(err_0_6["patient_sn"])
    ].copy()

    opt_err_0_6.to_csv(
        os.path.join(output_dir, "optimization_misclassified_0_6m.csv"),
        index=False,
        encoding="utf-8-sig")

    err_6_24 = extract_misclassified_samples(
                df=ground_truth_df,
                target_col="Target_6_to_24m",
                prob_col="p_6_24",
                output_path=os.path.join(output_dir, "misclassified_6_24m.csv")
                )
    opt_err_6_24 = optimization_df[
        optimization_df["Patient_ID"].isin(err_6_24["patient_sn"])
    ].copy()

    opt_err_6_24.to_csv(
        os.path.join(output_dir, "optimization_misclassified_6_24m.csv"),
        index=False,
        encoding="utf-8-sig")

    err_24_60 = extract_misclassified_samples(
                df=ground_truth_df,
                target_col="Target_24_to_60m",
                prob_col="p_24_60",
                output_path=os.path.join(output_dir, "misclassified_24_60m.csv")
            )

    opt_err_24_60 = optimization_df[
        optimization_df["Patient_ID"].isin(err_24_60["patient_sn"])
    ].copy()

    opt_err_24_60.to_csv(
        os.path.join(output_dir, "optimization_misclassified_24_60m.csv"),
        index=False,
        encoding="utf-8-sig")

