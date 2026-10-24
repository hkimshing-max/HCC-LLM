# --*-- conding:utf-8 --*--
# @Time : 2026/1/30 14:17
# @Author : Jacob Xu
# Personal email omitted from publication copy; original author attribution retained.
# @File : deepSeek_v32_run_cot.py
# @Software : PyCharm
# --*-- coding:utf-8 --*--
import asyncio
import os
import json
import pandas as pd
from openai import AsyncOpenAI
from cot_prompt import COT_PROMPT
from utils import save_messages_to_txt, load_patient_data, split_patient_json

# 配置 API
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.environ.get('HCC_API_KEY', '')
BASE_URL = "https://api.siliconflow.cn/v1"
def save_cot_result(content, filepath, file_name):
    """
    针对单次 CoT 输出的保存函数。
    直接保存内容并更新已完成记录的 CSV。
    """
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")

    # 更新 CSV 记录逻辑（参考你的原始函数）
    folder = os.path.dirname(filepath)
    csv_path = os.path.join(folder, "finished_records_cot.csv")
    new_row = pd.DataFrame({"file_name": [file_name]})
    if os.path.exists(csv_path):
        new_row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        new_row.to_csv(csv_path, mode="w", header=True, index=False)

async def main(split_data, file_name, OUTPUT_DIR, client):
    """
    单个患者的处理逻辑。
    """
    task_input = (
        "Following is the clinical data for a single patient. Please analyze it strictly using your CoT workflow:\n\n"
        f"Imaging Data:\n{split_data['imaging_data']}\n\n"
        f"Pathology Data:\n{split_data['pathology_data']}\n\n"
        f"Patient Info:\n{split_data['patient_info']}\n\n"
        f"Lab Data:\n{split_data['lab_data']}\n\n"
        "Generate the final JSON output based on your internal decision tree."
    )

    print(f">>> Processing: {file_name}")

    try:
        response = await client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            messages=[
                {"role": "system", "content": COT_PROMPT},
                {"role": "user", "content": task_input}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )

        content = response.choices[0].message.content
        output_path = os.path.join(OUTPUT_DIR, f"{file_name}_CoT_output.txt")

        # 调用新写的保存函数
        save_cot_result(content, output_path, file_name)

    except Exception as e:
        print(f"❌ Error in {file_name}: {e}")

async def run_all_patients(DATA_DIR, OUTPUT_DIR, max_concurrent):
    """
    并发运行所有患者。
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

    json_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')]

    # 预先读取已完成名单，避免重复请求
    csv_path = os.path.join(OUTPUT_DIR, "finished_records_cot.csv")
    if os.path.exists(csv_path):
        finished_df = pd.read_csv(csv_path)
        finished_set = set(finished_df["file_name"].astype(str))
    else:
        finished_set = set()

    semaphore = asyncio.Semaphore(max_concurrent)

    async def sem_task(data, name, outdir):
        async with semaphore:
            await main(data, name, outdir, client)

    tasks = []
    for json_file in json_files:
        file_base = json_file.replace(".json", "")
        if file_base in finished_set:
            continue

        patient_info_path = os.path.join(DATA_DIR, json_file)
        patients_info = load_patient_data(patient_info_path)
        if patients_info is None: continue

        split_data = split_patient_json(patients_info)
        tasks.append(asyncio.create_task(sem_task(split_data, file_base, OUTPUT_DIR)))

    if tasks:
        print(f"--- Starting batch processing ({len(tasks)} patients) ---")
        await asyncio.gather(*tasks)
    else:
        print("No new patients to process ✅")

    await client.close()

if __name__ == "__main__":
# Configure input/output locations using the HCC_* environment variables.
    # OUTPUT_DIR = os.path.join(DATA_DIR, 'cot_test_result_direct')

    DATA_DIR = os.environ.get('HCC_DATA_DIR', 'data/test_json')
    OUTPUT_DIR = os.environ.get('HCC_OUTPUT_DIR', os.path.join(DATA_DIR, 'cot_test_DeepSeek-R1-Distill-Qwen-7B-1'))

    asyncio.run(run_all_patients(DATA_DIR, OUTPUT_DIR, max_concurrent=10))