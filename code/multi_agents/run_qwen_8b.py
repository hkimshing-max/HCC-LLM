# --*-- conding:utf-8 --*--
# @Author : Jacob Xu
# Personal email omitted from publication copy; original author attribution retained.
# @File : run_qwen_8b.py
# @Software : PyCharm

import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from openai import images
from sqlalchemy import false
import os
import asyncio
import pandas as pd

from utils import save_messages_to_txt,load_patient_data,split_patient_json
import agents_configs as a_configs
import os

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.environ.get('HCC_API_KEY', '')

async def main(split_data,file_name,OUTPUT_DIR):

    model_client = OpenAIChatCompletionClient(
        model="Qwen/Qwen3-8B",
        base_url="https://api.siliconflow.cn/v1",
        api_key=API_KEY,
        temperature=0,
        model_info={
            "vision": True,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",
            "structured_output": True,
        },
    )

    # Medical_NER_Agent, cTNM_MDT_Agent，Liver_Function_Agent，Patient_Status_Agent，HCC_PATHOLOGY_AGENT,
    # Recurrence_Risk_Agent x n,FINAL_DECISION_AGENT

    medical_ner_agent = AssistantAgent(
        "Medical_NER_Agent",
        model_client=model_client,
        description=a_configs.Medical_NER_AGENT_DESCRIPTION,
        system_message=a_configs.Medical_NER_AGENT_SYSTEM_MESSAGE,
    )

    ctnm_agent = AssistantAgent(
        "cTNM_Staging_Agent",
        model_client=model_client,
        description=a_configs.cTNM_MDT_AGENT_DESCRIPTION,
        system_message=a_configs.cTNM_MDT_AGENT_SYSTEM_MESSAGE,
    )

    liver_function_agent = AssistantAgent(
        "Liver_Function_Agent",
        model_client=model_client,
        description=a_configs.LIVER_FUNCTION_AGENT_DESCRIPTION,
        system_message=a_configs.LIVER_FUNCTION_AGENT_SYSTEM_MESSAGE,
    )

    patient_status_agent = AssistantAgent(
        "Patient_Status_Agent",
        model_client=model_client,
        description=a_configs.PATIENT_STATUS_AGENT_DESCRIPTION,
        system_message=a_configs.PATIENT_STATUS_AGENT_SYSTEM_MESSAGE,
    )

    hcc_pathology_agent = AssistantAgent(
        "Hcc_Pathology_Agent",
        model_client=model_client,
        description=a_configs.HCC_PATHOLOGY_AGENT_DESCRIPTION,
        system_message=a_configs.HCC_PATHOLOGY_AGENT_SYSTEM_MESSAGE,
    )

    recurrence_risk_agent_1 = AssistantAgent(
        "Recurrence_Risk_Agent_1",
        model_client=model_client,
        description=a_configs.RECURRENCE_RISK_AGENT_DESCRIPTION,
        system_message=a_configs.RECURRENCE_RISK_AGENT_SYSTEM_MESSAGE,
    )

    recurrence_risk_agent_2 = AssistantAgent(
        "Recurrence_Risk_Agent_2",
        model_client=model_client,
        description=a_configs.RECURRENCE_RISK_AGENT_DESCRIPTION,
        system_message=a_configs.RECURRENCE_RISK_AGENT_SYSTEM_MESSAGE,
    )

    recurrence_risk_agent_3 = AssistantAgent(
        "Recurrence_Risk_Agent_3",
        model_client=model_client,
        description=a_configs.RECURRENCE_RISK_AGENT_DESCRIPTION,
        system_message=a_configs.RECURRENCE_RISK_AGENT_SYSTEM_MESSAGE,
    )

    recurrence_risk_agent_4 = AssistantAgent(
        "Recurrence_Risk_Agent_4",
        model_client=model_client,
        description=a_configs.RECURRENCE_RISK_AGENT_DESCRIPTION,
        system_message=a_configs.RECURRENCE_RISK_AGENT_SYSTEM_MESSAGE,
    )

    fianl_decsions_agent = AssistantAgent(
        "Final_Decision_Agent",
        model_client=model_client,
        description=a_configs.FINAL_DECISION_AGENT_DESCRIPTION,
        system_message=a_configs.FINAL_DECISION_AGENT_SYSTEM_MESSAGE,
    )

    termination = MaxMessageTermination(max_messages = 8)
    # termination = TextMentionTermination("FINAL_DECISION_DONE")
    group_chat = RoundRobinGroupChat(
        [medical_ner_agent, ctnm_agent, liver_function_agent,
         patient_status_agent,hcc_pathology_agent,recurrence_risk_agent_1,fianl_decsions_agent],
        termination_condition=termination  # 取消终止条件
    )

    task_input = (
        "以下是患者的全部临床数据，请各位智能体协作分析：\n\n"
        f"【影像学资料】\n{split_data['imaging_data']}\n\n"
        f"【病理资料】\n{split_data['pathology_data']}\n\n"
        f"【患者信息】\n{split_data['patient_info']}\n\n"
        f"【实验室检查】\n{split_data['lab_data']}\n\n"
        "请各自根据职责分析相关部分，并在需要时引用前一个智能体的输出。\n"
    )

    print("=== Multi-Agent Collaboration Start ===")
    result = await group_chat.run(task=task_input,)
    messages = result.messages

    # mac_result_path = "./multi_agents_conservation/"
    # output_dir = os.path.join(mac_result_path, file_name + "_output.txt")
    output_dir = os.path.join(OUTPUT_DIR, file_name + "_output.txt")
    save_messages_to_txt(messages, output_dir, file_name)


async def run_all_patients(DATA_DIR, OUTPUT_DIR, max_concurrent):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    json_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')]
    print(f"number of patients: {len(json_files)}")

    # ========= finished_records.csv 处理 =========
    csv_path = os.path.join(OUTPUT_DIR, "finished_records.csv")

    if os.path.exists(csv_path):
        finished_df = pd.read_csv(csv_path)
        finished_set = set(finished_df["file_name"].astype(str))
        print(f"found existing csv, finished: {len(finished_set)}")
    else:
        pd.DataFrame(columns=["file_name"]).to_csv(csv_path, index=False)
        finished_set = set()
        print("created new finished_records.csv")

    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = []

    async def sem_task(data, name, outdir):
        async with semaphore:
            return await main(data, name, outdir)

    for json_file in json_files:
        file_base = json_file.replace(".json", "")
        if file_base in finished_set:
            print(f"⏭ skip finished: {file_base}")
            continue

        print(f"processing patient: {file_base}")
        patient_info_path = os.path.join(DATA_DIR, json_file)
        patients_info = load_patient_data(patient_info_path)
        split_data = split_patient_json(patients_info)

        task = asyncio.create_task(
            sem_task(split_data, file_base, OUTPUT_DIR)
        )
        tasks.append(task)

    if tasks:
        await asyncio.gather(*tasks)
    else:
        print("No new patients to process ✅")

if __name__ == "__main__":
    # asyncio 异步并发
    DATA_DIR = os.environ.get('HCC_DATA_DIR', 'data/test_json')

    OUTPUT_DIR = os.environ.get('HCC_OUTPUT_DIR', os.path.join(DATA_DIR, 'test_qwen_8B_1'))
# Configure input/output locations using the HCC_* environment variables.
    # OUTPUT_DIR = os.path.join(DATA_DIR, 'test_result')

    asyncio.run(run_all_patients(DATA_DIR, OUTPUT_DIR, max_concurrent=1))