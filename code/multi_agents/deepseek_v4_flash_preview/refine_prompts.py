# --*-- conding:utf-8 --*--
# @Author : Jacob Xu
# Personal email omitted from publication copy; original author attribution retained.
# @File : refine_prompts.py
# @Software : PyCharm

import asyncio
import csv
import os
import json
from datetime import datetime

# Autogen Imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage

# Custom Imports
from utils import save_messages_to_txt, load_patient_data, split_train_patient_json
import agents_configs as a_configs

from dotenv import load_dotenv
load_dotenv()

# apiKey = os.getenv("DEEPSEEK_API_KEY")

API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL = os.getenv("DEEPSEEK_MODEL")          # DeepSeek-V3.2 官方模型名
BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

# ==========================================
# 1. 定义 Optimization/Debugging Agent 的系统提示词
# ==========================================
OPTIMIZATION_FEEDBACK_PROMPT = """
    You are an Expert Analyst specializing in Medical AI Logic Evaluation.

    GOAL
    Your task is to compare the AI's "Predicted Recurrence Interval" against the Patient's "Actual Outcome" (Ground Truth).
    You must identify *why* the prediction deviated from reality by tracing the logic used in the Decision Tree nodes.

    INPUT DATA
    1. AI Prediction: The interval chosen by the model (e.g., 6-24 Months) and its reasoning.
    2. Actual Outcome: The patient's actual Time-to-Recurrence (TTR) or Disease-Free Survival (DFS).

    TASK: ERROR CLASSIFICATION & ROOT CAUSE ANALYSIS
    Compare the two data points and classify the result into one of the following categories:

    TYPE A: PREDICTION ALIGNED (Correct)
    - The predicted interval matches the actual outcome.
    - Action: Briefly confirm which logical node or rule successfully captured the risk.

    TYPE B: RISK OVER-ESTIMATION (False Alarm)
    - Scenario: Model predicted Early/Intermediate Recurrence (e.g., 0-6m or 6-24m), but the patient survived much longer (e.g., >60m or No Recurrence).
    - Analysis Focus: Did the model fail to apply a "Mitigation Rule" (e.g., Resilient Host, Benign Giant)? Did it weigh a single risk factor (like MVI or AFP) too heavily without considering protective factors?

    TYPE C: RISK UNDER-ESTIMATION (Missed Hazard)
    - Scenario: Model predicted Late/No Recurrence, but the patient recurred quickly (e.g., within 6 months).
    - Analysis Focus: Did the model miss a "Synergistic Risk" (e.g., Small tumor but terrible liver background)? Did it downgrade a risk too aggressively?

    OUTPUT FORMAT (Strict JSON-like Structure for Analysis)
    1. Verdict: [Aligned / Over-estimated / Under-estimated]
    2. The Gap: "Predicted [X months] vs Actual [Y months]"
    3. Failure Point: 
       - Identify the specific Node (1, 2, 3, or 4) where the logic faltered.
       - Example: "Failed at Node 2: The model stopped at 'High Risk' due to MVI, ignoring the patient's ALBI Grade 1 status."
    4. Prompt Optimization Advice: 
       - Provide a specific, actionable suggestion to modify the 'FINAL_DECISION_AGENT' prompt.
       - Example: "Add a condition in Node 2 to downgrade risk if Tumor < 3cm and Age < 40."
"""

async def main(split_data, file_name, OUTPUT_DIR, csv_writer, csv_lock):

    # 初始化模型客户端
    model_client = OpenAIChatCompletionClient(
        model=MODEL,
        base_url=BASE_URL,
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

    # ==========================================
    # 2. 初始化所有 Agents
    # ==========================================

    medical_ner_agent = AssistantAgent(
        "Medical_NER_Agent", model_client=model_client,
        description=a_configs.Medical_NER_AGENT_DESCRIPTION,
        system_message=a_configs.Medical_NER_AGENT_SYSTEM_MESSAGE,
    )
    ctnm_agent = AssistantAgent(
        "cTNM_Staging_Agent", model_client=model_client,
        description=a_configs.cTNM_MDT_AGENT_DESCRIPTION,
        system_message=a_configs.cTNM_MDT_AGENT_SYSTEM_MESSAGE,
    )
    liver_function_agent = AssistantAgent(
        "Liver_Function_Agent", model_client=model_client,
        description=a_configs.LIVER_FUNCTION_AGENT_DESCRIPTION,
        system_message=a_configs.LIVER_FUNCTION_AGENT_SYSTEM_MESSAGE,
    )
    patient_status_agent = AssistantAgent(
        "Patient_Status_Agent", model_client=model_client,
        description=a_configs.PATIENT_STATUS_AGENT_DESCRIPTION,
        system_message=a_configs.PATIENT_STATUS_AGENT_SYSTEM_MESSAGE,
    )
    hcc_pathology_agent = AssistantAgent(
        "Hcc_Pathology_Agent", model_client=model_client,
        description=a_configs.HCC_PATHOLOGY_AGENT_DESCRIPTION,
        system_message=a_configs.HCC_PATHOLOGY_AGENT_SYSTEM_MESSAGE,
    )
    recurrence_risk_agent_1 = AssistantAgent(
        "Recurrence_Risk_Agent_1", model_client=model_client,
        description=a_configs.RECURRENCE_RISK_AGENT_DESCRIPTION,
        system_message=a_configs.RECURRENCE_RISK_AGENT_SYSTEM_MESSAGE,
    )

    # 最终决策 Agent (被调试对象)
    final_decision_agent = AssistantAgent(
        "Final_Decision_Agent", model_client=model_client,
        description=a_configs.FINAL_DECISION_AGENT_DESCRIPTION,
        system_message=a_configs.FINAL_DECISION_AGENT_SYSTEM_MESSAGE,
    )

    # 调试/复盘 Agent (使用新的 Prompt)
    feedback_agent = AssistantAgent(
        "Optimization_Analyst", model_client=model_client,
        description="A consultant analyzing the gap between prediction and reality.",
        system_message=OPTIMIZATION_FEEDBACK_PROMPT,
    )

    # ==========================================
    # 3. Phase 1: 初始多智能体协作 (盲测)
    # ==========================================
    # 注意：这里我们故意不把 recurrence_info 给前面的 Agent，模拟真实预测场景

    termination = MaxMessageTermination(max_messages=8)
    group_chat = RoundRobinGroupChat(
        [medical_ner_agent, ctnm_agent, liver_function_agent,
         patient_status_agent, hcc_pathology_agent, recurrence_risk_agent_1, final_decision_agent],
        termination_condition=termination
    )

    task_input = (
        "以下是患者的全部临床数据，请各位智能体协作分析：\n\n"
        f"【影像学资料】\n{split_data['imaging_data']}\n\n"
        f"【病理资料】\n{split_data['pathology_data']}\n\n"
        f"【患者信息】\n{split_data['patient_info']}\n\n"
        f"【实验室检查】\n{split_data['lab_data']}\n\n"
        "请各自根据职责分析相关部分，并在需要时引用前一个智能体的输出。\n"
    )

    print(f"[{file_name}] === Multi-Agent Collaboration Start ===")

    # 运行群聊
    result = await group_chat.run(task=task_input)
    messages = result.messages

    # 获取模型生成的预测结果 (Draft)
    draft_decision_msg = messages[-1]
    draft_content = draft_decision_msg.content

    # 保存完整的群聊日志
    txt_output_path = os.path.join(OUTPUT_DIR, file_name + "_chat_history.txt")
    save_messages_to_txt(messages, txt_output_path, file_name)

    # ==========================================
    # 4. Phase 2: Hindsight Analysis (复盘分析)
    # ==========================================

    # 提取真实标签
    ground_truth = split_data['recurrence_info']

    print(f"[{file_name}] === Starting Optimization Analysis ===")

    # 构造复盘 Prompt：同时包含【模型预测】和【真实答案】
    analysis_input = (
        f"Model Prediction JSON:\n{draft_content}\n\n"
        f"--- REAL GROUND TRUTH ---\n"
        f"{ground_truth}\n\n"
        "Please compare the prediction with the reality. "
        "Identify WHY the model failed (or succeeded). "
        "Which specific logic rule in the prompt needs to be tuned?"
    )

    # 请求 Feedback Agent 进行分析
    feedback_response = await feedback_agent.on_messages(
        [TextMessage(content=analysis_input, source="user")],
        cancellation_token = None
    )
    analysis_report = feedback_response.chat_message.content

    # 注意：在这个模式下，我们不需要让 Final Decision Agent 重写 JSON。
    # 因为我们的目的是收集分析报告去改代码，而不是改这个特定的 JSON。
    print(f"[{file_name}] Analysis Complete.")

    # ==========================================
    # 5. 保存结果到 CSV
    # ==========================================

    async with csv_lock:
        csv_writer.writerow([
            file_name,
            ground_truth,    # 真实结果
            draft_content,   # 模型预测
            analysis_report  # 优化建议报告
        ])

async def run_all_patients(DATA_DIR, OUTPUT_DIR, max_concurrent):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')]
    print(f"Total patients: {len(json_files)}")

    # 初始化 CSV 文件 - 适配优化模式的表头
    csv_path = os.path.join(OUTPUT_DIR, "optimization_analysis.csv")
    csv_headers = ["Patient_ID", "Ground_Truth", "Model_Prediction", "Optimization_Analysis"]

    csv_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max_concurrent)

    with open(csv_path, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

        tasks = []
        for json_file in json_files:
            patient_info_path = os.path.join(DATA_DIR, json_file)
            file_base = json_file.replace(".json", "")

            # 读取数据 (使用 split_train_patient_json)
            try:
                patients_info = load_patient_data(patient_info_path)
                split_data = split_train_patient_json(patients_info)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue

            async def sem_task(data, name, outdir, writer_obj, lock_obj):
                async with semaphore:
                    try:
                        await main(data, name, outdir, writer_obj, lock_obj)
                    except Exception as e:
                        print(f"Error processing {name}: {e}")
                        async with lock_obj:
                            writer_obj.writerow([name, "ERROR", "ERROR", str(e)])

            task = asyncio.create_task(sem_task(split_data, file_base, OUTPUT_DIR, writer, csv_lock))
            tasks.append(task)

        await asyncio.gather(*tasks)

    print(f"All tasks finished. Optimization Analysis saved to {csv_path}")

if __name__ == "__main__":
    # 请根据实际路径修改
    DATA_DIR = os.environ.get('HCC_DATA_DIR', 'data/development_json')
    out_path =os.environ.get('HCC_OUTPUT_ROOT', 'outputs')
    OUTPUT_DIR = os.environ.get('HCC_OUTPUT_DIR', os.path.join(out_path, 'DeepSeek-V4-flash-03'))

    asyncio.run(run_all_patients(DATA_DIR, OUTPUT_DIR, max_concurrent=20))