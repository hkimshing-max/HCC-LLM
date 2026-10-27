# Filename mapping

Names use lowercase snake_case and describe each script's function. Model identifiers are retained; no filenames contain creation dates or working-copy timestamps.

| Published snapshot path | Review package path |
|---|---|
| `code/chain_of_thought/cot_performance_evaluate.py` | `code/chain_of_thought/evaluate.py` |
| `code/chain_of_thought/cot_prompt.py` | `code/chain_of_thought/prompts.py` |
| `code/chain_of_thought/deepSeek-R1-Distill-Qwen-7B_run_cot.py` | `code/chain_of_thought/run_deepseek_r1_distill_qwen_7b.py` |
| `code/chain_of_thought/deepSeek_v32_run_cot.py` | `code/chain_of_thought/run_deepseek_v3_2.py` |
| `code/chain_of_thought/qwen_8B_run_cot.py` | `code/chain_of_thought/run_qwen_8b.py` |
| `code/chain_of_thought/utils.py` | `code/chain_of_thought/utils.py` |
| `code/figs/conflict_win_with_time.py` | `code/figures/plot_conflicts.py` |
| `code/figs/model_test_results_present.py` | `code/figures/plot_performance_basic.py` |
| `code/figs/model_test_results_present_with_dsv4.py` | `code/figures/plot_performance.py` |
| `code/figs/node_blocking.py` | `code/figures/plot_conditional_blocking.py` |
| `code/figs/weight_score_with_time.py` | `code/figures/plot_feature_weights.py` |
| `code/multi_agents/deepseek_r1_distill_qwen_7B_hcc_agents.py` | `code/multi_agents/run_deepseek_r1_distill_qwen_7b.py` |
| `code/multi_agents/deepseek_v3_2/agents_configs.py` | `code/multi_agents/deepseek_v3_2/agents_configs.py` |
| `code/multi_agents/deepseek_v3_2/performance_evaluate.py` | `code/multi_agents/deepseek_v3_2/evaluate.py` |
| `code/multi_agents/deepseek_v3_2/run_hcc_agents_deepseek.py` | `code/multi_agents/deepseek_v3_2/run_inference.py` |
| `code/multi_agents/deepseek_v3_2/train_and_self_refine.py` | `code/multi_agents/deepseek_v3_2/refine_prompts.py` |
| `code/multi_agents/deepseek_v3_2/utils.py` | `code/multi_agents/deepseek_v3_2/utils.py` |
| `code/multi_agents/deepseek_v4_flash_preview/agents_configs.py` | `code/multi_agents/deepseek_v4_flash_preview/agents_configs.py` |
| `code/multi_agents/deepseek_v4_flash_preview/performance_evaluate.py` | `code/multi_agents/deepseek_v4_flash_preview/evaluate.py` |
| `code/multi_agents/deepseek_v4_flash_preview/run_hcc_agents_deepseek_v4.py` | `code/multi_agents/deepseek_v4_flash_preview/run_inference.py` |
| `code/multi_agents/deepseek_v4_flash_preview/self_refine_result_organize.py` | `code/multi_agents/deepseek_v4_flash_preview/organize_feedback.py` |
| `code/multi_agents/deepseek_v4_flash_preview/train_and_self_refine.py` | `code/multi_agents/deepseek_v4_flash_preview/refine_prompts.py` |
| `code/multi_agents/deepseek_v4_flash_preview/utils.py` | `code/multi_agents/deepseek_v4_flash_preview/utils.py` |
| `code/multi_agents/qwen_8B_hcc_agents.py` | `code/multi_agents/run_qwen_8b.py` |

The three CoT runners now import `COT_PROMPT` from `prompts`.
The initial naming pass changed only names, header comments and prompt imports.
In this confirmed package, the three code-provider-returned CoT files additionally
set temperature to 0. Prompts, metric definitions, output schemas and runtime
filenames are unchanged.

## Added companion modules

- `code/multi_agents/utils.py`: identical to the V3.2 utilities.
- `code/multi_agents/agents_configs.py`: identical to the V3.2 agent definitions.

These shared copies support the standalone Qwen and additional R1 multi-agent
runners, following the code provider's confirmation.
