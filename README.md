# HCC-LLM

Source code accompanying **Time Window Stratified Recurrence Risk Prediction
After HCC Resection Using Large Language Models**.

This repository provides the supplied code for large language model (LLM)
inference, multi-agent workflows, performance evaluation and explanation analysis
for recurrence-risk classification after hepatocellular carcinoma resection.
It contains code and prompt definitions. Patient-level data, model-generated
patient explanations, API credentials and hosted model weights are not included.

## Repository layout

```text
code/
  chain_of_thought/                 CoT prompt, inference and evaluation
  multi_agents/
    deepseek_v3_2/                   V3.2 agents, prompts, feedback and evaluation
    deepseek_v4_flash_preview/       Environment-configured V4 preview workflow
    qwen_8B_hcc_agents.py            Qwen multi-agent orchestration
    deepseek_r1_distill_qwen_7B_hcc_agents.py  Additional supplied runner
  figs/                             Performance and explanation plots
docs/
  input_formats.md                  Input keys and column requirements
  source_scope.md                   Source provenance and reproducibility limits
data/README.md                      Local data handling
requirements.txt
.env.example
```

## Installation

The manuscript reports Python 3.11 and AutoGen 0.7.5. Use a separate environment:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The dependency file pins the reported AutoGen version. The other package
versions are installation dependencies, not a recovered historical lockfile.
API-based inference also requires an authorized account with the relevant model
provider. API usage may incur charges.

## Configuration

Copy `.env.example` to `.env` locally and fill in the credentials required by the
selected runner. The inference runners load `.env`; plotting and evaluation
commands should receive their `HCC_*` values through the shell environment.
Never commit `.env` or real clinical files.

| Environment variable | Purpose |
|---|---|
| `HCC_API_KEY` | Credential for the supplied CoT, V3.2 and standalone runners |
| `HCC_DATA_DIR` | Directory of locally authorized clinical JSON inputs |
| `HCC_OUTPUT_DIR` | Output directory; use a distinct directory for each configuration |
| `HCC_RESULT_DIR` | Saved output directory used by evaluation |
| `HCC_GROUND_TRUTH` | Local outcome-label CSV used by evaluation |
| `HCC_EXPLANATION_DIR` | Saved multi-agent TXT outputs for explanation analysis |
| `HCC_PREDICTIONS_CSV` | Linked outcome/prediction CSV used for grouping |
| `HCC_METRICS_WORKBOOK` | Aggregate performance workbook for plotting |
| `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL`, `DEEPSEEK_BASE_URL` | Provider settings retained by the V4 preview scripts |

Run commands from the repository root. Example environment assignment syntax:

```bash
# macOS/Linux
export HCC_DATA_DIR=/path/to/authorized/json_inputs
export HCC_OUTPUT_DIR=/path/to/local/outputs/v32_multi_agent
```

```powershell
# Windows PowerShell
$env:HCC_DATA_DIR = 'D:\authorized-data\json_inputs'
$env:HCC_OUTPUT_DIR = 'D:\local-results\v32_multi_agent'
```

## LLM inference

| Manuscript configuration | Entry point |
|---|---|
| DeepSeek-V3.2 CoT | `code/chain_of_thought/deepSeek_v32_run_cot.py` |
| DeepSeek-R1-Distill-Qwen-7B CoT | `code/chain_of_thought/deepSeek-R1-Distill-Qwen-7B_run_cot.py` |
| Qwen3-8B CoT | `code/chain_of_thought/qwen_8B_run_cot.py` |
| DeepSeek-V3.2 multi-agent | `code/multi_agents/deepseek_v3_2/run_hcc_agents_deepseek.py` |
| DeepSeek-V4-flash preview multi-agent | `code/multi_agents/deepseek_v4_flash_preview/run_hcc_agents_deepseek_v4.py` |
| Qwen3-8B multi-agent | `code/multi_agents/qwen_8B_hcc_agents.py`; matching companion configuration required |

For example:

```bash
python code/multi_agents/deepseek_v3_2/run_hcc_agents_deepseek.py
```

For CoT, select the corresponding entry point and a separate output directory.
The Qwen multi-agent runner requires its matching `agents_configs.py` and
`utils.py` on the Python module search path. Do not substitute prompts from
another configuration without establishing that they are the intended files.

The supplied scripts retain their original inference settings. CoT temperature
is 0.1 in this snapshot and multi-agent temperature is 0. See
`docs/source_scope.md` before interpreting this snapshot as an exact historical
execution record.

## Performance evaluation

Set `HCC_RESULT_DIR` to the saved transcript directory and `HCC_GROUND_TRUTH` to
the local outcome-label CSV, then select the appropriate evaluator:

```bash
python code/chain_of_thought/cot_performance_evaluate.py
python code/multi_agents/deepseek_v3_2/performance_evaluate.py
python code/multi_agents/deepseek_v4_flash_preview/performance_evaluate.py
```

These are alternative commands for their respective configurations. They are
not a request to evaluate every output directory with every evaluator. Outputs
include `ground_truth_with_predictions.csv` and performance summaries.

The archived evaluators use the key `roc_auc` for calculations that, when based
on binary predicted labels, equal balanced accuracy. The manuscript distinguishes
this quantity from continuous-score ROC-AUC. No metric definition is changed
during repository preparation.

## Explanation analysis and figures

Set `HCC_EXPLANATION_DIR` and, where required, `HCC_PREDICTIONS_CSV`. Set a
separate `HCC_OUTPUT_DIR` for each command:

```bash
python code/figs/weight_score_with_time.py
python code/figs/conflict_win_with_time.py
python code/figs/node_blocking.py
```

These scripts respectively implement explanation-assigned feature weights,
pairwise conflict summaries and conditional blocking rates. Their outputs
describe model-generated explanations and decision records, not independent
clinical prognostic effects or observed patient recurrence probabilities.

For performance plots including V4, set `HCC_METRICS_WORKBOOK` to a compatible
aggregate workbook and run:

```bash
python code/figs/model_test_results_present_with_dsv4.py
```

The older `model_test_results_present.py` is retained as supplied source and
does not replace the later V4 plotting entry point.

## Prompt development

`train_and_self_refine.py` in each DeepSeek multi-agent directory collects
development-set predictions and outcome-informed feedback for prompt refinement.
Use only an authorized development dataset for this operation. The normal test
inference path does not use the outcome-feedback adapter. Prompt strings remain
in `agents_configs.py` and `chain_of_thought/cot_prompt.py`.

## Data availability

Patient-level clinical data, outcome tables, prediction records and explanation
texts are not distributed in this code repository. Their access conditions must
be obtained from the study's separately finalized Data Availability statement.
This repository does not promise access on request or establish permission to
share clinical data.

## Authors and citation

Study authors: Jincheng Hong, Tianxin Xu, Songhui Rao, Jun Liang, Haozhong Li,
Jiangnan Han, Gengjie Jia and Dandan Hu. Original source headers attribute the
supplied scripts to Jacob Xu; those author attributions are retained.

Cite the manuscript title and the exact repository release or commit used.
No journal acceptance, article DOI or repository DOI is asserted here.

See [Source scope and reproducibility](docs/source_scope.md) for the scope of
the supplied snapshot and the limits of the packaging checks.

## License

The source code and accompanying documentation are released under the
[MIT License](LICENSE). Original source author attributions are retained.
The software license does not grant access to patient-level data or change
the terms of third-party models, datasets or dependencies.
