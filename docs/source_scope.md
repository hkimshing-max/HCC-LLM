# Source scope and reproducibility

The `code/` tree contains the supplied LLM source snapshot: CoT inference and
evaluation, multi-agent workflows, development feedback, and explanation plots.
The later `model_test_results_present_with_dsv4.py` plotting file is included
alongside the older performance plotting script.

Publication preparation is limited to configurable input/output locations,
environment-based credentials, omission of private email addresses and obsolete
workstation paths in comments, and documentation. Original author attribution
is retained. Prompt strings, model identifiers, temperatures, decision rules,
feature extraction, grouping and metric calculations are preserved.

The source also includes `deepseek_r1_distill_qwen_7B_hcc_agents.py`, an additional
multi-agent runner. Its presence does not mean this configuration was included
among the six configurations reported in the manuscript.

The provided Qwen multi-agent runner imports local `utils` and `agents_configs`.
Its matching companion configuration must be supplied explicitly; this package
does not silently substitute the V3.2 or V4 prompt configuration.

The supplied CoT runners use temperature 0.1 and multi-agent runners use 0. This
release preserves those source settings; a complete historical runtime record
is not distributed with the supplied code.

The API endpoint and model identifier for the V4 preview runner are read from
`DEEPSEEK_BASE_URL` and `DEEPSEEK_MODEL`. A current provider endpoint or similarly
named model is not evidence of the historical preview model's identity.

The snapshot is not a complete historical execution environment. The repository
does not contain clinical data, hosted model weights, provider credentials or
all archived run metadata. It also does not supply every conventional-model or
clinical feature-comparison analysis described in the manuscript. The README
does not claim that a fresh API run will exactly reproduce historical results.

Validation during packaging is limited to syntax, source-preservation checks,
local input-adapter checks and publication-file screening. No clinical inference
requests or new statistical analyses are run during packaging.
