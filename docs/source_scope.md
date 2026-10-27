# Source scope and reproducibility

The `code/` tree contains the supplied CoT inference and evaluation, multi-agent
workflows, development feedback, and explanation-analysis scripts. File naming
is standardized, author attribution is retained, and private credentials and
clinical records are excluded.

## Code-provider confirmation

The author supplied three returned CoT runners for DeepSeek-V3.2,
DeepSeek-R1-Distill-Qwen-7B and Qwen3-8B. Each differs from the earlier review
copy only in changing temperature from 0.1 to 0. These returned versions are
included in `code/chain_of_thought/`; the multi-agent runners also retain 0.

The code provider confirmed that the standalone Qwen runner and the additional
R1 multi-agent runner use the V3.2 companion modules. Byte-identical copies of
the V3.2 `utils.py` and `agents_configs.py` are included in `code/multi_agents/`.
The model-specific DeepSeek directories retain their original companion files.
No prompts or metric calculations were rewritten to implement this confirmation.

The additional R1 multi-agent runner is retained as supplied source. Its
presence does not mean this configuration was among the six configurations
reported in the manuscript.

## Remaining source boundaries

The V4 preview runner reads its model identifier and endpoint from
`DEEPSEEK_MODEL` and `DEEPSEEK_BASE_URL`. A currently available provider model
is not independent evidence of the historical preview model's identity.

This package is not a complete historical execution environment. It does not
include clinical data, hosted model weights, provider credentials, all archived
run metadata, or every conventional-model and clinical feature-comparison
analysis described in the manuscript. Author-confirmed source updates do not
constitute an independent rerun or verification of the published results.

## Packaging checks

All 26 Python files compile, all local imports resolve to included source files,
and the standalone runners' requested utility functions and agent-definition
symbols are present. The returned CoT files have exactly the temperature change
described above. Relative to the normalized review copy, the other 21 existing scripts are
unchanged. The two shared modules are identical to the V3.2 copies. README entry points and archive
hashes are checked. No clinical inference, provider API call, statistical
reanalysis or full dependency/runtime test is performed during this packaging.
