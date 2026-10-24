# Input formats

These keys and columns are transcribed from the supplied source code. They are
interface descriptions, not a release of the study's clinical data or a newly
validated clinical data dictionary.

## Clinical JSON inputs

The batch inference scripts read one UTF-8 JSON file per record. The filename
stem is used as the record key when outputs are saved.

| Top-level key | Content consumed by the input adapter |
|---|---|
| `patient_info` | Clinical patient information |
| `postoperative_pathology` | Resection pathology information |
| `preoperative_examinations.imaging` | Imaging descriptions |
| `preoperative_examinations.laboratory` | Laboratory information |
| `preoperative_examinations.viral_and_other` | Viral and other laboratory information |
| `recurrence_info` | Used by the development feedback adapter, not the standard inference adapter |

The normal inference adapter combines the laboratory and viral sections, then
passes patient, imaging, pathology and laboratory information to the prompt.
Development feedback scripts additionally read outcome information for prompt
refinement. Keep development and test records separate.

## Evaluation inputs

Evaluation merges parsed model outputs with a CSV keyed by `patient_sn`. The
scripts read the following outcome columns where applicable:

- `Target_0_to_6m`
- `Target_6_to_24m`
- `Target_24_to_60m`
- `new_pfs_months`
- `recurrence`

Parsed outputs include the final decision node and a risk score. The evaluators
construct cumulative-window targets and predictions according to their existing
code. Do not substitute a different eligibility rule when preparing local inputs.
Some archived evaluators expose more than one 0–60-month definition; these are
retained as supplied and are not all asserted to represent manuscript analyses.

## Explanation analysis

Weight and conflict scripts read saved multi-agent TXT transcripts and
`ground_truth_with_predictions.csv`, including `patient_sn`, `p_0_6`, `p_0_24`
and `p_24_60` for prediction-defined grouping. Node-blocking analysis reads the
transcripts directly. These identifiers are linkage keys and should remain local.

## Performance plotting

The Excel plotting scripts expect aggregate performance summaries rather than
patient records. `model_test_results_present_with_dsv4.py` includes V4 and fills
down the workbook's `Time window` column before plotting. Inspect its column
references when preparing a compatible aggregate workbook.
