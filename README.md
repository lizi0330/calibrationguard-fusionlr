# CalibrationGuard FusionLR code release

This repository contains the **fixed FusionLR method** used in the current manuscript: a source-specific, L2-regularized logistic post-processor fitted from a guard probability and an auxiliary-model score. It implements grouped cross-validation for regularization selection and strict calibration/test ID and group separation.

The repository intentionally does not bundle datasets, model weights, prompt text files from third parties, cached predictions, or server logs. It contains only the current primary fusion method and its current single-signal controls (`guard_lr`, `qc_lr`). Exploratory or superseded approaches are out of scope and are not part of the paper's main method: uncertainty gates (`sqrt_gate_lr` and other gates), beta-additive fusion, the earlier TS-plus-auxiliary pipeline, and CoT variants. The article describes some of these historical experiments only as exploratory context.

## Repository status

This is the cleaned publication-code staging version. The end-to-end guard/auxiliary model inference pipeline and redistribution rights for processed prediction caches are being audited separately. The code here begins with already computed, licensed score records; it does not regenerate model scores or itself reproduce every manuscript table.

## Install

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# activate the environment, then:
pip install -r requirements.txt
```

## Input record format

Calibration input is JSON Lines with one record per example:

```json
{"sample_id":"cal-001","group_id":"conversation-001","guard_probability":0.82,"q_c":0.71,"label":1}
```

`label` must be binary (`1` = unsafe); `guard_probability` and `q_c` are probabilities in `[0,1]`. Test-score records use the same fields except they must not include `label`. Samples and groups must be disjoint between calibration and test. Grouping keeps related prompts/conversations together during five-fold stratified group cross-validation.

## Python API

```python
from calibrationguard.fusion_lr import fit_calibrator, predict_calibrator, evaluate

calibrator = fit_calibrator(calibration_rows, method="fusion_lr")
predictions = predict_calibrator(calibrator, test_score_rows)
# After inference is complete and labels are released, join labels by sample_id.
metrics = evaluate(test_probabilities, test_labels, bins=15)
```

The default ridge grid is `1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1`. Selection uses pooled out-of-fold NLL; among values within `1e-6` of the minimum NLL, the largest penalty is selected. The final calibrator is refit on all calibration records. ECE uses 15 equal-width bins. The method does not claim that lower Brier/NLL alone demonstrates improved calibration.

The same API supports `method="guard_lr"` and `method="qc_lr"` as single-signal controls. These controls are not replacements for the paper's main FusionLR method.

## Reproducibility limits

To reproduce a reported experiment, provide the matching score-extraction code, exact model/tokenizer revisions and prompts, dataset access/version, split/group manifests, protocol hash, and prediction records where licenses permit. The article's present data statement is still subject to author confirmation. Do not treat the code in this repository as evidence that new GPU inference has been run.

## Citation

Please cite the accompanying manuscript after its publication record is available, and cite the benchmark/model sources used in any reproduction. The project was developed from the research codebase associated with *On Calibration of LLM-based Guard Models for Reliable Content Moderation*; see that work and its upstream repository for the original guard-calibration implementation.
