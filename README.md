# CalibrationGuard FusionLR code release

This repository contains the **fixed FusionLR method** used in the current manuscript: a source-specific, L2-regularized logistic post-processor fitted from a guard probability and an auxiliary-model score. It implements grouped cross-validation for regularization selection and strict calibration/test ID and group separation.

The repository intentionally does not bundle datasets, model weights, prompt text files from third parties, cached predictions, or server logs. It contains only the current primary fusion method and its current single-signal controls (`guard_lr`, `qc_lr`). Exploratory or superseded approaches are out of scope and are not part of the paper's main method: uncertainty gates (`sqrt_gate_lr` and other gates), beta-additive fusion, the earlier TS-plus-auxiliary pipeline, and CoT variants. The article describes some of these historical experiments only as exploratory context.

## Protocol provenance

The `reproducibility_candidate/` directory contains a source-task protocol index, model/runtime hashes, the auxiliary scoring prompt, inference environment versions, and separate code-hash indexes for paired and refit bootstraps. It contains no benchmark examples, labels, sample IDs, prediction caches, weights, or absolute server paths. These records help audit experiment provenance; they do not provide the missing score files and inference scripts needed to regenerate all manuscript results. See [the reproducibility notes](reproducibility_candidate/README.md) for evidence-stage distinctions and known source-code gaps.

## Repository status

This is the cleaned publication-code version for score calibration and analysis. It begins with precomputed score records supplied by the user; the repository does not distribute those records or the end-to-end guard/auxiliary-model inference pipeline, and it does not regenerate every manuscript table.

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

## Command-line workflow

The CLI keeps test labels separate from score fitting. First fit the three current calibrators and save their test probabilities:

```bash
calibrationguard predict \
  --calibration calibration_scores.jsonl \
  --test-scores test_scores.jsonl \
  --predictions test_predictions.jsonl \
  --calibrators-out fitted_calibrators.json
```

Calibration records require `sample_id`, `group_id`, `guard_probability`, `q_c`, and binary `label`. Test-score records require the first four fields and must not contain labels. The CLI checks that calibration and test sample IDs and groups are disjoint. It writes one prediction row per sample and method.

After the prediction file is saved, evaluate it using a separate JSONL file containing `sample_id` and binary `label`:

```bash
calibrationguard evaluate \
  --predictions test_predictions.jsonl \
  --test-labels test_labels.jsonl \
  --output test_metrics.json
```

Evaluation verifies exact test-ID coverage for every method and reports 15-bin ECE, Brier score, and NLL. Defaults match the package API: five grouped folds, random state `20260930`, and the six-value ridge grid documented above. Methods can be selected with `--methods fusion_lr,guard_lr,qc_lr`; avoid changing defaults when reproducing manuscript results.

## Paired group-bootstrap intervals

For uncertainty intervals conditional on fitted calibrators, create a CSV manifest with one row per source and guard. The prediction paths point to JSONL files produced by `calibrationguard predict`; the test-label path is shared across guards within a source. Paths may be relative to the manifest:

```csv
source,guard,predictions_path,test_labels_path
CARES,WildGuard,predictions/cares_wildguard.jsonl,labels/cares_test_labels.jsonl
CARES,LG2,predictions/cares_lg2.jsonl,labels/cares_test_labels.jsonl
```

Then specify the paired contrasts. Each reported difference is method A minus method B; groups are resampled within each source, with the same resampling weights shared by guards and methods. The macro interval gives equal weight to sources:

```bash
calibrationguard bootstrap \
  --manifest bootstrap_manifest.csv \
  --pairs fusion_lr:guard_lr,fusion_lr:qc_lr \
  --replicates 10000 \
  --seed 20261008 \
  --output paired_intervals.json
```

This bootstrap does not refit the calibrators. It therefore captures test-group sampling variation conditional on the fitted models, not calibrator-fitting uncertainty or variation across arbitrary domains. The historical refit-bootstrap workflow is not included as a portable command because one recorded script snapshot and its dependency hashes are unavailable.

## Reproducibility limits

To reproduce a reported experiment, provide the matching score-extraction code, exact model/tokenizer revisions and prompts, dataset access/version, split/group manifests, protocol hash, and prediction records where licenses permit. The article's present data statement is still subject to author confirmation. Do not treat the code in this repository as evidence that new GPU inference has been run.

## Citation

Please cite the accompanying manuscript after its publication record is available, and cite the benchmark/model sources used in any reproduction. The project was developed from the research codebase associated with *On Calibration of LLM-based Guard Models for Reliable Content Moderation*; see that work and its upstream repository for the original guard-calibration implementation.
