# Reproducibility materials (review candidate)

This directory records the protocol and runtime metadata for the experiments discussed in the manuscript. It is a provenance bundle, not a complete rerunnable copy of the experiments.

## What is included

| File | Contents |
|---|---|
| `source_protocol_index.csv` | One row for each of the 17 source-task protocol entries, with split/group counts, protocol and result hashes, and the role assigned in the current paper. |
| `model_runtime_manifest_public.json` | Model/tokenizer file hashes, candidate token IDs, prompt/template hashes, revision status, and captured runtime settings. Absolute checkpoint paths and model weights are omitted. |
| `auxiliary_prompt_protocol.txt` | The Mistral auxiliary-model instruction, template, generation suffix, and score rounding rule used in the archived protocol. It contains no benchmark examples. |
| `runner_code_hash_index.csv` | Code paths and hashes recorded by the experiment protocols, compared with files currently present on the research server and in the public calibration package. |
| `bootstrap_code_hash_index.csv` | For each source-level paired bootstrap, links the result hash to its script hash and the current matching script, plus resample count and seed. It does not cover the separate bootstrap that refits calibrators. |
| `refit_bootstrap_provenance.csv` | Records the two batches used for the calibrator-refitting intervals, including the script hash stored in each run plan and whether that exact script is currently available. |
| `inference_environment_observed.txt` | Dependency versions captured for historical inference. This is not a lock file for the calibration-only Python package. |
| `artifact_sha256.txt` | SHA-256 checksums for the files in this directory, excluding the checksum list itself. |

The protocol index groups the 17 entries into two development/method-selection entries, two prespecified source evaluations, eleven fixed-method extensions, and two post-selection new-source evaluations. The `confirmatory_evaluation_enabled` column is copied from each historical protocol; it does not override the paper's evidence-stage classification. In particular, OR-Bench and Aegis2 are development evidence, and their archived protocol names `sqrt_gate_lr` as its primary candidate. They must not be described as independent confirmation of FusionLR.

The paired-bootstrap index covers 16 run-level records for 17 source-task entries; the development run covers both OR-Bench and Aegis2. The other 15 fixed-method entries use the identifiers `fusion_lr`, `guard_lr`, `qc_lr`, and `ts_nll`; the development run also contains gate candidates. Its method column records identifiers found in source code, not an independent audit of every executed branch. These paired intervals are conditional on fitted calibrators. The separate bootstrap that refits calibrators is summarized in `refit_bootstrap_provenance.csv`.

## What can be reproduced from the public code

The code package at the repository root can fit `fusion_lr`, `guard_lr`, and `qc_lr` from supplied score records, make predictions, and compute the listed metrics. Its CLI separates fitting/prediction from the later test-label join, checks exact test-ID coverage, and provides a paired group-bootstrap command for intervals conditional on fitted calibrators. Install it from the repository root with:

```bash
python -m pip install -e .
```

Then use `calibrationguard predict` with a labeled calibration JSONL and an unlabeled test-score JSONL. Once predictions are saved, use `calibrationguard evaluate` with a separate test-label JSONL. For paired intervals, provide a CSV manifest mapping each source and guard to prediction and test-label files, then use `calibrationguard bootstrap`. The input fields and commands are documented in the root README. The bootstrap resamples groups within each source, shares resampling weights across guards and methods, and averages source-level contrasts equally. It does not refit calibrators. This is a portable implementation of the audited paired group-bootstrap design; it has not yet been compared numerically with every archived output, and it does not recreate model scores or the separate historical refit-bootstrap intervals.

The package does not contain the experiment score files. It therefore cannot, by itself, reproduce the manuscript's tables or regenerate guard and auxiliary-model scores. Full reproduction also needs the matching inference and evaluation source, model checkpoints, benchmark versions and access, split/group manifests, and score-level records. This directory provides protocol references and hashes for those materials, not the materials themselves.

## Provenance notes and open discrepancy

The archived protocols identify four guard models and Mistral-7B-Instruct-v0.3 as the auxiliary model. The manifest records upstream commit hashes for the LG2 and Mistral checkpoints. LG3, LG7B, and WildGuard are identified by local file hashes, but the archived manifest does not provide a verified upstream revision for them.

The auxiliary prompt includes a structured analysis template, but the model is not asked to generate a rationale during scoring. The score `q_c` is the next-token probability mass for `unsafe`, normalized against the corresponding `safe` token score at the `Final Answer:` prefix. The guard prediction is not used to flip or adjust `q_c`.

The code-hash index records `eval_cot.py` as SHA-256 `0d016b9bd94ff909ec6b997c688e85c60f5cceb5ed58117ac9ec615746e920a2`. The current server file hashes to `d4a5ce637ffcb35871a19a2db36a870207cac4fc972341664f7df0ef1bc0bbe3`; the exact archived file was not found among the Python files in the current research tree. The current source and four available backups have the same prompt-definition block (SHA-256 `2a924cd40fb8910ee87ba5fac9a77f7142b7445af4c3a25002ca92a1307d9c03`), and the archived runners import only `COT_SYSTEM_PROMPT` and `COT_TEMPLATE` from this file. The scorer file `calibration/cot_scorer.py` still matches its recorded hash. This suggests the full-file mismatch may be outside the scoring path, but without the historical file it cannot prove that the original source had identical contents. Treat it as an unresolved byte-level provenance gap; it is not, by itself, evidence that the reported scores are wrong or that GPU inference must be repeated.

Historical environment versions are listed in `inference_environment_observed.txt`. They describe the inference environment captured in the archive; they are not a tested installation specification for the public calibration package.

The calibrator-refitting bootstrap used two batches. The exact script for CARES and HateCheck is available in the archive and matches its recorded hash. The Safety Moderation and HarmMetric batch records hash `ff778a4b222f7479ecc151e2bffce9825bb3eb8aab31d2c8157943c892be4466`, but that source file was not found; the currently available script has a different hash. Both plans specify 500 replicates and the same refitting rule. The current script depends on the historical shared runner and a calibration-size helper; these dependencies are not yet bundled as a clean, portable analysis package. The archived aggregate results remain traceable by their plan/result hashes, but code for regenerating all four intervals is not yet complete in the public package.

## Data and publication boundary

This candidate intentionally contains no examples, labels, sample identifiers, prediction caches, model weights, server logs, or absolute server paths. Dataset and model redistribution terms have not been cleared here. Use each benchmark through its official source and access terms; do not infer redistribution permission from this index. No new GPU inference was run to prepare these files.

This provenance bundle contains no benchmark examples, labels, sample identifiers, prediction caches, model weights, or absolute server paths. It does not grant permission to redistribute benchmark data or model files. A repository license has not been added, following the authors' instruction to defer that decision.
