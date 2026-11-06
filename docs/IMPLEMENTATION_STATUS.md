# Implementation and validation status

This independent clean-history companion contains the validated scientific source,
original configurations and aggregate references. It imports no research Git
history,branches,PRs,datasets,checkpoint binaries or plate-level records.

## Available and tested

- Original CNN–ViT CE/AdamW/ReduceLROnPlateau/patience60/EMA-warmup5configuration and original validation-selected EMA58 release asset;see [MODEL_WEIGHTS.md](MODEL_WEIGHTS.md).
- Seven referenced native baseline implementations/adapters with retained license/copyright/modification notices. No baseline or additional training-state binary is distributed.
- Frozen8-model and18-state ablation identity manifests;full-cohort dataset/checkpoint guards;four original temporal fusion rules;unchanged native model/evaluation bodies.
-35 CPU regression tests, including strict state compatibility,asset/config fingerprints,temporal controls,display disclosures,output safety,clean-export integrity, guarded frame subsets, manuscript-table parsing and the single-state reader entrypoint.
- New synthetic CPU F1example tested with the exact original EMA:strict loading,finite FP32scores `[1,7,38]`,original RGB/bicubic/normalization/native decoding. No GPU campaign repeated for publication.
- Historical GPU validation:8main recognizers and6seed42ablation entries passed13unique workers,96frame/fusion paths,8FP16 efficiency forwards and5additional ablation forwards;109bounded calls total. That original compatibility evidence is documented in [GPU_SMOKE_VALIDATION.md](GPU_SMOKE_VALIDATION.md).

Original Python execution bodies,frozen YAML bytes,checkpoints,selections,metrics
and Tables1–5references are preserved. Publication additions are documentation,
license/attribution comments,dependency constraints and a separate reader entrypoint.
The exact initial150-file export and subsequent changes were independently
inventoried;156source files are present in this clean repository.

## Remaining limits

No full numerical campaign,training or original efficiency benchmark was rerun.
Clean independent CUDA installation remains unverified;the known Python3.9
scientific environment and isolated controlled-site reader imports were checked.
Baseline/other-seed/primary/ablation states are unavailable from this release.
Dataset access must be obtained independently. The missing original continuation
predecessor/RNG lineage prevents an identical full training-trajectory replay.

Only the original EMA is authorized for weight distribution;source Apache-2.0
and inherited terms do not grant dataset or broader weight rights. The authors
will synchronize manuscript/response code-availability wording after verifying
public source and asset access. The manuscript continues to evolve;historical
paper snapshots are not represented as the author's finalized version. X3 remains
deferred without implementation changes.
