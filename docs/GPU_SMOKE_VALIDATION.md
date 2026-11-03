# GPU compatibility validation — 3 November 2026

**PASS: eight original recognizers and six seed-42 ablation entries.** Thirteen distinct checkpoint workers completed; the identical full seed-42 EMA/configuration reused the CNN–ViT evidence. All available requested assets were found and hash-verified. No unresolved execution blocker remains for this bounded smoke scope.

This report concerns synthetic execution compatibility, not independent reproduction of the numerical paper tables. The machine-readable [sanitized summary](../reproducibility/reference/GPU_SMOKE_COMPATIBILITY.json) includes full checkpoint/configuration SHA-256 identities and actual tensor shapes/dtypes. Private raw logs and local artifact paths are excluded from Git.

## Model-validation matrix

All entries passed complete state-key/shape checks followed by `strict=True` loading, GPU inference, finite native scores, decoding, and seven-position metric-interface checks. Configuration paths and full SHA-256 identities follow below. Input is `[1,F,3,32,96]` FP32 on `cuda:0`, under CUDA FP16 autocast. Native FP32 score outputs are valid mixed-precision behavior; they were not forcibly cast. Memory is the observed PyTorch peak allocated/reserved, in MiB, across each worker, not the paper benchmark measurement.

| Model/asset | Original checkpoint | SHA-256 prefix | GPU paths | Native score dtype | Peak allocated / reserved MiB | Status |
| --- | --- | --- | --- | --- | --- | --- |
| Ours | `last.pth` | `1a7a640dafc3` | F1/F3/F5 × four rules; FP16 efficiency F1 | torch.float16 | 207.77 / 292.00 | PASS |
| MDiff4STR | `model_acc_0.6040_ep_21.pth` | `e7ec4524f397` | F1/F3/F5 × four rules; FP16 efficiency F1 | torch.float32 | 204.68 / 216.00 | PASS |
| CPPD | `model_acc_0.6880_ep_77.pth` | `1de6f9cd0d42` | F1/F3/F5 × four rules; FP16 efficiency F1 | torch.float32 | 167.32 / 186.00 | PASS |
| IGTR | `model_acc_0.6600_ep_61.pth` | `5d6a367dbb88` | F1/F3/F5 × four rules; FP16 efficiency F1 | torch.float32 | 195.93 / 216.00 | PASS |
| LISTER | `model_acc_0.6830_ep_113.pth` | `bd94b4bec91d` | F1/F3/F5 × four rules; FP16 efficiency F1 | torch.float16 | 48.21 / 62.00 | PASS |
| SVTRv2 | `model_acc_0.7030_ep_156.pth` | `6590d3da20d2` | F1/F3/F5 × four rules; FP16 efficiency F1 | torch.float32 | 36.05 / 50.00 | PASS |
| OTE | `model_acc_0.7430_ep_88.pth` | `2a64e2fcee16` | F1/F3/F5 × four rules; FP16 efficiency F1 | torch.float32 | 126.34 / 134.00 | PASS |
| SVTRv2-AR | `student_acc_0.7500_ep_52.pth` | `c6f89c74deed` | F1/F3/F5 × four rules; FP16 efficiency F1 | torch.float32 | 149.29 / 160.00 | PASS |
| full-42-ghost | `last.pth` | `1a7a640dafc3` | F1 PRF (reused from ours) | torch.float16 | 207.77 / 292.00 | PASS |
| no_restormer-42-ghost | `ghost_acc_0.7710_ep_69.pth` | `9bf18a7f5840` | F1 PRF | torch.float16 | 110.97 / 130.00 | PASS |
| no_pixelshuffle-42-ghost | `ghost_acc_0.7620_ep_66.pth` | `fdcfd7629f48` | F1 PRF | torch.float16 | 108.59 / 122.00 | PASS |
| no_sfb-42-ghost | `ghost_acc_0.7670_ep_50.pth` | `2776bb5eba22` | F1 PRF | torch.float16 | 111.90 / 130.00 | PASS |
| linear_head-42-ghost | `ghost_acc_0.7670_ep_51.pth` | `615fbfadc1a2` | F1 PRF | torch.float16 | 116.81 / 130.00 | PASS |
| full-42-student | `student_acc_0.7510_ep_96.pth` | `6f5e908a6338` | F1 PRF | torch.float16 | 116.81 / 130.00 | PASS |

## Exact identities and configurations

- **ours**: `experiments/revision_eval/submitted_model/config_snapshot.yaml`; config SHA-256 `9db50ded91aabfff67d2863057e51bd4ef6a8dd8da5ce7a67c61fe3626fd7673`; checkpoint SHA-256 `1a7a640dafc368a8dda9ed6a18239b2d54a1f39e9aa0b581a5a405ec0697b11a`.
- **mdiff4str**: `baselines_configs/MDIFF_BASELINE.yaml`; config SHA-256 `bbca4c50586fe343f7e6bf14e1f75086403d910ccd7b4327c43383a354c516d9`; checkpoint SHA-256 `e7ec4524f397a15fe1ce7629b40c7cf7415028434eec2a3be0d092c9d64f5143`.
- **cppd**: `baselines_configs/CPPD_BASELINE.yaml`; config SHA-256 `3a3e2e1e76b18d92fad2629b33cc05b5c5cd4a7d21a4f83e49b65a1267cbf402`; checkpoint SHA-256 `1de6f9cd0d424e714cb2c7721747ea296a82760f8a786f1849793fe5c1e7e89f`.
- **igtr**: `baselines_configs/IGTR_BASELINE.yaml`; config SHA-256 `5d33570d2e08e3792d8d6af985c7b1133a923247835031d743fe237944ccf773`; checkpoint SHA-256 `5d6a367dbb888a65e540b3491779cfa1fefa8dd69e33048c35d3faaa8d9d75d0`.
- **lister**: `baselines_configs/LISTER_BASELINE.yaml`; config SHA-256 `99a1e17a2f104ab0ce0c7cb1717ee67b86c631cfed50f08896bb3804e3f64416`; checkpoint SHA-256 `bd94b4bec91dadd1cc6ad93cb4f2941819ce486451e32d0b0cff226d48ba4ec0`.
- **svtrv2**: `baselines_configs/SVTRV2_BASELINE.yaml`; config SHA-256 `9a48913d5ef5e93d1037219687d872f67b37d4cdbdba0c4ef22b345d5acb89cb`; checkpoint SHA-256 `6590d3da20d204c235e88a65871b8bc04b85ca3dd6d52ad81f73face8b68f231`.
- **ote**: `baselines_configs/OTE_BASELINE.yaml`; config SHA-256 `f431a42fd9eb1c446b95b527973062f0ef37db9054251f03a56e89ab52421636`; checkpoint SHA-256 `2a64e2fcee16e1aaa1ebd2fdfa93a140d55e31231d8c44f4608b831fe47408a1`.
- **svtrv2_ar**: `baselines_configs/SVTRV2_AR_BASELINE.yaml`; config SHA-256 `f8c656f5486018553c88746678de842d7dd5ff3ebb30f1697ad48a3e8dc0a31f`; checkpoint SHA-256 `c6f89c74deed70ab9dcc7ca46d7dd0f7c21cb0b3074a4dd15313689eb8692838`.
- **full-42-ghost**: `experiments/revision_eval/submitted_model/config_snapshot.yaml`; config SHA-256 `9db50ded91aabfff67d2863057e51bd4ef6a8dd8da5ce7a67c61fe3626fd7673`; checkpoint SHA-256 `1a7a640dafc368a8dda9ed6a18239b2d54a1f39e9aa0b581a5a405ec0697b11a`.
- **no_restormer-42-ghost**: `experiments/revision_eval/reviewer2/01_controlled_ablation/configs/no_restormer_seed42.yaml`; config SHA-256 `5361feb843fad8a0549f806279ddc0fb58cdae754e5b8440c05fbfbd0cae5227`; checkpoint SHA-256 `9bf18a7f584094c9df658b77e0a36186e9e4f59a11d2d260ce0b5107b2c33f4c`.
- **no_pixelshuffle-42-ghost**: `experiments/revision_eval/reviewer2/01_controlled_ablation/configs/no_pixelshuffle_seed42.yaml`; config SHA-256 `11e31246a53d65e64c795f0cc6f3d81a189f68c9b8f8103e1d20048fb496972f`; checkpoint SHA-256 `fdcfd7629f4891c0a0983a25c4db67c770227fee6e140c5956ef6041e83f550a`.
- **no_sfb-42-ghost**: `experiments/revision_eval/reviewer2/01_controlled_ablation/configs/no_sfb_seed42.yaml`; config SHA-256 `4b75c9d6a410451fde61cdf349f30c5970e3088d71f827c92c9ccf8cd5b44d0f`; checkpoint SHA-256 `2776bb5eba22033af82fcf0da007adc8514bbbbda268dc89fbca39840f65fad6`.
- **linear_head-42-ghost**: `experiments/revision_eval/reviewer2/01_controlled_ablation/configs/linear_head_seed42.yaml`; config SHA-256 `91cc9857dae70bd142cc6dd9da8c98cee34b1fd64085d8572835e7a994d02b40`; checkpoint SHA-256 `615fbfadc1a27ebae33f38b4df4cc9ab4a5785b5e69204f0c06f4b0b58dd8040`.
- **full-42-student**: `experiments/revision_eval/submitted_model/config_snapshot.yaml`; config SHA-256 `9db50ded91aabfff67d2863057e51bd4ef6a8dd8da5ce7a67c61fe3626fd7673`; checkpoint SHA-256 `6f5e908a63382a19da137b1f0147aff3adf9c1deae23761470809fe431e9ef55`.

## Actual GPU execution and protocol coverage

- Physical GPU0 on controlled host was verified as a Quadro RTX 8000 (49,152 MiB total). Every worker used `CUDA_VISIBLE_DEVICES=0`; PyTorch reported one visible GPU, with the same complete physical UUID as `nvidia-smi`. Full device identity and protected production process snapshots are retained privately.
- All eight main recognizers completed all twelve F=1/3/5 × `bayes`/`average`/`majority`/`char_majority` paths: **96 actual GPU inference/decoding checks**, not CPU substitutes. PRF retains the `--fusion bayes` interface. SVTRv2-AR used its original shared-prefix score-level path and independent per-frame decoding for votes. Native CTC decoding/collapse and the original score-fusion implementation were preserved; repeated input does not establish timestep alignment across different observations.
- The original evaluation wrapper consumed one generated 13×37 RGB gradient and performed OpenCV bicubic resizing to 32×96, tensor conversion and channel-wise normalization `(x-0.5)/0.5`. F=3/5 repeated that same image. There was no TTA, adaptation, sample label, data rebuild or dataset opening. This also exercised the Table3-compatible single-crop preprocessing/inference interface; it does not test real LPLCv2 acquisition or cohort reconstruction.
- All eight models completed exactly one efficiency-worker F1 forward under FP16 autocast, with benchmark TF32 disabled. The smoke used evaluation deterministic backend settings, not the original benchmark kernel-search/timing campaign. No 100-track input cache, 30 warmups, 100 timing iterations, FLOP profiling or efficiency-number comparison was run.
- Five additional seed-42 ablation checkpoints completed one F1 PRF inference each; the sixth full-model EMA entry reused identical main-model evidence after independent mapping/hash checks. Seeds123/2026 were not loaded. No three-seed aggregates were recalculated.
- Total successful bounded calls: **109** (96 main inference paths + eight efficiency forwards + five additional ablation paths). Empty/variable-length native predictions were permitted on synthetic input; fixed seven-position scoring compatibility was checked without reporting recognition accuracy or decoded strings.

## Corrections, retries and regression evidence

1. Added `reproducibility/smoke.py`, a separate standard-library controller and bounded GPU worker. Original complete-cohort fingerprint checks in `reproducibility/run.py` and `common.py` were not weakened. SHA-256 verification precedes checkpoint deserialization. Each worker has a fresh process, bounded timeout and private output; the controller handles independently blocked assets and reuses only matching checkpoint/config identities.
2. Mechanically extracted the original native inference body into `test.py:infer_sequence`, used by both normal evaluation and smoke execution. Extracted `benchmark_one.py:forward_logits_once` similarly. No fusion arithmetic, model architecture, native decoder or scientific tensor interpretation changed. Original-to-candidate source hashes are recorded in `source_inventory.json`.
3. The initial pre-inference attempt rejected a valid device because PyTorch reports the UUID without the `GPU-` prefix. Added prefix-only normalization while retaining full UUID equality. That attempt loaded no checkpoint and performed no model inference. Only its own controller/worker processes were stopped. A fresh complete campaign then passed all14 entries.
4. The new frozen-AST parity tests initially failed on paper Python3.9 because Python3.12 adds an empty `FunctionDef.type_params` schema field. Canonical serialization now ignores that schema field while preserving the scientific statements; expected hashes were derived from the starting revision, not from the refactored implementation. Both environments pass.
5. **26 CPU tests pass** on local Python3.12 and on controlled host Python3.9 with GPUs hidden (17 existing + nine focused smoke tests). They cover original frozen inference/benchmark-body parity, exact matrix scope, UUID representation, device/resource/competing-process guards and bounded retries, missing/mismatched assets, output-directory safety, and native-string/fixed-slot interface checks. Existing state-compatibility, fusion mathematics/ties and approved Table3-display checks remain passing.
6. The entire GPU matrix passed after the shared production-path extraction and UUID correction. Subsequent changes affected tests, documentation and provenance summaries only; no model adapter or scientific execution path required correction. Known harmless timm import deprecation warnings were retained; no installed package was modified.

## Artifact, runtime and production integrity

- Original checkpoint files were SHA-256 checked before load and again after successful use; original configuration files on controlled host independently match every frozen requested configuration hash. All original configuration snapshots, checkpoint selections, reference-count files and scientific results remain unchanged. No restricted source-history merge, dataset/weight upload or sample-level export occurred.
- Runtime matches the frozen profile: Python3.9.21, torch2.6.0+cu124, torchvision0.21.0+cu124, NumPy2.0.2, PyYAML6.0.2, CUDA12.4 and cuDNN90100. Existing installation used; clean installation remains unverified.
- GPU0 had47,071 MiB free and0% utilization before and after the campaign. All four protected GPU0 production process IDs remained present. Read-only before/after container checks showed the same running services and unchanged uptime classes/health status. No container, service, network or production process was changed or terminated. Other GPUs were not assigned to any inference worker.
- Repository remains private; only the isolated reproducibility branch is updated. No merge, visibility change, release, checkpoint publication, manuscript/response edit, training or full evaluation was performed. X3 remains deferred without investigation or changes.
- Table3 reported values80.5%,67.7%,63.8%, exact counts and the three approved historical-display disclosures are preserved. Strict reported-display reference verification passes; direct-rounding differences remain visible.

## Remaining limits and release gates

Smoke compatibility is established for the requested original models. Full-cohort numerical reproduction, real temporal variation/calibration/alignment, real LPLCv2 access/preprocessing coverage, three-seed training reproducibility and original timing measurements remain unverified in this campaign. The missing original continuation predecessor/RNG lineage is unchanged. Independently licensed datasets, checkpoint/source redistribution rights, coauthor approval, restricted Git-history remediation, clean-reader installation and the final comparison with the author's new paper commit remain separate publication gates.

See [command recipes](COMMAND_RECIPES.md) for the single smoke entry point. This commit and its parent chain identify the validated private candidate; the exact tested source-file hashes are included in the sanitized summary.
