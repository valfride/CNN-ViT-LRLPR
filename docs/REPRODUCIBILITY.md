# Reproducing the final OJITS manuscript

This private, unmerged source candidate maps the previously reviewed paper snapshot `b0b0e94` to original artifacts. It is not assumed to be the latest paper commit. Manuscript revisions are in progress; final alignment review is pending the new commit, while the established scientific protocol remains fixed. A bounded synthetic checkpoint-loading/inference campaign is documented in [GPU_SMOKE_VALIDATION.md](GPU_SMOKE_VALIDATION.md); no training, full-cohort evaluation or efficiency benchmark was performed. Plans and aggregate verification use only the standard library.

## Environment

Original inference: Python3.9.21, torch2.6.0+cu124, torchvision0.21.0+cu124, NumPy2.0.2, CUDA12.4, cuDNN90100. Other packages are pinned in `requirements.txt`. A prospective isolated installation (not executed here) is:

```bash
python -m pip install -c requirements/constraints-paper-py39.txt torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -c requirements/constraints-paper-py39.txt -r requirements.txt
```

Use a fresh Python3.9.21 environment. Clean installation remains unverified. Preflight's optional runtime probe hides GPUs. Inference sets OMP/MKL/OPENBLAS threads2, matmulTF32=false, cuDNN TF32=true, global deterministic algorithms=false, float32 matmul precision=highest. Original `test.py` sets seed42 and cuDNN benchmark=false/deterministic=true. Precision follows exact config; no TTA/SWA. Table4 retains its original benchmark-worker backend profile instead.

## Inputs and outputs

`paper_manifest.json` pins eight checkpoint/config identities and exact test/validation/LPLCv2 file hashes. `ablation_manifest.json` pins18 selected Table5 files, their config hashes and original per-seed aggregate counts; hashes were calculated read-only on original controlled host files. No weights are included. The private path map uses keys from `weights.example.json`; relative paths resolve beside that JSON. Input paths can refer to existing files. The runner verifies hashes and stages one symlink under the original filename; it never reselects a checkpoint.

Preflight streams hashes without deserializing model or dataset files. Exact LPLCv2 metadata/data/selection/dataset-info bytes are required; byte-different reconstruction needs a separately reviewed identity process. Use only trusted, approved originals: the scientifically preserved evaluator uses checkpoint deserialization during explicitly requested execution after hashes pass. The source/config/reference fingerprint ledger rejects unreviewed changes.

Execution requires `--execute` and a new output outside the source checkout. Private predictions, GT-linked failures, logs, checkpoint links and input tensor caches stay there. Nothing uploads or stages them. Successful evaluation emits nonidentifying `aggregate.json`; partial failures leave diagnostics and no completion aggregate. Aggregate exports still require rights review before public distribution.

## Distinct reproduction goals

- **Frozen inference, Tables1–3:** exact original states/configs/cohorts/native decoding and frozen PRF choices. Structured evaluator `--mode val --dataset-phase test` computes held-out metrics; it does not select on validation.
- **Full training:** the new CE recipe only changes the historical snapshot's resume to null and adds explicit seed42. All other settings, including deferred X3, remain unchanged. It cannot reconstruct the missing original predecessor/RNG history or promise identical weights/scores. Legacy POLY and warmup15 CE configs are not the paper recipe.
- **Ablations:** five variants x3 seeds,15 selected EMA states and3 primary states from the same full-model trajectories. Best-vs-best selection may use different epochs; no matched-last substitution. The historical continued full42 trajectory is retained.
- **Efficiency:** same 100 authorized tracks and original FP16 RTX8000 protocol. Hardware timings/memory are measurements, not universally portable constants.

## Known limitations

Three Table3 one-decimal displays differ from direct rounding of original counts:

| Model/metric | Original count | Direct one decimal | Paper | Archived two decimals |
| --- | --- | --- | --- | --- |
| CNN–ViT partial≥6 | 5650/7023 | 80.4 | 80.5 | 80.45 |
| MDiff4STR partial≥5 | 4751/7023 | 67.6 | 67.7 | 67.65 |
| SVTRv2-AR partial≥6 | 4477/7023 | 63.7 | 63.8 | 63.75 |

This is consistent with second rounding of the archived display, not a changed evaluator metric. Both originals and paper values remain unchanged. The author approved retaining 80.5%, 67.7% and 63.8% on 3 November 2026 as authoritative reported displays. `PAPER_DISPLAY_DISCLOSURES.json` records that scope: display/reference consistency only, never changed counts or metrics. Verification returns three distinct sections: `reported_manuscript_values`, `direct_count_arithmetic`, and `approved_historical_display_differences`. `--strict-paper-display` passes for these exact approved entries; `--strict-direct-rounding` remains a separate diagnostic and fails because direct rounding does not produce those displays. Unknown differences, changed counts, or altered arithmetic claims fail. No tolerance or substitution is silently added.

Other limits: no full-cohort reproduction or clean-environment reader run; original CE run snapshot9db50ded… and epoch58checkpoint1a7a640d… are preserved, but its predecessor, exact May code and stochastic continuation remain incomplete. It completed 127 epochs and logged a partial 128; this is not proven early-stop termination. Generic legacy selector ties differ, so paper commands use manifest pins. LISTER stays113,SVTRv2 stays156. IGTR's recorded device-dependent historical F1 caveat remains; mismatches are reported without adjusting selections or results.

Permissions, final licensing/coauthor approval and existing restricted Git history remain release gates. This candidate tree does not sanitize other branches or PR refs. **X3 remains deferred**, with no related investigation, flag change or manuscript edit in this implementation phase.
