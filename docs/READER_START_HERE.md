# Reader workflow and current access limits

This clean source tree preserves the original scientific protocol. Only the proposed
EMA epoch58 is authorized for its model release;see [MODEL_WEIGHTS.md](MODEL_WEIGHTS.md)
for the actual upload/download verification status. Baseline and ablation weights
are not distributed. The active Tables 1–5 match the pinned `775a045` manuscript snapshot; later revisions require a new comparison.

## Understand and cite the implementation

The proposed CNN–ViT uses a topology-aware spatial extractor with Restormer-derived
MDTA/GDFN blocks, deformable projection, PixelShuffle and Surgical Focus Block
(SFB), followed by a fixed seven-query cross-attention recognizer and cosine
classifier. The frozen configuration has three decoder layers, twelve attention
heads,384 hidden dimensions and1536 feed-forward dimensions. Read
[PAPER_TRACEABILITY.md](PAPER_TRACEABILITY.md) and the original model source for
exact execution details. Training uses CE, AdamW and ReduceLROnPlateau; the paper
model is the validation-selected EMA epoch58, not the terminal EMA state.

The [seven-baseline reference table](../reproducibility/reference/BASELINE_PROVENANCE.csv)
links each original paper, implementation project and local adaptation. The
OpenOCR encoder/decoder reconstruction of SVTRv2-AR is not the unavailable
competition implementation. Native CTC, neighbor, one-token, parallel,
instruction-guided, diffusion and autoregressive decoders remain distinct.
Cite the original method papers as well as the companion manuscript; software
provenance and remaining source rights are documented in
[third-party attribution](../third_party/README.md).

## Install without changing another environment

Use a fresh Python3.9.21 Linux x86_64 environment. The observed dependency closure
is constrained; this is not a certified clean-install lock for every platform.
Install the CUDA12.4 PyTorch wheels separately, then the remaining requirements:

```bash
python -m pip install -c requirements/constraints-paper-py39.txt \
  torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -c requirements/constraints-paper-py39.txt -r requirements.txt
python -m pip check
python -m unittest reproducibility.test_protocol reproducibility.test_smoke reproducibility.test_release -v
```

No source or CUDA runtime is vendored. GPU drivers and CUDA distribution terms
remain separate. Installing requirements alone in an empty environment can select
newer, unvalidated torch/torchvision and transitive packages. The verified
installation also contains both OpenCV distributions at the same version; avoid
assuming their shared `cv2` namespace is conflict-free on a fresh installation.
An isolated reader import check against the existing installation is distinct
from a clean installation and from GPU execution.

## Obtain authorized artifacts

Request [LRLPR-26](https://github.com/raysonlaroca/lrlpr-26-dataset) access through
its official academic/noncommercial agreement. The original19k/1k training/
validation partition must match the frozen dataset metadata; do not generate a
new holdout. The held-out test has3000 tracks, five LR observations per track.
Request [LPLCv2](https://github.com/UFPR-IPASP-PR/LPLCv2-Dataset) independently;
its published terms restrict samples to the applicant and academic research.
The paper uses7023 selected annotations from6709 source images, not the entire
legibility-classification dataset or its native cross-validation experiment.
Do not redistribute images, annotations, sample identities, transcriptions or
prediction records through this repository.

The [checkpoint release manifest](../CHECKPOINT_RELEASE_MANIFEST.json) preserves
scientific identities and the single-state publication scope. It does not advertise
missing baseline or ablation weights. No baseline/student/other-seed states are
authorized for distribution. Copy the [private mapping template](../reproducibility/weights.example.json)
outside the checkout only when independently authorized original files are available.
Never convert,reserialize or reselect weights to pass a guard.

## Single-image, temporal and table verification

The proposed model needs only its original EMA file for a synthetic CPU F1example:

```bash
python -m reproducibility.infer --checkpoint /private/weights/ema_weights.pth \
  --output /private/NEW-inference-output
```

Add `--image /private/authorized-crop.png` for an authorized crop. The existing
[bounded GPU matrix](COMMAND_RECIPES.md#bounded-synthetic-compatibility-smoke-test-not-table-reproduction)
requires independently authorized additional weights and an explicitly verified
physical GPU0/RTX8000. It is not the default reader example and must not be
repeated merely to check installation. [GPU_SMOKE_VALIDATION.md](GPU_SMOKE_VALIDATION.md)
records the historical successful campaign. No GPU fallback is hidden in the CPU example.

PRF uses the existing `--fusion bayes` interface. The alternatives are `average`,
`majority` and `char_majority`. CTC fusion before collapse remains a heuristic,
without an alignment guarantee; SVTRv2-AR score fusion uses a shared prefix,
whereas voting uses independently decoded strings.

[COMMAND_RECIPES.md](COMMAND_RECIPES.md) maps Tables1–5 to exact commands,
fingerprints and aggregate outputs. Full evaluation needs independently licensed
data and frozen cohort identities. Table4 is the original FP16 RTX8000 benchmark;
Table5 has18 selected states across three seeds, not six newly trained models.
Prospective from-scratch training is separate and cannot reconstruct the missing
original continuation predecessor/RNG lineage.

Check existing references without model loading or experiments:

```bash
python -m reproducibility.verify --references --strict-paper-display
python -m reproducibility.run plan --table 1
python -m reproducibility.run plan --table 5
```

The reported Table3 values80.5%,67.7%,63.8% remain authoritative displays. Exact
integer counts and direct arithmetic stay separate; optional
`--strict-direct-rounding` intentionally exposes the three historical differences.
