# Model and dataset access

Only the original CNN–ViT validation-selected EMA epoch58 is authorized for this
repository's model release. Asset publication and independent download verification
are tracked in [MODEL_WEIGHTS.md](MODEL_WEIGHTS.md). No other model state is included.

`reproducibility/paper_manifest.json` retains all8 original model/config/checkpoint
identities;`ablation_manifest.json` retains18original Table5state identities. These
scientific records are not claims that the corresponding binaries are available.
The seven baseline checkpoints,primary states,other seeds and ablation assets are
not released here. A complete eight-model/Table5reader package is unavailable.

The approved CNN–ViT asset alias `ema_weights.pth` is byte-identical to the original
paper-evaluation `last.pth` (EMA58),SHA-256
`1a7a640dafc368a8dda9ed6a18239b2d54a1f39e9aa0b581a5a405ec0697b11a`.
The frozen config SHA-256 is
`9db50ded91aabfff67d2863057e51bd4ef6a8dd8da5ce7a67c61fe3626fd7673`.
Never substitute the terminal EMA state or convert the approved binary.

Obtain datasets independently:

- [LRLPR-26](https://github.com/raysonlaroca/lrlpr-26-dataset):official academic/noncommercial access agreement. Original19k/1k training/validation partition and3000test-track identities matter;do not select a new holdout.
- [LPLCv2](https://github.com/UFPR-IPASP-PR/LPLCv2-Dataset):applicant-only academic access to original images/annotations. The paper's7023annotation padded-crop cohort and frozen metadata/data digests are retained,not redistributed.

No images,annotations,IDs,transcriptions,predictions,LMDBs or input tensors are
included. Model-distribution permission is separate from dataset access and the
source-code license. Preflight hashes trusted artifacts without loading them;
explicit evaluation uses the checked original reader/native decoders. Raw outputs
must stay in new private directories outside the source checkout.

This repository has independent clean history. It does not import research
branches,PRs,release assets or restricted historical records. See
[protocols](PROTOCOLS.md) and [limitations](REPRODUCIBILITY.md#known-limitations).
