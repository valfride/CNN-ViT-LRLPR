# Third-party source attribution

The authors approved Apache-2.0 for their original contributions;see [LICENSE](../LICENSE).
Inherited components retain their applicable terms;see [NOTICE](../NOTICE),the
original in-file headers,and [SOURCE_PROVENANCE_LEDGER.csv](../SOURCE_PROVENANCE_LEDGER.csv).
Local modification notices were added without changing the validated executable
implementations. Bibliographic citation does not replace software permission.

| Component | Reference and retained terms | Local integration |
| --- | --- | --- |
| OpenOCR |[Topdu/OpenOCR](https://github.com/Topdu/OpenOCR/tree/1ccfc6ee6161f7133b192e16af3a9265273997ba);[Apache-2.0](OpenOCR.LICENSE)|Native baseline encoders/decoders/targets/losses and shared utilities;local imports,32x96 inputs,LRLPR alphabet and adapters.|
| LISTER |AlibabaResearch/AdvancedLiterateMachinery/OCR/LISTER via OpenOCR;[nested Apache-2.0](LISTER.LICENSE),[root reference](AdvancedLiterateMachinery.LICENSE)|Original Copyright2023Alibaba Group and affiliates retained in `models/lister/lister_decoder.py`.|
| FocalNet/FocalNets |Microsoft/Jianwei Yang;[MIT](FocalNet.LICENSE)|Microsoft Copyright2022and MIT header retained in `models/svtrv2/focalsvtr.py`;OpenOCR/local integration does not override those terms.|
| Restormer-derived blocks |Syed Waqas Zamir and contributors;[MIT](Restormer.LICENSE)|MDTA/GDFN-derived design with directional depthwise convolutions,GroupNorm/FReLU and local spatial-extractor integration;attribution retained in NOTICE.|

[UPSTREAM_SOURCES.json](UPSTREAM_SOURCES.json) pins the reference snapshots whose
license texts/components were inspected. The original copy dates for older local
files are not established;reference similarity is not represented as a claim of
historical byte-for-byte derivation. The AR bridge's pinned architectural recipe
is an OpenOCR reconstruction rather than the unavailable competition code.
Original source notices and the known nested MIT/Apache terms are retained,with
prominent per-file local-modification comments. No external binary distribution is
bundled under these source notices.

[DEPENDENCY_LICENSE_LEDGER.csv](DEPENDENCY_LICENSE_LEDGER.csv) records declared
license metadata for the observed67-distribution dependency closure. Dependencies
are installed separately under their own terms;CUDA-related NVIDIA distributions
are proprietary and are not relicensed by this repository. Installed metadata is
not an exhaustive audit of every dependency's nested notices.

Model weights and datasets have separate permission scopes. Only the original
proposed EMA state is authorized for release;no baseline or additional state is
redistributed. See [MODEL_WEIGHTS.md](../docs/MODEL_WEIGHTS.md).

The provenance ledger’s `sha256_at_review` values are dated review records, including bytes before later attribution headers and release documentation updates; they are not a current whole-tree integrity manifest. `reproducibility/scientific_source_hashes.json` verifies the current scientific implementation and metadata. Original copy revisions remain explicitly unestablished where the evidence does not recover them; pinned reference licenses do not establish those historical copy dates.
