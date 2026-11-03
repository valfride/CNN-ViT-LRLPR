# Checkpoint distribution scope

Exactly one original state is approved for this release: CNN–ViT’s validation-
selected EMA from epoch58. Original evaluation name:`last.pth`;release asset name:
`ema_weights.pth`. SHA-256:`1a7a640dafc368a8dda9ed6a18239b2d54a1f39e9aa0b581a5a405ec0697b11a`.
Exact size:64,267,208bytes. Naming is an alias only;no conversion,reserialization,
state-key rewriting or checkpoint reselection is performed.

Public source uses Apache-2.0 for original contributions,with retained third-party
terms. The authors have separately confirmed permission covering public
redistribution of this exact trained checkpoint under the applicable data/weight
terms. This confirmation is not inferred from the dataset’s academic-use agreement
or from the source license. No broader downstream weight license is invented or
implied by the root source LICENSE;contact the maintainers regarding additional
reuse rights. Dataset images/annotations remain separately restricted.

The approved asset is to be attached to a GitHub Release,never committed to Git.
Publication requires a streamed upload hash/size check and an independent download
of the actual release asset with the same SHA-256 and size. Until those checks pass,
this staging document does not claim a public release/download exists.

No baseline checkpoints,primary/student states,other seeds,or ablation checkpoints
are approved for distribution here. Their identities remain in the scientific
manifests to document the paper;those manifests are not download promises or
redistribution grants. Eight-model and Table5numerical evaluation require separate
authorized access to missing weights and independently licensed datasets.

The public reader example uses only the original EMA asset:

```bash
python -m reproducibility.infer --checkpoint /private/weights/ema_weights.pth \
  --output /private/NEW-inference-output
```

No image argument means generated synthetic RGB input. For an authorized crop,add
`--image /private/crop.png`. CPU FP32 is explicit;the underlying original bicubic
resize,RGB normalization,strict state loading and native decoder are reused. This
is not a full experiment or the original FP16 efficiency protocol. Predictions are
written only outside the source checkout. Do not add them to Git.
