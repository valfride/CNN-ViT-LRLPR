# Working paper reference and approved display policy

The manuscript remains in its separate private repository. The manifest pins `775a045acae01770859a4c51ea2a788dd436c5ed`: active Tables 1–5 were compared read-only with frozen references and all 284 numeric display cells match. Historical `b0b0e94` remains provenance. No manuscript or response-letter files are bundled or edited; future revisions require a fresh explicit comparison. This verifies the supported table layout, not experimental reproduction or arbitrary methodological prose.

The established scientific protocol and frozen original execution records remain fixed: CE/AdamW/ReduceLROnPlateau/patience60/EMA warmup5 and original EMA58 for CNN–ViT; all original baseline files/configs including LISTER113 and SVTRv2 156; original splits, preprocessing, native decoding, metrics and results. X3 remains deferred.

## Author decision recorded on 3 November 2026

Keep these authoritative Table3 reported values in documentation and reference tables:

| Model / metric | Original count | Direct percentage | Direct one decimal | Approved reported value |
| --- | --- | --- | --- | --- |
| CNN–ViT partial≥6 | 5650/7023 | 80.449950… | 80.4% | **80.5%** |
| MDiff4STR partial≥5 | 4751/7023 | 67.649153… | 67.6% | **67.7%** |
| SVTRv2-AR partial≥6 | 4477/7023 | 63.747686… | 63.7% | **63.8%** |

The approval applies to historical reported displays, not to new checkpoint/prediction/metric/experimental-output values. The archived two-decimal displays (80.45,67.65,63.75) remain provenance for the historical display process. The evaluator continues to calculate metrics directly from original counts; no second rounding is added to it.

Verification exposes three independent conclusions:

1. **Reported manuscript values:** consistency with the authoritative reference table, accepting the three exact approved historical displays.
2. **Direct count arithmetic:** unchanged exact counts and calculated percentages, including the fact that direct one-decimal rounding differs in three cells.
3. **Approved historical display differences:** explicit model/metric/count/display records and approval scope. No blanket tolerance is allowed; missing approval, altered counts/claimed arithmetic, duplicate entries or additional unapproved differences fail.

`--strict-paper-display` enforces reported-reference consistency and therefore accepts these disclosed approvals. `--strict-direct-rounding` asks a different mathematical question and intentionally fails on the three differences. Both outputs preserve the differences. Neither result establishes end-to-end model execution or final consistency with the not-yet-reviewed revised manuscript.
