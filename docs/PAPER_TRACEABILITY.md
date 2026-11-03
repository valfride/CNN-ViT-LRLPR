# Bidirectional paper–code mapping

Previously reviewed paper snapshot (historical, not assumed latest): `valfride/CNN-ViT-LRLPR-paper@b0b0e947ae7bf30cfd5f340ba3c1f240515112b6`. The source-only candidate starts from mainc739714 and selectively copies reviewed implementation/config contents from report heada0f9844. It does not merge that history or copy its results tree.

| Paper / reviewer response | Implementation | Original identity | Entry point / expected output |
| --- | --- | --- | --- |
| III-A/B architecture,SFB;R1.3/R2.5 | `models/custom/VSR_curvature_att.py` | Original CNN–ViT config9db50ded…/EMAep58 | Table1;seven-slot native outputs |
| III-C decoder;R1.3/R2.5 | Same file,ViT_CrossAttn_OCR/CustomOCR | 3 layers,12 heads,384 hidden,1536 FFN,7 queries/38 classes | Native decoder;no architecture/interface change |
| III-D cosine;R2.1 | CosineClassifierHead and controlled linear flag | Original default vslinear ablation config | Table5one-factor comparison |
| III-E EMA/IV-B CE training;R2.5 | `train_gan.py`,`train_funcs/train_utils.py` | CE/AdamW/plateau/patience60/warmup5;original continued run | Frozen inference vsseparate prospective training recipe |
| Seven baselines;R1.4/R2.3 | `models/{cppd,igtr,lister,mdiff,ote,svtrv2}/`;ARbridge uses existing encoder/NRTR | Exact8model/config pins inpaper_manifest | Tables1–4;native objectives/decoders retained |
| Table1;R1.4/R2.2 | `test.py`,checkpoint_state_compatibility,runwrapper | 8files,F1/3/5,3ktest | `run ... --table 1`;24records/72 cells |
| Table2;R2.3 | `test.py`,`temporal_votes.py`,ARshared-prefix | Fixed64validation records;PRF in 16 settings | Table2;64 test controls;`bayes` remainsPRF |
| Table3;R1.2/R1.4 | Original cohortbuilder/geometry,strict evaluator,private subgroupjoin | Same padded7023;LISTER113 | Table3;8x5columns andinteger counts;3display discrepancies exposed |
| Table4;R1.1 | Original benchmark_efficiency/benchmark_one,explicit wrapper | Same100tracks,8files,FP16RTX8000 | Table4;parameters/FLOPs/timing/memory;hardware differences disclosed |
| Table5;R2.1/R2.2/R2.4 | Original15configs,default-compatible flags,ablation_manifest,aggregate verifier | 18 selected original states,3 seeds | Table5;54 evaluations,18 mean/std pairs |
| IV-A/B data/metrics/environment;R2.5 | Original datasetreader/wrappers,strict frozen file/runtime checks | Exactsplit/data hashes,firstF,RGB32x96/normalization | Correctcounts,partials,diagnostics;no new split/metric |

Full scientific identifiers are machine-readable in `paper_manifest.json`/`ablation_manifest.json`. Every supported recipe in [COMMAND_RECIPES.md](COMMAND_RECIPES.md) resolves to a table above. Per-model original papers/implementation references and adaptation gaps are in `reference/BASELINE_PROVENANCE.csv`; source transfer/differences are in `reproducibility/source_inventory.json`.

Reference counts and paper displays are separate. The original selected state,configuration,cohort andprocedure define execution; missing original attribution or a printed-rounding discrepancy is reported, never repaired by changing results. Full training/latency equivalence and final publication readiness remain unverified. X3 is deferred; manuscript and response letter are untouched.

The author is revising the paper independently. Do not auto-update this snapshot or claim current manuscript consistency; final reconciliation will use the new paper commit after the author completes revisions. Three Table3 displays are now explicitly approved as reported references, with original arithmetic retained separately.
