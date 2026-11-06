# Bidirectional paper–code mapping

Reviewed manuscript snapshot: `valfride/CNN-ViT-LRLPR-paper@775a045acae01770859a4c51ea2a788dd436c5ed`. The publication candidate preserves clean public history and selectively incorporates verified source improvements from private main `ceac5f495c498fc29e467cd7f5218072fbff9997`. Historical private commit identifiers are provenance records, not a claim that those commits or their restricted evidence are publicly accessible.

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

Reference counts and paper displays are separate. The original selected state,configuration,cohort andprocedure define execution; missing original attribution or a printed-rounding discrepancy is reported, never repaired by changing results. Full training/latency equivalence remains unverified. X3 is deferred; manuscript and response letter are untouched.

The active Tables 1–5 in the pinned snapshot match all 284 displayed numeric cells. The optional combined-ablation table remains inactive. This does not certify arbitrary later manuscript revisions or methodological prose. Three Table 3 displays remain explicitly approved reported values; integer counts and direct arithmetic are preserved separately. No manuscript or response-letter source is included or modified.
