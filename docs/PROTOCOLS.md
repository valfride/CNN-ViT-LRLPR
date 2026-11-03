# Frozen scientific protocols

Scientific authority: the established manuscript protocol together with original execution records. Paper `b0b0e94` is the reviewed historical snapshot, not assumed latest; final review against the new paper commit is pending. Complete identities are in `reproducibility/paper_manifest.json` and `ablation_manifest.json`. No new checkpoint selection, architecture/objective/metric change or experiment is introduced.

## Proposed model and baselines

CNN–ViT: CE, AdamW LR1e-4/weight decay0.001, ReduceLROnPlateau max/factor0.5/patience15/minLR1e-6, training batch64, maximum 1000 epochs/early-stop patience60, EMA warmup5/cycle50. Original EMAep58 is the frozen submitted file; historical execution resumed at 19. A training budget is not an observed training duration. Prospective from-scratch and original trajectory are distinct.

The seven baseline configs retain their native targets, objectives, schedules and decoders. SVTRv2-AR is a controlled OpenOCR SVTRNet/NRTR reconstruction: same LRLPR19k/1ksplit,32x96LR,AdamW1e-4/wd0.05,batch64,100epochs/10warmup OneCycle, no external pretraining. It is not an exact reproduction of the original competition implementation. Preserve all other baseline configuration bytes; do not homogenize schedules. Original sources and adaptation gaps are listed in `reference/BASELINE_PROVENANCE.csv`.

## Tables1 and2

LRLPR-26:19,000 training tracks,1,000 validation,3,000 held-out test. Evaluation takes metadata insertion order, excludes HR and uses firstF LR observations atF1/3/5. Preserve original repeat-last handling for short sequences. Input is RGB32x96bicubic,normalized with mean/std0.5; no validation/test augmentation,TTA orSWA. Dataset digest/phase checks prevent substituting a similarly named split. Exact-match RR requires7/7; partial recognition counts positionally correct slots≥6and≥5. CER/character accuracy are separate diagnostics, not partial recognition.

Checkpoint selection is fixed from validation; paper commands stage exactly the approved file. Table2 has16model/Fsettings and4methods,64validation scores and64 post-selection held-out test controls. PRF was frozen in all 16 settings. Test values do not drive selection. PRF sums per-frame log-probabilities; probability outputs use the original clamp/log treatment, logits use log-softmax. Correlated frames do not establish a calibrated Bayesian posterior. CTC pre-collapse timesteps have no proven cross-frame alignment. AR PRF and probability averaging share one generated prefix; sequence/character voting combines independently decoded strings. Native tie rules remain unchanged.

## Table3

Frozen20%-padded acquisition-style LPLCv2 cohort:7,023 annotations from6,709 images,6,240 Poor and783 Illegible. F1only,same crop per annotation for all 8 models. This is not the full LPLCv2 distribution, older3kstudies or tight-crop control. The original builder's geometry/filtering/order and exact manifest are retained; no new selection. LISTER uses strict ep113 and the final original counts, not old ep103. Group verification privately joins annotation IDs and ground truths and exposes only counts.

See `PAPER_DISPLAY_DISCLOSURES.json` for three paper-vs-direct-rounding differences. The author-approved reported displays remain 80.5%, 67.7% and 63.8%; direct arithmetic is reported separately. No result or metric is changed to hide them.

## Table4

NVIDIA Quadro RTX8000,FP16,100 same test tracks,30 warmups/100 synchronized timing iterations. F1latency is per frame; paper F5 batches five frames and includes PRF fusion. Exclude disk/LMDB I/O,preprocessing,host-to-device transfer and final string conversion. FLOPs count multiply-add as2 and analytically supplement unsupported deformable-convolution arithmetic, excluding bilinear-sampling overhead. Memory is peak allocated GPU memory. Original worker formulas and backend settings are preserved; input cache/IDs remain private. Timing is hardware/runtime-dependent, not an edge-deployment claim.

## Table5

Full,noRestormer,noPixelShuffle,noSFB,linear-head variants at seeds42/123/2026. Architectural flags change only the named component and preserve default checkpoint shapes/interfaces for the submitted model. Full-vs-primary uses independently validation-selected best primary and best EMA states from the same trajectories, not matched-final-epoch or new architecture training. All selected states are pinned by hash; no directory reselection. Report mean and sample standard deviation over three run RRs; no p-values,confidence intervals or significance claim is introduced. Table1's selected single checkpoint need not equal these means.

**X3 is deferred.** Dataset training code and frozen config settings are not changed as part of this work.
