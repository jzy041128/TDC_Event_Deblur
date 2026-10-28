# REBlur temporal fusion ablation

## Scope

Five optional fusion modes replace the per-stage fusion only. All legacy modes,
state-dict paths and forward formulas remain available. New readers receive E3
as B,C,T,H,W; legacy modes still average E3 over T. The three encoders, TDC,
encoder SA, decoder, loss and event preprocessing are unchanged.

The common outer norm_ffn/ffn/gamma_ffn runs exactly once per stage. Intermediate
event memories do not overwrite the E2/E3 encoder streams at the next stage.
fusion_dim=2d specifies the output, not the input E3 dimensionality.

## New experiment decisions (2026-10-28)

- Unless the user explicitly requests otherwise, future fusion experiments must
  use the intended fusion/reader consistently at all three stages: 0, 1 and 2.
  Do not silently substitute another fusion at finer stages to save resources.
  Measure memory first; report a resource limit before proposing a scope change.
- The agreed mean baseline is projected uniform temporal pooling:
  M = out_proj(v_proj(norm_v(E3)).mean(dim=2)). Keep the same value normalization,
  value projection, output projection and downstream CA/Inject/gate/residual/FFN
  structure as the conditional temporal experiment. Replace adaptive temporal
  weights by 1/T; do not substitute raw E3.mean(dim=2) or the legacy Direct 2CA.
- These decisions are implemented by the matched readout configurations below.
  Preserve historical configurations and checkpoints. The earlier coarse/hybrid
  table records historical experiments, not the default for new experiments.

## Matched all-scale readout study

All three configurations use fusion_mode=cond_temporal_2ca, with an explicit
fusion_options.e3_readout switch. Only the experiment name and this switch
differ between their YAML files. Each selected reader runs at stages 0, 1 and 2;
no stage falls back to another reader. The common RGB/E2 CA, subsequent E3 CA,
Inject convolutions, gate, residuals and outer FFN are identical.

| e3_readout | Config under configs/ | Reader |
| --- | --- | --- |
| mean | train_tdc_reblur_mean_v_2ca_allscales_3gpu_scratch.yml | UniformTemporalRead: mean of normalized/projected V, followed by out_proj |
| temporal | train_tdc_reblur_temporal_2ca_allscales_3gpu_scratch.yml | Original single-head ConditionalTemporalRead |
| local | train_tdc_reblur_local_st_2ca_allscales_3gpu_scratch.yml | Single-head LocalSTRead, 3x3 neighborhood, followed by the same E3 CA |

- Q/K/V widths are 8/16/32 at C=32/64/128. The mean reader keeps V/out_proj only;
  it neither calculates Q/K nor registers unused Q/K or time-bias parameters.
  This is a pooling-module ablation, not a parameter-matched proof that RGB/E2
  guidance alone is responsible for a gain.
- Temporal and local readers use one head and the same normalization, Q/K/V,
  output projections and learned time-bias MLP. The new local reader has no
  additional learned spatial bias, so enlarging 1x1 to 3x3 adds candidates rather
  than another positional parameter family. Existing four-head local modes keep
  their original learned spatial bias and formulas.
- At base_dim=32/event_in=6 the parameter counts are mean=2,960,299 and
  temporal=local=2,977,966. These are parameter counts, not GPU-memory estimates.
- Constructor-only temporary initialization consumes the original temporal
  reader's RNG draws for the mean variant, then retains just norm_v/v_proj/
  out_proj. Thus every retained tensor, downstream gate, encoder, decoder and
  common FFN starts identically across the three variants. No trained weights
  are used for initialization. Temporal and local variants have the same state
  tensor names/shapes and initialization; checkpoint metadata still rejects
  loading/resuming one as the other.
- The new temporal YAML is numerically identical to the existing scheme-1 YAML
  for initialization, forward output and gradients. Its existing training result
  can be reused with its ORIGINAL YAML/checkpoint. Do not load an old checkpoint
  with the new YAML: options differ, and the safety guard correctly rejects it.
- LocalSTRead(kernel_size=1, num_heads=1, spatial_bias=False) is tested against
  the original temporal reader for outputs and gradients. The 3x3 implementation
  is tested against a padded/unfold reference, including invalid border masking.
- Historical local/dual-expert/GRU modes now also accept explicit stage lists
  such as [0,1,2]. Their legacy default [2] and original YAML files are preserved
  for checkpoint compatibility. No additional dual-expert/GRU training YAMLs are
  added in this change; those architecture experiments remain deferred.
- Keep FP32, three GPUs 0,2,3, per-GPU batch 1, seed 42, 200 epochs and the existing
  REBlur data/loss/evaluation protocol. Single-head applies to the reader only;
  channel CA and encoder SA still use four heads. Do not switch to AMP or alter
  the batch/crop on just one variant to fit memory.

Run the mean baseline first, then local, after their real three-GPU preflights:

```bash
tmux new -As reblur_meanv_allscales_3gpu
conda activate tdc_deblur
cd ~/projects/TDC_Event_Deblur
set -o pipefail
CFG=configs/train_tdc_reblur_mean_v_2ca_allscales_3gpu_scratch.yml
if CUDA_VISIBLE_DEVICES=0,2,3 torchrun --standalone --nproc_per_node=3 \
  tools/check_attention_memory.py --config "$CFG" \
  2>&1 | tee "reblur_meanv_preflight_$(date +%Y%m%d_%H%M%S).log"
then
  CUDA_VISIBLE_DEVICES=0,2,3 torchrun --standalone --nproc_per_node=3 \
    train_ddp.py --config "$CFG" \
    2>&1 | tee "reblur_meanv_train_$(date +%Y%m%d_%H%M%S).log"
else
  echo "Preflight failed; training was not started."
fi
```

If already inside tmux, skip tmux new rather than creating a nested session.
The memory checker does not write model/config files and its smoke test does
not guarantee every later dataset sample fits. CPU full-size tests are not GPU
memory measurements. Keep the full training config for future resume/evaluation.

```bash
python -m unittest discover -s tests -p test_matched_e3_readout.py -v
TDC_MATCHED_FULL_SIZE_TESTS=1 python -m unittest discover -s tests \
  -p test_matched_e3_readout.py -k full_resolution -v
```

## Experiments and order

| Order | fusion_mode | Stages 0 / 1 / 2 |
| --- | --- | --- |
| 1 | cond_temporal_2ca | conditional temporal / conditional temporal / conditional temporal |
| 2 | anchored_local_st | conditional temporal / conditional temporal / local ST |
| 3 | identity_preserving_dual_expert | temporal expert / temporal expert / local ST expert |
| 4 | event_controlled_dynamic_filter | dynamic filter / dynamic filter / dynamic filter |
| 5 | conditioned_bi_gru | conditional temporal / conditional temporal / bidirectional GRU |

Config names follow configs/train_tdc_reblur_<fusion_mode>_3gpu_scratch.yml.
No new config runs automatically or loads GoPro weights.

Parameter counts at base_dim=32, event_in=6 (not memory measurements): A
3,510,868; conditional temporal 2,977,966; local ST coarse 2,911,745; dual expert
2,905,980; dynamic filter 2,892,326; bidirectional GRU coarse 3,658,332.

## Fixed comparison protocol

The reference is configs/train_tdc_reblur_rgb_gated4ca_channel_3gpu_scratch.yml
(the existing A experiment), not the RGB/E2 spatial-SA B experiment.

- REBlur train/test paths, 6 bins, 5-tap TDC, base_dim=32.
- All three encoder SA branches and ordinary fusion CAs remain channel-based.
- Physical GPUs 0,2,3; one image per GPU; global batch 3.
- From scratch, AdamW at 0.0002, 200 epochs, FP32, seed 42, deterministic=True.
- Same loss, event normalization, train crop 256, full-frame validation.
- Validation every five epochs; best PSNR selects best.pth; latest every epoch.
- Preserve the existing 903-frame evaluation protocol for this screening study.
  It is model-selection data, not a newly held-out independent test set.
- Report best PSNR/epoch/SSIM, final PSNR, parameter count, step time and per-GPU
  memory. These are architecture comparisons, not parameter-matched ablations.
- Do not selectively extend only the winning model's budget for the main table.
  For small gains, repeat both reference and selected models at additional seeds.

### Shared initialization and reuse of A

New modules initialize inside a forked CPU RNG context. Each new stage then
constructs an untrained legacy RGB-gated 4CA reference stage to consume exactly
the legacy RNG sequence, reusing only its common FFN parameters. The temporary
legacy attention modules are discarded, not registered, optimized or run.

This preserves all non-fusion weights, common fusion FFNs and the RNG state after
construction against A under the same seed/config. It does not use trained A or
GoPro parameters. Encoder/decoder initialization does not depend on which new
fusion was selected. The existing A training result can be reused; rerunning A
training is not required just because the optional modes were added.

The existing DDP sampler uses a separate seed and set_epoch. Dataset crops use
seed + epoch * sample_count + index. Neither depends on new module RNG draws.

## Operators

ConditionalTemporalRead normalizes each query source independently, projects Q,
K and V to d=max(C/4,8), and softmaxes only along T at each spatial location.
Sources are X,E2 except for the independent expert, which uses original R only.
Q/K are not spatially L2-normalized. The content dot product uses sqrt(d) scaling.
Time bias uses [tau,tau^2] through an MLP; its last layer starts at zero. T=1 uses
tau=0. Attention scores and softmax explicitly disable autocast and use FP32.

The historical LocalSTRead config searches a 3x3 neighborhood across all T with
four heads. Total width is max(C/4,8), not that width per head. For C=32/64/128 the
total widths are 8/16/32 and head widths 2/4/8. Invalid boundary candidates are
masked before softmax. Chunk size is 512 query positions; gather happens INSIDE
each non-reentrant checkpointed chunk during training. Cached indices/masks
are graph-free and bounded to the latest size/device. There is no global THW
attention, learned offset, optical-flow estimation or full-resolution local ST.

SpatialGate uses independent source norms, 1x1 reduction, 3x3 depthwise conv,
GELU and 1x1 output. Independent sigmoid gates start at 0.5, not the old
2*sigmoid exchange gain of 1. The initial zero output layer can give gate-body
zero gradients on the first step; two-step gradient checks are included.

The dynamic filter uses four groups, row-major 3x3 offsets and replicate padding.
Its signed residual kernel is (tanh(logits)-mean(tanh(logits)))/9. The operator
annihilates spatial constants, including borders, up to float rounding; the
subsequent biased Inject convolution need not output zero for that input.
Nine offset multiplications avoid a permanently unfolded feature volume, but
their autograd state is not free. Kernel-head output weights start N(0,0.02).

The historical GRU config runs only at stage 2 with hidden width C/2. Forward/backward cells are
independent. RGB/E2 context conditions initialization and every update. Original
time coordinates are retained in reverse scanning. There is no detach/truncated
BPTT and no extra final spatial gate in this mode.

Dual experts use original R for both queries inside each stage. They do not
first update R with E2 before querying E3. Two independent gates need not sum to
one. This is within-stage separation: later-stage R already includes preceding
fusion outputs, so it is not global statistical independence or disentanglement.

## Checkpoints and configuration safety

Unknown fusion_options and unsupported kernels/scale indices fail early.
Explicit nonempty, unique stage lists within [0,1,2] are supported for the
historical local ST and GRU modes; their old YAMLs retain [2]. C/groups and
projection-width/heads divisibility are checked. E3 supports runtime T>=1, but
the whole model still needs event_in matching the input E2 channels.

New checkpoints include original/resolved model config, fusion mode/options,
event_in, base_dim, TDC kernel and FP32 metadata. Old modes load old checkpoints
strictly as before. New same-mode resume/pretraining also uses strict loading;
mismatched mode/options are rejected even when tensor shapes happen to agree.

Optional future backbone initialization requires BOTH:

```yaml
path:
  pretrain_model: /path/to/source/best.pth
  pretrain_backbone_only: true
  resume_state: null
```

This drops ALL fusions.* source weights, requires exact keys/shapes everywhere
else, and starts a new optimizer/epoch 0. It never turns resume into partial
loading and cannot bypass the TDC-versus-Conv3d guard. The five supplied configs
do not enable this feature.

## Known compatibility limitations and reminder triggers

Recorded on 2026-10-26. The user is already training scheme 1 in FP32 and asks
to retain these reminders rather than change the current experiment.

1. AMP is not fully supported yet. CPU BF16 autocast reproduces a dtype mismatch
   in ConditionalTemporalRead: scores disable autocast but TemporalPositionBias
   receives low-precision q while its MLP weights remain FP32. The current FP32
   training path is unaffected. Before suggesting AMP, FP16, BF16, GradScaler or
   a precision change to save memory, raise this limitation and add full-model
   mixed-precision forward/backward tests, including residual dtype promotion.
   Do not claim full AMP support after only changing the position-bias input.

2. Checkpoint metadata compares some raw constructor settings rather than fully
   resolved semantics. Explicit per-branch restormer_channel overrides and omitted
   overrides inheriting the same global setting produce identical predictions,
   but metadata validation rejects loading between them. The exact original
   model YAML works for resume/evaluation. In-memory training validation does
   not reload a checkpoint and is unaffected. Before creating/simplifying an
   evaluation config, removing branch overrides, changing inherited defaults or
   diagnosing a checkpoint-setting mismatch, raise this limitation. A future
   fix should canonicalize actual resolved settings while retaining strict
   rejection of genuine architecture/bin/kernel/option mismatches.

Neither issue is evidence that the current FP32 model formula, gradients or
scores are wrong. No network/precision/config change was made for this reminder.

## Local verification

```bash
python -m unittest discover -s tests -q
TDC_FULL_SIZE_TESTS=1 python -m unittest discover -s tests \
  -p test_temporal_fusion_integration.py -k full_resolution -v
```

Coverage includes C=32/64/128, T=1/6/16, singleton/odd spatial boundaries,
time softmax, constant-response/grouped filtering, two-step gate gradients,
local slow reference and gradients, chunk/checkpoint equivalence, GRU direction
independence, per-scale routing, FFN exactly once, 5D E3 hooks, checkpoint guards,
shared initialization/RNG and old mean paths. Full-size tests use CPU synthetic
inputs, not server datasets; they do not measure GPU memory or validate NCCL.

A separate comparison against pre-edit HEAD verified all 40 legacy fusion/SA
combinations: state keys, initial tensors, strict loading and eval predictions
were bitwise equal.

Local CPU verification passed 87 regular tests plus the opt-in full-size test:
all five modes completed two AdamW steps at base_dim=32, T=6, 256x256, followed
by 260x320 full-frame inference. CUDA, NCCL/DDP training and actual GPU memory
remain server preflight items; these CPU results are not GPU measurements.

The 2026-10-28 matched-readout update passed the complete 102-test discovery run
(100 passed, two opt-in tests skipped). Its separate full-size CPU test passed
two AdamW steps at 256x256 and inference at 260x320 for mean, temporal and local.
A comparison with pre-update HEAD also verified bitwise-equal initialized state,
forward output and parameter gradients for all five historical fusion modes.
The old desktop checkpoint folders were no longer present during this update,
so trained-checkpoint compatibility was not rechecked against those files;
strict loading and option rejection were covered by generated-checkpoint tests.

## Server preflight and first training

Run these yourself on the server after pulling; no server actions are performed
by the local coding agent. A preflight writes no checkpoint or config.

```bash
conda activate tdc_deblur
cd ~/projects/TDC_Event_Deblur
CFG=configs/train_tdc_reblur_cond_temporal_2ca_3gpu_scratch.yml
CUDA_VISIBLE_DEVICES=0,2,3 torchrun --standalone --nproc_per_node=3 \
  tools/check_attention_memory.py --config "$CFG"
```

The tool warms up three AdamW steps and measures three complete training steps,
including DDP when launched with torchrun. It reports per-process allocated and
reserved GiB, cross-rank maxima and synchronized step time, then checks the first
three full validation frames. It uses a repeated real training batch for stable
measurement; this is a smoke test, not training-quality evaluation or a guarantee
that every later image/shape fits. Keep sufficient VRAM headroom.

If preflight succeeds, start the first experiment from a fresh SSH terminal:

Success requires both the final `Preflight PASS` marker and a normal torchrun
exit without NCCL/CUDA errors. Memory numbers printed before an error do not
constitute a complete pass. The tool synchronizes every rank after the memory
all-reduce, then barriers and synchronizes before process-group destruction.
This removes an unchecked shutdown race; a remaining NCCL error still requires
communication/device diagnostics and must not be dismissed as harmless.

```bash
tmux new -As reblur_temporal1_3gpu
conda activate tdc_deblur
cd ~/projects/TDC_Event_Deblur
set -o pipefail
CFG=configs/train_tdc_reblur_cond_temporal_2ca_3gpu_scratch.yml
CUDA_VISIBLE_DEVICES=0,2,3 torchrun --standalone --nproc_per_node=3 train_ddp.py \
  --config "$CFG" 2>&1 | tee "reblur_temporal1_$(date +%Y%m%d_%H%M%S).log"
```

Do not create nested tmux sessions. For interruption, set the SAME config's
path.resume_state to that experiment's latest.pth and keep pretrain_model null.
Do not change fusion options during resume. An evaluation config is not needed
for REBlur's standard full-frame result: the training validation is full-frame.
