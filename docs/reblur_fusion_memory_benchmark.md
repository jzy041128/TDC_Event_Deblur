# Archived REBlur fusion memory comparison

## Scope

Run tools/benchmark_reblur_fusion_memory.py with ordinary python, not torchrun.
The launcher starts eight independent three-rank torchrun subprocesses in
sequence on physical GPUs 0,2,3. Run only after those GPUs are free. It does not
connect to the server or start any job automatically when imported.

The manifest configs/benchmark_reblur_fusion_memory_3gpu.yml selects A gated
4CA, M projected mean, T conditional temporal, L local ST plus CA, all-scale
dual expert, all-scale conditional BiGRU, dynamic filtering and Direct 2CA.
Each source is the exact train_config.yml inside its historical experiment
directory. There is no fallback to a current repository config and no automatic
head/scope/crop change. All eight archives and their logged GPU/batch history
are checked before any GPU worker starts. Missing or incompatible files stop
the launch and must be inspected, not silently replaced.

The Direct 2CA source was trained on five GPUs. Its model/data settings are
retained, but this measurement deliberately uses three GPUs. This is a new
three-GPU resource measurement, not a new three-GPU training score. The other
seven rows use the historical three-GPU configurations. Old coarse/hybrid
architectures are excluded; unused local/GRU scale options in T and dynamic
filtering do not restrict those modules' actual all-scale deployment.

## Measurement protocol

- FP32, batch 1 per rank, train crop 256, 6 bins, TDC5, base_dim=32.
- Three warmup AdamW steps initialize optimizer state and DDP buckets, followed
  by three measured steps including forward/backward/AdamW on a fixed batch.
- Every worker builds fresh weights with the archived seed. It does not load
  or overwrite best/latest checkpoints, train for 200 epochs, or select models.
  Weights do not change these architectures' tensor shapes. The saved config
  snapshot is byte-for-byte equal to the archive, even if an old path section
  mentions resume/pretraining; the memory checker intentionally ignores those.
- Original loss, normalization, readout heads/widths, checkpoint recomputation,
  query chunk and branch attention settings are preserved. The preflight loads
  its fixed training batch with num_workers=0, so step time excludes data I/O
  and is not the historical epoch speed.
- Validation checks the first three original full REBlur frames. The source
  dataset must still contain 486 train and 903 evaluation frames. These checks
  do not establish a maximum over every sample or the whole 200-epoch run.
- Report train_allocated_gib in the PPT: the largest per-rank steady-state
  training peak tracked by the PyTorch CUDA allocator. train_reserved_gib,
  validation peaks and the slowest rank's mean step duration are also saved.
  These are not nvidia-smi total memory and exclude allocations outside the
  PyTorch allocator. Do not mix them with the old single-GPU screenshot values.

## Commands

First validate paths without CUDA or output files:

```bash
python tools/benchmark_reblur_fusion_memory.py \
  --experiments-root /data1/jzy/experiments --dry-run
```

After the GPUs are free, inside tmux:

```bash
python tools/benchmark_reblur_fusion_memory.py \
  --experiments-root /data1/jzy/experiments \
  --devices 0,2,3 \
  --output-dir /data1/jzy/benchmarks/reblur_fusion_memory_3gpu_$(date +%Y%m%d_%H%M%S)
```

Each run creates summary.csv (UTF-8 with BOM), summary.json, unmodified config
snapshots, per-module logs and rank-zero metrics JSON. The launcher accepts a
row only after a zero exit status, successful synchronized shutdown and matching
structured metrics/config hash. Failure, timeout or interruption preserves
completed rows, marks the failed row and stops before starting later modules.
Only the child process group started by this launcher is terminated on timeout
or interruption; no unrelated process is killed. Inspect failures before a new
run. A fixed 1800-second per-module timeout can be changed with
--timeout-seconds. Existing output directories are never overwritten.

## EVRB L full-image evaluation

The configuration already exists:
configs/eval_tdc_evrb_16bin_local_st_2ca_allscales_full.yml. Its model dictionary
matches both L training YAMLs exactly, including all-scale one-head LocalSTRead
followed by E3 CA. Do not use the gated-4CA evaluation YAML for this checkpoint.

After training completes, use the selected best.pth for all 894 official
16-bin test images, with tile 256 and overlap 32. The current five-GPU run is:

```bash
EXP=/data1/jzy/experiments/Seed42_LocalST2CA_EncoderRestormerChannelSA_EVRB_16bin_TDC5_AllScales_5GPU_FromScratch_400ep_20261030_220633
CUDA_VISIBLE_DEVICES=0,2,3,4,5 \
torchrun --standalone --nproc_per_node=5 evaluate_ddp.py \
  --config configs/eval_tdc_evrb_16bin_local_st_2ca_allscales_full.yml \
  --checkpoint "$EXP/best.pth" \
  --full-resolution --tile-size 256 --tile-overlap 32 \
  2>&1 | tee "$EXP/evrb_localst16_best_full_$(date +%Y%m%d_%H%M%S).log"
```

Use set -o pipefail before the pipeline. If only three cards are stable, evaluate
the SAME checkpoint on 0,2,3 with nproc_per_node=3; this changes evaluation
sharding, not training or the test image set. No max-samples limit is used.

## Local verification

On 2026-11-03 the full CPU suite ran 125 tests: 122 passed, three opt-in tests
were skipped. The new tests cover all eight config snapshots, three-rank worker
commands, dry-run behavior, CSV/JSON aggregation, first-failure stopping, wrong
config/world-size/nonfinite-result rejection, structured CUDA-metric output
helpers and synchronized cleanup. Subprocess GPU measurements are mocked in
unit tests. A read-only dry run also passed against all eight actual desktop
experiment archives. No server was contacted and no real GPU result was
generated locally. The existing EVRB evaluation-config compatibility tests pass.
