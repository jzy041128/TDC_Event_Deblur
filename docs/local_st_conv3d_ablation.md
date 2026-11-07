# All-scale local ST readout plus CA with plain Conv3d

## Scope

The existing implementation already supports `event3d_conv_type: conv3d` for
all E3 stages. No network or trainer rewrite is needed. Keep the TDC path and
all old configurations unchanged. This group has EVRB, REBlur, GoPro, and REVD
configs. All use the same fusion mechanism and ordinary Conv3d at each scale.

- Train: configs/train_tdc_evrb_16bin_local_st_2ca_conv3d5_allscales_5gpu_scratch.yml
- Evaluate: configs/eval_tdc_evrb_16bin_local_st_2ca_conv3d5_allscales_full.yml

The sole computational change from the existing five-GPU EVRB L control is
`model.event3d_conv_type: tdc` to `conv3d`. Experiment name and output root also
change for bookkeeping. The legacy field `tdc_kernel_size: 5` still specifies
the temporal kernel length for either convolution mode; it does not enable TDC.

Each of the three E3 blocks becomes plain Conv3d(5,3,3), LeakyReLU, plain
Conv3d(1,3,3), LeakyReLU. The first convolution uses padding (2,1,1), no bias,
and temporal stride 1. Its spatial stride is 1 at the stem and 2 at each down
stage. All 16 time positions are preserved. The spatial-only second convolution,
encoder channel SA, all three local readers followed by E3 CA, decoder,
normalization, loss and training schedule are unchanged. The local reader is
one-head 3x3xT at every scale, widths 8/16/32, query chunk 512 and checkpoint
recomputation enabled, with no added spatial bias.

At base_dim=32 the EVRB model still has 2,980,846 nominal trainable parameters.
The raw initial tensors match the TDC model at seed 42; their convolution
operators differ because only the TDC mode transforms temporal kernel weights.
This is the same ordinary-convolution replacement used by the earlier REBlur
gated-4CA convolution ablation, now combined with L instead of gated 4CA.

## EVRB Protocol

Use official 16-bin voxels, train/test counts 1639/894 when the server data is
complete, five GPUs 0,2,3,4,5, per-GPU batch 1, global batch 5, FP32, AdamW
lr=0.0002, seed 42, 400 epochs from scratch and validation every five epochs.
Train with 256x256 random crops and validate the same 256x256 center crops used
by the existing L+TDC run. Do not initialize or resume this new ablation from
TDC weights. Both pretrain_model and resume_state start as null.

The existing L+TDC five-GPU 400-epoch EVRB run is the direct control. The new
run isolates the convolution change under the same protocol; comparisons
against other fusion architectures additionally change the fusion method.
Do not switch GPU count or batch size halfway through a claimed matched run.
The selected checkpoint remains chosen using the existing test split, as in
the historical protocol; this change does not introduce an independent split.

Run tools/check_attention_memory.py with the training YAML under the same five
ranks before launching train_ddp.py. Keep a complete log. Ordinary Conv3d does
not remove the 16-bin feature volumes or local ST attention work, so substantial
memory or speed reductions are not guaranteed. CPU checks are not a CUDA or
NCCL stability test.

Full-image evaluation uses the matching new evaluation YAML, best.pth, all 894
images and evaluate_ddp.py --full-resolution --tile-size 256 --tile-overlap 32.
The train/evaluation model dictionaries are exactly equal, including explicit
branch SA overrides and fusion_options, for strict checkpoint compatibility.

## Follow-up Runs

1. REBlur: use `configs/train_tdc_reblur_6bin_local_st_2ca_conv3d5_allscales_3gpu_scratch.yml`
   with physical GPUs 0,2,3. It matches the existing L+TDC REBlur configuration
   except for convolution mode, name, and output root. Use 6 bins, global batch 3,
   200 epochs from scratch, seed 42, and all 903 complete validation frames
   every five epochs. A separate full-image evaluation YAML is unnecessary.
2. GoPro: use `configs/train_tdc_gopro_6bin_local_st_2ca_conv3d5_allscales_5gpu_scratch.yml`.
   Keep the prior GoPro six-bin server voxels, five ranks, per-rank batch 1,
   600 epochs, lr=0.0002, 256x256 crops, and validation every epoch. The paired
   `configs/eval_tdc_gopro_6bin_local_st_2ca_conv3d5_allscales_full.yml`
   disables the validation crop without changing the model.
3. REVD: use `configs/train_tdc_revd_6bin_local_st_2ca_conv3d5_allscales_5gpu_finetune.yml`
   only after GoPro L+Conv3d is trained. Set path.pretrain_model to that run's
   actual best.pth and leave path.resume_state null. The template deliberately
   names a nonexistent checkpoint until filled in, so it cannot silently run
   from scratch. Keep the existing six-bin trilinear caches, five ranks,
   200 epochs, lr=0.00002, and validation every five epochs. The matching
   `configs/eval_tdc_revd_6bin_local_st_2ca_conv3d5_allscales_full.yml`
   evaluates the uncropped test set. GoPro and REVD model dictionaries match
   exactly so full-weight initialization can use strict loading.

REBlur and EVRB already have matched L+TDC controls. The earlier GoPro and REVD
4CA/2CA runs use different fusion architectures and are historical references,
not controls that isolate TDC versus Conv3d. This follow-up does not add new
GoPro L+TDC or REVD L+TDC training runs.

## Grouped Experiment Output

The new YAML sets:

```yaml
path:
  experiments_root: /data1/jzy/experiments/LocalST2CA_Conv3d5_AllScales
```

The current trainer already creates missing parent directories recursively and
adds an independent timestamped run folder. No new path option is introduced.
Future dataset YAMLs for this same architecture should use this same group root
and distinct dataset/bin/budget/seed experiment names, for example:

```text
/data1/jzy/experiments/
  LocalST2CA_Conv3d5_AllScales/
    launch_configs/
    Seed42_LocalST2CA_..._EVRB_16bin_Conv3d5_..._400ep_TIMESTAMP/
      train_log.txt
      latest.pth
      best.pth
    Seed42_LocalST2CA_..._REBlur_..._TIMESTAMP/
    Seed42_LocalST2CA_..._GoPro_..._TIMESTAMP/
    Seed42_LocalST2CA_..._REVD_..._TIMESTAMP/
```

The REBlur/GoPro/REVD lines describe the layout of future run directories.
Old experiments are neither moved nor renamed. In particular, old absolute
resume and pretraining paths continue to work. On resume, the existing trainer
uses the directory containing latest.pth, irrespective of experiments_root.

The trainer does not automatically snapshot the YAML. Preserve a copy before
launching so later local edits do not lose the actual run settings:

```bash
GROUP=/data1/jzy/experiments/LocalST2CA_Conv3d5_AllScales
mkdir -p "$GROUP/launch_configs"
CFG="$GROUP/launch_configs/EVRB_16bin_Conv3d5_5gpu_$(date +%Y%m%d_%H%M%S).yml"
cp configs/train_tdc_evrb_16bin_local_st_2ca_conv3d5_allscales_5gpu_scratch.yml "$CFG"
```

Use that same `$CFG` for the preflight and training launch. After obtaining the
printed experiment directory, the same file can be copied to that run as
train_config.yml. Keep logs and evaluation outputs in their own run directory.
Do not put multiple runs' best.pth/latest.pth directly in the group root.

## Local Tests

Run `python -m unittest discover -s tests -p test_local_st_conv3d_configs.py -v`.
Tests cover exact configuration parity with L+TDC, matching evaluation metadata,
existing trainer grouping and resume behavior, plain convolution in every E3
block, all-scale full-time local-read/CA hooks and finite AdamW updates for
6/16 bins, parameter counts and seeded initial weights, and Conv3d checkpoint
roundtrip plus tiled evaluation and rejection of TDC checkpoints.
The follow-up tests also compare the REBlur configuration against the existing
matched control, keep GoPro/REVD dataset protocols unchanged, check both full
image configurations, and load a matching GoPro checkpoint into the REVD model.
