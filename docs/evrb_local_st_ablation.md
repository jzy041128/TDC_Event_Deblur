# EVRB all-scale local spatiotemporal readout plus CA

## Configurations

- Five GPUs: configs/train_tdc_evrb_16bin_local_st_2ca_allscales_5gpu_scratch.yml
- Three GPUs: configs/train_tdc_evrb_16bin_local_st_2ca_allscales_3gpu_scratch.yml
- Shared evaluation: configs/eval_tdc_evrb_16bin_local_st_2ca_allscales_full.yml

Only the experiment name differs between the two training YAMLs. The actual
number of GPUs is set by CUDA_VISIBLE_DEVICES and torchrun, not by the name.
Use physical devices 0,2,3,4,5 for five ranks and 0,2,3 for three ranks; avoid
the previously problematic physical GPU 1. Do not start both runs concurrently.

Both use the original EVRB 16-bin event_voxel files with no alternate cache,
1639 train/894 evaluation samples when the server dataset is complete, TDC5,
base_dim=32 and all-channel encoder SA/fusion CA (four heads). Every stage uses
the same L architecture: a one-head 3x3xT local reader with projected widths
8/16/32, no extra spatial bias, followed by E3 channel CA. Query chunk=512 and
checkpoint recomputation are enabled; full E3 time length 16 reaches all stages.

The other settings match the existing EVRB gated 4CA training protocol: FP32,
AdamW lr=0.0002, seed 42, deterministic=True, 400 epochs from scratch, L1 loss,
same edge_weight setting and event normalization, per-GPU batch 1, train crop
256, validation center crop 256 on all 894 samples every five epochs. Keep
pretrain_model and resume_state null for a new run. No AMP support was added.

## Fairness and fallback

These configurations keep every explicit non-hardware setting the same, but
do not make three- and five-GPU optimization numerically equivalent. Global
batch changes from 5 to 3, and the current DistributedSampler gives 328 vs 547
optimizer steps per epoch (131200 vs 218800 over 400 epochs). Sampling padding
and gradient groupings differ too. There is no gradient accumulation option in
the trainer; adding an ignored YAML field cannot equalize the effective batch.
These are hardware fallback configurations, not a controlled GPU-count ablation.

Try the five-GPU preflight before training. If it fails, retain the complete
error log, stop any surviving processes from that job and verify the intended
GPUs are free before starting the three-GPU preflight. A communication/hardware
failure is not necessarily an out-of-memory error. Fewer cards with batch 1 do
not by themselves reduce each rank's model/activation memory. Do not silently
reduce stage scope, bins, heads or crop on just one configuration.

If switching early after instability, start a fresh three-GPU experiment using
its own YAML rather than mixing five- and three-GPU training in a claimed
single-protocol result. A stable five-GPU run follows the old gated 4CA hardware
protocol; a three-GPU L run needs a matching three-GPU baseline for strict
architecture attribution. Old five-GPU scores remain useful historical context.

## Evaluation

The evaluation YAML's model dictionary exactly equals both training model
dictionaries, including explicit encoder branch overrides and fusion options.
Only the validation crop changes to null. Use all 894 full 640x960 images with
evaluate_ddp.py --full-resolution --tile-size 256 --tile-overlap 32, without a
max-samples cap, and the selected run's best.pth. This keeps the historical EVRB
full-image tiled protocol; it is not a claim of a newly independent test set.
Both configs still select best.pth on cropped evaluations of the existing test
split. Preserve the exact training YAML separately from the evaluation YAML.

## Verification

Run python -m unittest discover -s tests -p test_evrb_local_st_configs.py -v.
The tests cover paired-config equality, historical data/training protocol,
all-stage full-time local readers followed by CA, finite gradients and AdamW
updates, shared evaluation checkpoint loading, and official NPZ crop/full-image
dataset loading. Set TDC_EVRB_LOCAL_ST_FULL_SIZE_TESTS=1 to opt into two CPU
AdamW steps at 256x256 and tiled 260x320 inference with the exact base_dim=32
model. CPU synthetic tests do not measure GPU memory or validate CUDA/NCCL.

Local verification on 2026-10-30: the full CPU suite ran 110 tests (107 passed,
three existing/opt-in tests skipped). The separate opt-in standard-size test
also passed: two 256x256 AdamW steps and tiled 260x320 inference with all
2,980,846 parameters in the exact 16-bin model. No server was contacted or
server training job launched; no GPU memory or communication result is claimed.
