# REBlur scratch fusion and temporal-convolution ablations

Three new runs share 6-bin H5 input, seed 42, five GPUs, per-GPU batch size 1,
200 epochs, AdamW learning rate 0.0002, and validation every five epochs.
Training uses 256-pixel random crops. Validation uses all 903 complete 260x320
images, without tiling. Both pretrain_model and resume_state start null.

| Config | Fusion | Event3D convolution |
| --- | --- | --- |
| train_tdc_reblur_direct2ca_scratch.yml | Direct 2CA | TDC |
| train_tdc_reblur_rgb_gated4ca_scratch.yml | RGB-gated 4CA | TDC |
| train_tdc_reblur_rgb_gated4ca_conv3d_scratch.yml | RGB-gated 4CA | Conv3d |

The archived original four-CA scratch run completed 200 epochs despite its
600ep directory name. Check its historical implementation before treating it
as a definitive matched baseline; it can serve as the initial reference.

## Convolution switch

`model.event3d_conv_type` accepts `tdc` (default) or `conv3d`.
The stem and both spatial-downsampling blocks use the selected computation.
In each block, only the first (5,3,3) convolution changes. TDC convolves with
the existing temporal-difference weights; Conv3d uses the raw weights directly.
Both keep bias=False, temporal padding 2 and temporal stride 1. The later
(1,3,3) spatial convolution, activations and all 3D self-attention layers stay
unchanged. No event preprocessing is required.

Modes have identical registered parameters and seeded initial raw weights,
but different effective kernels and parameter constraints. Train each mode
separately; changing the switch on an already trained checkpoint is not this
ablation. Omitted switches preserve the original TDC behavior and state keys.

New checkpoints record event3d_conv_type. Training preloads, resumes and
evaluate_ddp reject mismatched convolution types. Old checkpoints without this
field are treated as TDC. Use the matching training YAML for optional final
evaluation with --full-resolution and without tile-size; its val split already
uses full images.

## First run

```bash
tmux new -s reblur_rgb_gate4ca_tdc
conda activate tdc_deblur
cd ~/projects/TDC_Event_Deblur
CUDA_VISIBLE_DEVICES=0,2,3,4,5 \
torchrun --standalone --nproc_per_node=5 train_ddp.py \
  --config configs/train_tdc_reblur_rgb_gated4ca_scratch.yml
```

Do not use the old GoPro fine-tuning checkpoints for these scratch runs. Keep
the same validation/checkpoint-selection protocol for all compared models.
