# REBlur three-GPU attention experiments

All groups use six bins, TDC kernel 5, RGB-gated four-path fusion,
three physical GPUs (0, 2, 3), batch size 1 per GPU, seed 42,
200 epochs from scratch, and validation on all 903 complete images every
five epochs. E3 encoder SA remains channel attention throughout.

| Group | RGB/E2 SA | Four CA paths | Config suffix after train_tdc_reblur_rgb_gated4ca_ |
| --- | --- | --- | --- |
| A | Channel | Channel | channel_3gpu_scratch.yml |
| B | Fixed + shifted pair | Channel | rgb_e2_shifted_sa_3gpu_scratch.yml (server-created) |
| C | Channel | Fixed + shifted pair | swin_ca_3gpu_scratch.yml |
| D | Fixed + shifted pair | Fixed + shifted pair | swin_sa_ca_3gpu_scratch.yml |

B is already running from a server-side copy of the existing SA config.
Do not replace that run's config with A, C or D. Archive the actual server
config into its experiment directory after training.

## Cross-attention pair contract

The new `cross_attn_type: shifted_window_pair` implements, per fusion path:

```text
d0 = fixed_window_CA(Q, K, V)
d1 = shifted_window_CA(Q + d0, K, V)
delta = d0 + d1
```

The source K/V are unchanged between blocks. Both use 8x8 spatial windows;
the second shifts by four pixels along H and W, with wrap-around masking.
Non-divisible inputs are padded before shifting and cropped back afterward,
so the cyclic shift and its mask share the same padded spatial grid.
The surrounding fusion applies its original query residual, gate (on the
event exchange paths only), and learned gamma once. Four directional paths
remain, but each contains two CA computations. This changes depth and
parameter count, not only the attention axis. No new FFN is added inside
the CA pair; "Swin-style" refers to alternating spatial window partitions,
not an exact Swin Transformer block or a complete Swin backbone.

Legacy `channel`, `window`, `shifted_window` retain their definitions and
checkpoint layouts; the last remains a single shifted CA block.

## Memory checks

Static autograd saved-storage estimates for batch 1, base dimension 32,
six bins and 256x256 inputs (FP32) were A=3.897, B=5.262, C=4.890,
D=6.255 GiB. These are NOT CUDA peak-memory measurements: convolution
workspaces, allocator behavior, optimizer temporaries, DDP and CUDA
context memory are not fully represented.

Before launching C/D, stop any competing run on the selected GPU and run:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/check_attention_memory.py \
  --config configs/train_tdc_reblur_rgb_gated4ca_swin_sa_ca_3gpu_scratch.yml
```

This uses real training data for two FP32 training steps including AdamW,
then the first three complete validation images, without writing experiment
artifacts. It does not include DDP buffers or guarantee the peak for other
image sizes. Leave headroom and verify the first full DDP validation.

Compare A/B for the SA configuration, A/C for the CA configuration,
and B/D or C/D for their interaction. Do not mix the old five-GPU total
batch 5 runs with these total batch 3 runs as single-variable ablations.
