# REBlur three-GPU attention experiments

All groups use six bins, TDC kernel 5, RGB-gated four-path fusion,
three physical GPUs (0, 2, 3), batch size 1 per GPU, seed 42,
200 epochs from scratch, and validation on all 903 complete images every
five epochs. E3 encoder SA remains channel attention in groups A-D.

| Group | RGB/E2 SA | Four CA paths | Config suffix after train_tdc_reblur_rgb_gated4ca_ |
| --- | --- | --- | --- |
| A | Channel | Channel | channel_3gpu_scratch.yml |
| B | Fixed + shifted pair | Channel | rgb_e2_shifted_sa_3gpu_scratch.yml (server-created) |
| C | Channel | Fixed + shifted pair | swin_ca_3gpu_scratch.yml |
| D | Fixed + shifted pair | Fixed + shifted pair | swin_sa_ca_3gpu_scratch.yml |

An E3-only spatial-SA control uses
`train_tdc_reblur_rgb_gated4ca_e3_shifted_sa_3gpu_scratch.yml`.
It changes only the experiment name and E3 SA relative to A: RGB/E2 and
all CA remain channel attention. E3 uses the existing fixed + shifted
3D SA pair with temporal window 2 and spatial window 8. All six time
positions are preserved; only H/W are shifted. No network code or legacy
config is changed for this additional experiment.

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

## Why E3 spatial SA uses more memory

At each scale, RGB and E2 SA inputs are `[B,C,H,W]`; E3 is `[B,C,T,H,W]`.
The E2 2D stem mixes the input bins as channels, while the E3 TDC stem
retains T. For 256x256 features, 8x8 spatial windows, T=6 and temporal
window 2:

| Quantity (one attention block, one sample) | RGB or E2 | E3 |
| --- | --- | --- |
| Tokens per window | 64 | 128 |
| Windows | 1024 | 3072 |
| Attention entries per window per head | 4096 | 16384 |
| One score tensor, four heads, FP32 | 64 MiB | 768 MiB |

Thus E3's score tensor is 12 times that of one 2D branch, not merely twice
as large. Q/K/V and FFN activations also retain six time positions, and
the shifted pair uses two blocks. This is not a 12x prediction for total
network memory.

With channel SA and C=32, four heads each have eight channels: the
attention matrix is 8x8 per head (1 KiB in FP32 across four heads) for
all three branches. E3 normalizes/aggregates over T*H*W rather than H*W;
it still keeps larger Q/K/V, 3D convolution and FFN activations, so its
total memory is higher despite the small, equal-sized attention matrix.

The E3-only variant has a static autograd saved-storage estimate of
9.956 GiB for a 256x256 training crop. It is NOT a measured GPU peak and
may exceed a 12 GiB card once runtime overhead is included. Run
`tools/check_attention_memory.py` with the E3-only config before DDP.
If it fails, do not reduce only this variant's crop or change temporal
window silently: agree on a matched-memory implementation or a separate
experimental setting first.
