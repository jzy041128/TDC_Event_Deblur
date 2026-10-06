# RGB-guided event exchange in four-CA fusion

Select `model.fusion_mode: rgb_gated_event_then_rgb_4ca`.
At each of the three fusion scales, temporal-mean Event3D features and Event2D
features have the same `[B,C,H,W]` shape as the RGB features.

The added gate uses separate channel LayerNorms on RGB, E2 and E3, concatenates
them, then applies `Conv1x1(3C,C/4) -> GELU -> Conv1x1(C/4,2)`.
The two output channels produce independent gains `G2,G3 = 2*sigmoid(logits)`,
each `[B,1,H,W]`. They are not a softmax pair and need not sum to one.

The original event cross-attention directions are retained:

```
D2 = inject_e2(CA(Q=E2, K=E3, V=E3))
D3 = inject_e3(CA(Q=E3, K=E2, V=E2))
E2' = E2 + gamma_e2 * G2 * D2
E3' = E3 + gamma_e3 * G3 * D3
RGB' = RGB + gamma_rgb_e2 * inject_rgb_e2(CA(RGB,E2',E2'))
           + gamma_rgb_e3 * inject_rgb_e3(CA(RGB,E3',E3'))
```

`gamma_e2/e3` remain learned global scalars. Gates control each spatial position
and are broadcast across feature channels. Both gates use the original,
pre-exchange features. The last two RGB CAs and the surrounding network retain
their existing implementations. The pointwise gate receives spatial context
from the feature encoders; it adds no Gaussian smoothing or spatial convolution.

The last gate convolution is zero-initialized, so both gains start at one.
Its parameters learn on the first step; gradients to the earlier gate layers
become nonzero as the last convolution learns. A forked RNG scope preserves the
original random initialization of every shared network parameter at a fixed seed.
At base_dim=32, the gates add 17,646 parameters, approximately 0.5% of the network.

## Configurations

Only the first EVRB experiment is configured:

- `train_tdc_evrb_rgb_gated4ca_scratch.yml`: official 16-bin, TDC kernel size 5,
  400 epochs from scratch, seed 42, validation every 5 epochs.
- `eval_tdc_evrb_rgb_gated4ca_full.yml`: matching full-image evaluation model.

The model supports 6/16 bins and TDC kernel sizes 5/7. Dataset support for GoPro,
REVD and REBlur is retained. Add their gate configurations after assessing EVRB.

The planned transfer protocol trains the gated four-CA model on GoPro first,
then fine-tunes its complete checkpoint on REBlur and REVD. Training preloads,
resumes and final evaluation all load with `strict=True`: every gate and backbone
parameter must be present and have a matching shape. An original ungated GoPro
four-CA checkpoint is rejected when loading a gated model.

## First EVRB run

After the current 7-kernel two-CA run and its full-image evaluation finish:

```bash
tmux new -s evrb_rgb_gate4ca
conda activate tdc_deblur
cd ~/projects/TDC_Event_Deblur
CUDA_VISIBLE_DEVICES=0,2,3,4,5 \
torchrun --standalone --nproc_per_node=5 train_ddp.py \
  --config configs/train_tdc_evrb_rgb_gated4ca_scratch.yml
```

Save the training and evaluation YAMLs into the new experiment directory.
Evaluate with `evaluate_ddp.py`, the matching gate evaluation YAML, `best.pth`,
`--full-resolution --tile-size 256 --tile-overlap 32` for GoPro, REVD and EVRB.
For REBlur use `--full-resolution` without tiling, matching its complete 260x320
baseline evaluation. Keep checkpoint selection and all evaluation settings matched
when comparing the original four-CA, direct two-CA and gated four-CA runs.

The existing three exchange-strength diagnostic evaluations support a useful
exchange path within the trained four-CA checkpoint, but do not guarantee a benefit
from spatial gates over either independently trained baseline. Judge PSNR/SSIM
and actual step time after training. Kernel and gate comparisons should be separate.
