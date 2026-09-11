# Event-reference channel reorganization (B23)

This is an input-dependent channel mixing hypothesis, not a guarantee of semantic
alignment or spatial registration. Existing fusion modes are unchanged.

## Configuration

- B23: `fusion_mode: event_reorg_b23_ca`. E2 is the reference; E3 supplies content.
- B23 requires `fusion_dim: 2d` and `cross_attn_type: channel`.
- `swapped_kv_order` does not affect this mode.

## Per-scale computation

At each of H, H/2 and H/4, let R, E2 and E3 be the post-self-attention branch
features. Mean E3 over time only for the local fusion computation. Choose
reference S=E2 and content U=E3.

```text
Q = normalize(heads(Wq * LNq(R)))
S = normalize(heads(Wref * LNref(S)))
K = normalize(heads(Wk * LNk(U)))
V = heads(Wv * LNv(U))

P = softmax(temperature_reorg * S @ transpose(K))
V_reorganized = P @ V
A = softmax(temperature_read * Q @ transpose(S))
delta = Project1x1(merge_heads(A @ V_reorganized))

F = R + gamma1 * Inject3x3(delta)
output = F + gamma_ffn * ConvFFN2D(LNffn(F))
```

Softmax is over the last channel axis; Q/K normalization is over the flattened
spatial axis, as in the existing normalized channel CA. Each map has shape
`[batch, heads, channels_per_head, channels_per_head]`.

The SAME projected reference S is Q for P and K for A. There is deliberately no
projection, V reprojection, FFN, gamma or event residual between P@V and A@V_reorganized.
Those operations would change the stated intermediate channel coordinates.
The two temperatures and the RGB/reference/content-K/content-V normalization
layers are independent. The reused reference is intentional; this is not two
accidentally shared complete CA modules. Different scales have separate parameters.

Event branches still downsample their own post-SA features, never the reorganized
fusion result. RGB still downsamples the fusion output. Decoder skips, the image
residual, loss, optimizer and data loading have not changed.

## Comparison boundary

Reference: Swapped-KV cascade + Restormer channel SA + plain decoder, seed 42,
600 epochs (reported best PSNR 36.49 dB). The archived reference YAML was checked
against the new YAML: data, loss, learning rate, seed/determinism, batch size,
patch size, heads, SA and decoder settings agree. The archived reference has no
additional active training options missing from the new YAML.

| Model, base_dim=32 | Total trainable parameters |
| --- | ---: |
| Swapped-KV cascade | 2,930,952 |
| B23 | 2,671,781 |

At the time of the original comparison, B23 had 259,171 fewer parameters
(8.84%). It computes two attention maps,
but they share the intermediate reference projection and have one final output
path instead of two complete CA/projection/residual paths. Do not call this an
equal-parameter or equal-runtime ablation. No unused layers were added to match
the parameter count. Equal random seeds also do not guarantee identical decoder
initial weights across differently parameterized models.

Compare equal training budgets, best/final/last-50 PSNR, SSIM and measured time.
A higher PSNR would support this information path, not by itself prove semantic
alignment.

## Local verification

```bash
python -m unittest discover -s tests -p test_event_reorg.py -v
```

Tests cover the explicit two-map formula, the E2-reference/E3-content route,
temporal mean and rejection of incompatible attention selectors. No server
execution was used; server CUDA/DDP performance and peak memory remain to be
measured.
