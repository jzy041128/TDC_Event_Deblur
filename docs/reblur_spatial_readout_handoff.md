# REBlur 条件空间读取 S 实验与服务器交接说明

更新日期：2026-11-17。

## 当前状态

本地已新增 S：逐时间片条件空间读取，再按时间等权平均。M/T/L 和历史融合模块继续保留。
本次没有连接或修改旧服务器、新服务器，也没有替用户启动服务器训练。
本地改动不自动表示 Git 远端已更新：用户执行下面的提交和推送命令成功后，才表示已上传。
S 计划在旧服务器上从头训练；完成后，将完整实验目录及启动留档迁移到新服务器。
尚未完成训练，因此没有 S 的 PSNR、SSIM、训练速度或旧服务器显存结果。

## 给新服务器对话的说明

旧项目现在增加了第四种三尺度统一的 E3 读取消融 S，对应
`configs/train_tdc_reblur_spatial_2ca_allscales_3gpu_scratch.yml`。
实现已在本地加入，Git 是否已推送请以用户提供的提交号和远端实际状态为准；不要假定新服务器已有此代码。
用户在本地完成 Git 提交推送后，旧服务器会拉取同一提交，并在旧服务器进行 S 的 200 轮训练。
训练完成后，用户还会将 S 的实验结果及配置、代码提交号、预检日志一起传到新服务器。
请先保持接收和核对，不要擅自启动新服务器上的重训、续训或覆盖已有结果。
要加载 S 的模型，先同步含 `ConditionalSpatialRead` 的代码，再使用 S 的原始配置和断点。
M/T/L 的旧结果不需要因为新增 S 而重跑；它们必须继续使用各自原始配置。

## S 的算法与控制变量

先保持现有第一次通道 CA：X = R + gamma2 * Inject2(CA(R,E2,E2))。
Q 来自归一化后的 [X,E2]，K/V 来自完整五维 E3。
在每个位置 p、每个时间片 t 内，对有效的 3x3 邻域单独 softmax：

```text
beta[t,j,p] = softmax_j(dot(q[p], k[t,p+delta_j]) / sqrt(d))
u[t,p] = sum_j beta[t,j,p] * v[t,p+delta_j]
M[p] = Wo * (sum_t u[t,p] / T)
```

邻居不仅参与打分，也提供被读取的 V。每个时间片的总系数严格为 1/T，
不是把 54 个分数联合 softmax，也不是先把 E3 平均再做空间注意力。
后面的 E3 通道 CA、Inject、空间门控、残差和公共 FFN 都保持原样。

| 项目 | 与 M/T/L 一致的设置 |
| --- | --- |
| 数据 | REBlur H5；train/test 路径保持旧服务器原路径；事件归一化不变 |
| 训练规模 | 预期 486 张训练帧、903 张整帧验证，实际以数据加载日志为准 |
| 输入与 E3 提取 | 6-bin，TDC 时间核 5，不是普通 Conv3d 对照 |
| 尺度 | 0/1/2 三个尺度都用 S，不混入 T 或其他读取器 |
| 通道与投影 | C=32/64/128；Q/K/V 宽度 d=8/16/32 |
| 注意力 | 读取器单头；编码器 SA、后续通道 CA 保持四头 |
| 邻域与实现 | 滑动 3x3；边界无效位置屏蔽；query chunk=512；重计算开启 |
| 训练 | FP32，seed=42，deterministic=true，从头 200 epoch，初始学习率 0.0002 |
| GPU / batch | 旧服务器物理 GPU 0,2,3；每卡 batch=1，总 batch=3 |
| 图像尺寸 | 训练随机 256x256；验证 patch_size=null，保留整帧 |
| 选模 | 每 5 epoch 验证；沿用既有训练器、指标和 best 选取方式 |
| 初始化 | 保留的投影、门控、公共骨干等与 M/T/L seed42 初始化一致，无预训练加载 |

新 S YAML 与统一尺度的 M/T/L YAML 除 name 和 e3_readout 外完全一致，测试会检查这一点。
配置中 `temporal_position: quadratic_mlp` 为保持配置匹配而保留；S 不注册此 MLP。
原因是同一时间片的标量时间偏置会在该片内部 softmax 中抵消，保留会产生无效参数。
S 初始化时消耗与 L 相同的随机数，再移除这组参数，避免改变其他层初始权重。
这也避免在 `find_unused_parameters=False` 的 DDP 中留下未参与前向的参数。

| 读取器 | 模型参数量（base_dim=32，6-bin） |
| --- | ---: |
| M | 2,960,299 |
| T | 2,977,966 |
| L | 2,977,966 |
| S | 2,977,771 |

S 比 T/L 少三个时间偏置 MLP，共 195 个参数。因此不把这组实验描述为严格等参数量。
其余有效共享张量及随机数状态对齐由测试核对。未修改旧 M/T/L 配置，旧断点仍按原配置加载。
S 的元数据标记为 `e3_readout: spatial`，不能把 M/T/L 的 latest.pth 当作 S 续训断点。

## 本地提交与旧服务器同步

下面仅列出本次相关文件，不要使用 git add . 混入其他未跟踪文件。

```powershell
cd C:\Users\19099\Desktop\TDC_Event_Deblur
git add models/temporal_event_fusion.py utils/model_checkpoint.py tests/test_matched_e3_readout.py tests/test_spatial_e3_readout.py configs/train_tdc_reblur_spatial_2ca_allscales_3gpu_scratch.yml docs/reblur_temporal_fusion_ablation.md docs/reblur_spatial_readout_handoff.md
git diff --cached --stat
git commit -m "新增三尺度条件空间读取与时间等权消融"
git push
```

用户在旧服务器执行：

```bash
cd ~/projects/TDC_Event_Deblur
git pull --ff-only
git log -1 --oneline
```

如果 pull 提示本地改动冲突，不要 reset、checkout 或覆盖旧服务器的运行配置，应先保留并核对冲突文件。

## 旧服务器训练命令

不在 tmux 中时先执行下一行；已经在 tmux 中则跳过，不要嵌套。

```bash
tmux new -As reblur_spatial_allscales_3gpu
```

然后在该会话内运行；预检失败不会自动开始训练：

```bash
conda activate tdc_deblur
cd ~/projects/TDC_Event_Deblur
set -o pipefail
CFG=configs/train_tdc_reblur_spatial_2ca_allscales_3gpu_scratch.yml
STAMP=$(date +%Y%m%d_%H%M%S)
RUNLOG=/data1/jzy/experiments/_launch_records/REBlur_Spatial2CA_$STAMP
mkdir -p "$RUNLOG"
cp "$CFG" "$RUNLOG/train_config.yml"
git rev-parse HEAD > "$RUNLOG/git_commit.txt"
export CUDA_VISIBLE_DEVICES=0,2,3
if torchrun --standalone --nproc_per_node=3 tools/check_attention_memory.py \
  --config "$RUNLOG/train_config.yml" --result-json "$RUNLOG/preflight.json" \
  2>&1 | tee "$RUNLOG/preflight.log"
then
  torchrun --standalone --nproc_per_node=3 train_ddp.py \
    --config "$RUNLOG/train_config.yml" 2>&1 | tee "$RUNLOG/train_console.log"
else
  echo "Preflight failed; training was not started."
fi
```

实验输出仍由 YAML 的 `path.experiments_root` 决定，位于 `/data1/jzy/experiments/` 下，
名称以 `Seed42_Spatial2CA_EncoderRestormerChannelSA_REBlur_6bin_TDC5_AllScales_3GPU_FromScratch_200ep_` 开头。
`_launch_records` 只保存不可混淆的启动配置、提交号、预检和控制台日志，不是训练断点目录。
日志中的 `Experiment dir:` 是模型、best.pth、latest.pth、train_log.txt 的实际位置。

## 完成后迁移到新服务器

保留完整的实际实验目录，并把本次 RUNLOG 的 train_config.yml、git_commit.txt、
preflight.json、preflight.log 复制到该实验目录中，再将整个目录迁移。
若已创建同名文件，先比较内容，不要直接覆盖。
迁移后核对文件大小和 SHA256，确认完整后再讨论新服务器测试或续训。
当前还不知道新服务器最终目标目录，因此这里不填写猜测的 scp/rsync 地址，也不执行传输。
REBlur 原有每五轮验证使用整帧，不需要为了 S 临时切换成 EVRB 的 tiled 整图协议。

## 本地验证与限制

```bash
python -m unittest discover -s tests -p test_spatial_e3_readout.py -v
python -m unittest discover -s tests -p test_matched_e3_readout.py -v
TDC_SPATIAL_FULL_SIZE_TESTS=1 python -m unittest discover -s tests -p test_spatial_e3_readout.py -k full_resolution -v
```

验证覆盖独立 unfold 参考实现及梯度、时间等权、边界、退化到 Mean V、
分块重计算一致性、三尺度 6/16-bin 前后向、共享初始化、原 T 兼容性和断点隔离。
本次本地 CPU 已通过上述常规测试，以及 S 的实际 base_dim=32、6-bin 配置下
两次 256x256 AdamW 更新和一次 260x320 整帧推理；旧五种融合模块和 EVRB
L/TDC、L/Conv3d 配置的回归测试也通过。完整尺寸测试使用合成数据，不是新训练结果。
本地 CPU 检查不等同于旧服务器三卡 NCCL、真实数据和显存预检；以旧服务器预检结果为准。
