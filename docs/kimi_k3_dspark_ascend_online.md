# Kimi-K3 DSpark 在 Ascend NPU 上的在线训练

本文记录 8 台推理 + 1 台训练的在线增训流程。推理侧对数据集里已有的 token 做 prefill，抽出与 Inferact 草稿对齐的 hidden state；训练侧在单独一台机器上用 FSDP 更新草稿。推理不会把样本重新 decode 一遍。

配置文件：`configs/vllm_kimi_k3_dspark_ascend_8infer_1train.yaml`。

## 1. 角色和约束

九个 pod 必须在同一个超节点上，否则 HCCL / 专家并行 all-to-all 建不起来。`itask start` 没有 `--colocate`，不能保证这一点。新 pod 要用 `itask create --colocate <已有任务>` 钉到锚点任务所在的超节点。


| Pod                         | 角色              | Ray 资源                          |
| --------------------------- | --------------- | ------------------------------- |
| `dspark-training-sy0`–`sy7` | 推理，每台一个 TP16 引擎 | 各 16 GPU                        |
| `dspark-training-sy8`       | Ray 训练，FSDP     | 16 GPU                          |
| `sy7`                       | 同时是 Ray head    | `10.30.14.161:6380`（重建 pod 后要改） |


专家并行只包含 8 台推理机：`EP = 16 × 8 = 128`，896 个专家 / 128 = 每卡 7 个。训练机不加入这个组。9×16 = 144 除不尽 896，不能把第 9 台加进推理。

当前镜像（推理 pod）：

```text
hcr.meta-guian02.guian.hw-a3.local/antsys/vllm:release_v0.26.0_20260817_202609151548_aarch64
```

Python：`/usr/local/python3.11.15/bin/python`。Ray：2.58.0。

## 2. 路径

```text
CODE=/a3_inference/itask/workdir/shared/wjf02357757/TorchSpec-sunny
TARGET=/a3_inference/itask/workdir/shared/models/modelhub_69400046_kimi-k3-w4a8-norope-142600122_20260824234539/model
DATA=/a3_inference/itask/workdir/shared/libowen/success.jsonl
DRAFT=/a3_inference/itask/workdir/shared/wjf02357757/Kimi-K3-DSpark-Inferact
STUBS=/a3_inference/itask/workdir/shared/wjf02357757/cuda_stubs
OUTPUT=/a3_inference/itask/workdir/shared/wjf02357757/outputs/kimi-k3-dspark-online
```

`DRAFT` 是 Hugging Face 格式（`config.json` + `model.safetensors`）。`config.json` 的 `target_layer_ids` 是 `[2, 23, 47, 71, 89]`。不要改用 `configs/draft_models/kimi_k3_dspark_mla.json`，那里的层号是旧的 `[7, 31, 47, 63, 87]`。

也不要把 `training.load_path` 指到 `kimi_k3_dspark_init`。那个目录不是带 `latest_checkpointed_iteration.txt` 的训练 checkpoint，`checkpoint.load()` 会跳过，草稿会从随机权重开始。初始权重走 `model.initial_draft_model_path`。

## 3. 操作流程



### 3.1 安装 TorchSpec

在共享盘上的仓库做可编辑安装。sy0–sy8 都要装。推理 pod 还要打第 4 节的 vLLM 补丁；sy8 只训练，不需要那些补丁。

```bash
cd "$CODE"
/usr/local/python3.11.15/bin/python -m pip install -e . --no-build-isolation
```

容器重建后 site-packages 里的 vLLM 补丁会丢，需要重新打。可编辑安装指向共享盘，源码改动对已安装的 pod 立即生效。

### 3.2 给推理 pod 打 vLLM-Ascend 补丁

补丁来自 hs4 的 `k3-hs4-runtime-patches`（见本地 `K3_HS4_DEPLOYMENT.md`）。在 sy0–sy7 上、启动 vLLM 之前执行。换镜像后先 `--dry-run`，锚点对上再写入。当前这版镜像 dry-run 是对齐的。

六个补丁和它们改的文件：


| 补丁                                         | 文件                                            | 作用                                                                                                     |
| ------------------------------------------ | --------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `patch_kimi_k3_final_hidden.sh`            | `vllm_ascend/models/kimi_k3.py`               | 最后一层 hidden 取 pre-norm。标记 `K3_DSPARK_FINAL_HIDDEN_PRE_NORM`                                            |
| `patch_hidden_state_packed_kv_capacity.sh` | `vllm/v1/core/kv_cache_utils.py`              | 按 packed KV 计算容量，避免把 hidden cache 算进注意力页                                                               |
| `patch_hidden_state_separate_kv_pool.sh`   | 同上                                            | 独立 hidden-state 池，页大小保持 384，不把 hidden block 缩成几个 token。标记 `K3_DSPARK_SEPARATE_HIDDEN_POOL_NATIVE_PAGE` |
| `patch_hs_connector_atomic_publish.sh`     | `vllm/.../example_hidden_states_connector.py` | hidden state 原子发布                                                                                      |
| `patch_ascend_sampler_logprobs_mode.sh`    | `vllm_ascend/worker/model_runner_v1.py`       | 把 `logprobs_mode` 传给 `AscendSampler`                                                                   |
| `patch_force_moe_alltoall.sh`              | `vllm_ascend/ascend_forward_context.py`       | `VLLM_ASCEND_FORCE_MOE_ALLTOALL` 打开时强制 MoE all-to-all。当前启动没有设这个环境变量，补丁处于关闭状态                           |


TorchSpec 不使用 `ExampleHiddenStatesConnector`。独立 hidden 池是由 `vllm_engine.py` 把同样的 `kv_connector_extra_config` 交给 `MooncakeHiddenStatesConnector` 打开的。没有这个开关时，65537 上下文会被未拆池的公式拒绝（约 136.83 GiB KV，实际可用约 21 GiB）。

### 3.3 启动 Ray

先停掉占用 NPU 的旧 `torchspec.offline.generate` / `torchspec.train_entry` / `ray::VllmEngine` / `VLLM::` 进程，不要停 raylet。

sy7：

```bash
export PATH=/usr/local/python3.11.15/bin:$PATH
ray start --head --node-ip-address=10.30.14.161 --port=6380 \
  --num-gpus=16 --dashboard-host=127.0.0.1 --disable-usage-stats
```

sy0–sy6 和 sy8 加入同一个 head，`--node-ip-address` 用本 pod IP，`--num-gpus=16`。`ray status` 应看到 9 个节点、144 GPU，且启动前占用为 0。

### 3.4 启动训练

在 sy7 上执行。环境变量会通过 `torchspec.utils.env.get_torchspec_env_vars()` 进 Ray actor，所以 `LD_LIBRARY_PATH`、`HCCL_*`、`VLLM_USE_V2_MODEL_RUNNER` 必须出现在驱动进程里，并且列在 `_TORCHSPEC_ENV_KEYS` 中。

```bash
cd "$CODE"
export PATH=/usr/local/python3.11.15/bin:$PATH
export RAY_ADDRESS=10.30.14.161:6380
export PYTHONUNBUFFERED=1
export MC_STORE_MEMCPY=0
export TORCHSPEC_LOG_DIR=$CODE/../running_logs/dspark_k3_online
export ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export ASCEND_RT_VISIBLE_DEVICES=$ASCEND_VISIBLE_DEVICES
export VLLM_USE_V2_MODEL_RUNNER=0
export HCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=800
export HCCL_OP_EXPANSION_MODE=AIV
export LD_LIBRARY_PATH=$STUBS:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:${LD_LIBRARY_PATH:-}

python -m torchspec.train_entry \
  --config configs/vllm_kimi_k3_dspark_ascend_8infer_1train.yaml
```

`VLLM_USE_V2_MODEL_RUNNER=0` 是必须的：V2 runner 拒绝 `extract_hidden_states`。`cuda_stubs` 提供 `libcuda.so.1`，否则 Mooncake 在 NPU 机器上导入失败。

权重加载大约 4–7 分钟。成功标志：

- `Loaded initial draft weights from .../Kimi-K3-DSpark-Inferact/model.safetensors`
- `GPU KV cache size: 96,422 tokens`，`Available KV cache memory: 21.06 GiB`
- `Mooncake hidden-state cache layout NHD shape=(512, 384, 6, 7168) block_size=384`
- `All 16 training actors and 8 inference engines initialized`

日志在 `running_logs/online_8infer_1train_*.log`。

## 4. 训练配置

下面是当前 yaml 里实际生效的值。

### 4.1 模型


| 项                          | 值                                                                                                             |
| -------------------------- | ------------------------------------------------------------------------------------------------------------- |
| 目标                         | Kimi-K3 W4A8                                                                                                  |
| 草稿结构                       | Inferact `config.json`（`k3_dspark`，5 层，hidden 7168）                                                           |
| 初始权重                       | Inferact `model.safetensors`，rank 0 在覆盖 embedding 之前加载                                                        |
| embedding / lm_head / norm | `language_model.model.embed_tokens.weight`、`language_model.lm_head.weight`、`language_model.model.norm.weight` |
| 抽取层                        | `[2, 23, 47, 71, 89]`，`last_hidden_states_prenorm: true`                                                      |


vLLM 的 hook 打在层输入上，引擎会把层号 +1，并追加最后一层 pre-norm 槽。日志里是 `[3, 24, 48, 72, 90, 93]`，与 hs4 的捕获点一致。

### 4.2 优化和数据


| 项     | 值                                                               |
| ----- | --------------------------------------------------------------- |
| 数据    | `success.jsonl`，renderer `kimi-k3`，只监督最后一轮 assistant            |
| 样本    | 27732 条有效样本；序列上限 65537；`min_loss_tokens: 14`                    |
| 学习率   | `5e-5`，warmup `0.04`，cosine 收到 0，weight decay 0，grad clip 1.0   |
| batch | micro batch 1，累积 8，16 卡。一次 optimizer step = 128 条               |
| 步数    | 每 epoch 216 步，5 epoch，共 1080 步                                  |
| loss  | `0.1 * CE + 0.9 * L1 + 1.0 * confidence`，块内权重 `exp(-k / 4)`     |
| 草稿    | block size 7，anchor 512，目标层 5                                   |
| 注意力   | `sdpa`，FSDP `FULL_SHARD`，梯度检查点开，HCCL                            |
| 续训    | `continual_training: true`，但没有 `load_path`。这次不从训练 checkpoint 恢复 |


`save_interval: 1000`，另加 `save_per_epoch: true`。1080 步里，第 1000 步存一次；每个 epoch 结束（216、432、648、864、1080）再存一次。目录是 `outputs/kimi-k3-dspark-online/checkpoints/iter_*`。进程挂了再拉起不会自动接着这些目录，除非把 `training.load_path` 指到对应的 checkpoint。

### 4.3 推理

只做 prefill。`VllmEngine.generate` 使用 `SamplingParams(max_tokens=1)`，prefill 结束后再生成 1 个 token，用来把 hidden state 交出来。保存的序列长度是 prompt 长度。`train_with_decode` 为 false；打开它需要 SGLang，当前 8 台 vLLM 不会走。


| 项                                  | 值      | 原因                                                               |
| ---------------------------------- | ------ | ---------------------------------------------------------------- |
| TP / DP                            | 16 / 8 | 一台一个引擎，八台锁步                                                      |
| `mem_fraction_static`              | 0.90   | 0.98 配合关 chunked prefill 会 OOM                                   |
| `max_model_len` / `max_seq_length` | 65537  | 数据集有更长样本时加载会直接报错；`max_seq_length` 会覆盖 vLLM 的 `max_model_len`     |
| `max_num_batched_tokens`           | 8192   | chunked prefill 开着，chunk 再大容易在 prefill 激活上 OOM                   |
| `max_num_seqs`                     | 1      | 与 `inference_batch_size: 1` 一致。KV 池约 96422 token，两条长序列放不下        |
| `enable_chunked_prefill`           | true   | 长序列必须分块 prefill                                                  |
| `enforce_eager`                    | true   | 关闭 compile / cudagraph                                           |
| `quantization`                     | ascend | W4A8                                                             |
| `enable_expert_parallel`           | true   |                                                                  |
| 前缀缓存                               | 强制关闭   | `vllm_engine.py` 在 extra_args 之后写死 `enable_prefix_caching=False` |


8 个引擎在专家并行里锁步：任何一台跳过 `generate()` 都会把 all-to-all 卡死。一波 8 条，一个 optimizer step 要 16 波。步长时间由这 16 波 prefill 决定，训练计算大约 1–2 分钟，会和后续推理重叠。

Mooncake：TCP，`global_segment_size: 256GB`，`local_buffer_size: 8GB`，hidden dim 7168。样本池上限 32。

### 4.4 进度条字段


| 字段        | 含义                                                     |
| --------- | ------------------------------------------------------ |
| `thru`    | 最近 10 秒写入样本池的推理速度（样本/秒）。一波 prefill 往往超过 10 秒，刷新时空窗就是 0 |
| `I`       | 最近 10 秒 `generate()` 完成批次算出的推理吞吐。空窗也是 0                |
| `T`       | 训练吞吐 = 128 / 本步训练计算时间（样本/秒）                            |
| `wait`    | 这一步开训前，等待 128 条 hidden state 进队列的秒数                    |
| `pool`    | 已推理完、还没被训练取走的条数，上限 32                                  |
| `acc`     | 7 个块内位置的 token 加权命中率                                   |
| `acc_len` | `acc_0 + acc_0·acc_1 + …`，用各位置贪心命中率连乘，估计连续接受长度         |




## 5. TorchSpec 适配



### 5.1 设备

`torchspec/utils/device.py` 统一 CUDA / NPU：`get_accelerator_device_type()`、`set_accelerator_device()`、当前 stream。Ray actor 在 `torchspec/ray/ray_actor.py` 里用它绑定本卡，不再假设 `torch.cuda`。

`torchspec/utils/env.py` 把 `LD_LIBRARY_PATH`、`HCCL_CONNECT_TIMEOUT`、`VLLM_USE_V2_MODEL_RUNNER` 等转发给 Ray actor。默认 `VLLM_USE_V2_MODEL_RUNNER=0`。

### 5.2 推理引擎

`torchspec/inference/engine/vllm_engine.py`：

- 投机方法固定为 `extract_hidden_states`，不加载草稿权重。
- 配置里的抽取层 +1，并追加 `num_hidden_layers`（Kimi-K3 为 93）作为 pre-norm last hidden。
- KV connector 使用 `MooncakeHiddenStatesConnector`，并传入 `enable_separate_hidden_state_pool=True`、`enable_cross_layers_blocks=False`。这两项是受保护键，yaml 的 `extra_args` 覆盖不了。
- DP>1 时在 import vLLM 之前设置 `VLLM_DP_SIZE`、`VLLM_DP_RANK`、`VLLM_DP_MASTER_IP/PORT`，不把 `data_parallel_size` 传进 `LLM()`。
- `generate()` 使用 `max_tokens=1`。



### 5.3 Mooncake 在 NPU 上的布局和拷贝

`torchspec/inference/engine/mooncake_hidden_states_connector.py` 同时识别两种 4 维 cache：

- `LBNHC`：`(blocks, layers, block_size, hidden)`，`shape[2] == block_size`
- `NHD`：`(blocks, block_size, layers, hidden)`，`shape[1] == block_size`

Ascend 的 `CacheOnlyAttentionBackend` 给出的是 NHD。注意力 block size 是 384（mamba page 对齐），6 是捕获的层数，不是 block size。按旧逻辑用 `shape[2]` 当 block size 会报 `derived 384, cache view has 6`。

`torchspec/transfer/mooncake/eagle_store.py`：NPU 上 `copy_from_tensor(..., non_blocking=True)` 之后，先 `torch.npu` stream synchronize，再把 pinned buffer 交给 Mooncake。否则后续层读到的 `input_ids` 还是上一层的，出现 `Hidden-state input_ids mismatch`。CUDA 路径仍走原来的 event。

### 5.4 草稿初始权重

`torchspec/training/checkpoint.py` 的 `load_initial_draft_weights()` 读取 serving 格式的 `model.safetensors`，用 `to_internal_keys()` 映射后 `strict=True`。Inferact 的 key 已经是内部名（`layers.*`、`context_proj`、`markov_head`、`confidence_head`），映射会保持原样。

`DFlashTrainer.init_model`（DSpark 继承它）在 rank 0 上、`load_embedding()` 之前调用它。目标模型的 embedding 随后覆盖草稿 embedding 并冻结。这样 FSDP 广播的是 Inferact 权重，而不是随机初始化。

### 5.5 NPU 上的草稿注意力

`DFlashTrainer.init_model` 在 NPU 上设置 `torch._dynamo.config.disable = True`。NPU inductor 在动态 RoPE 下标上会 `KeyError: s39 + 1`。

`torchspec/models/draft/dflash.py`：`gradient_checkpointing=true` 时，每一层用 `torch.utils.checkpoint` 包起来，反向时只重算一层。

`torchspec/models/draft/dspark.py` 的 `K3DSparkMLAAttention`：不走 NPU SDPA。带 float mask 时 SDPA 会按整段 query 分配分数矩阵；512 anchor × block 7 = query 3584，64 head、约 6 万上下文的 fp32 分数约 57 GiB。实现改为每次 256 条 query 的 matmul，并对每个 chunk 做 checkpoint，只保留一块分数。anchor 数仍是 512。

### 5.6 放置

在线训练同时创建训练和推理两个 role，因此可以使用 `placement_strategy: custom`，并用 `training_node_ips` / `inference_node_ips` 把 sy8 和 sy0–sy7 分开。只建推理 role 的离线任务不能用 custom 放置。