---
title: "KV Cache 显存账本：形状、容量与生命周期"
published: 2026-08-12
description: "从 Tensor 形状推导每 token 的 KV 占用，并把理论容量扩展为真实在线服务显存账本。"
tags: ["AI Infra", "LLM 推理", "概念", "KV Cache", "GQA", "显存", "容量规划"]
category: "AI Infra · 系统学习"
draft: false
lang: zh_CN
comment: true
---

| 类型 | 难度 | 预计时间 | 前置知识 | 核验日期 |
|---|---|---:|---|---|
| 概念 | 进阶 | 42 分钟 | Prefill、Decode 与请求时间线；Tensor Shape；MHA/GQA 基础 | 2026-08-13 |

KV Cache 用显存换计算：历史 token 的 K/V 一旦生成便保存下来，后续 Decode 不再重复计算它们。容量规划的关键不是背一个数字，而是从模型形状推导。

> 30 秒结论：标准 Decoder 模型每个缓存 token 的理论 KV 大小是 `2 * L * H_kv * D_head * bytes_per_element`。真实显存还包括权重、激活、Workspace、CUDA Graph、通信 Buffer、分配器预留与 Block 向上取整，模型权重能装下不等于服务不会 OOM。

## 为什么缓存 K 和 V

在自回归 Decode 中，当前 token 的 Q 只使用一次；历史 token 的 K/V 会在每个后续 Step 被再次读取。缓存 K/V 能避免每轮重新计算全部历史 token。

这里的“每 token”指引擎已经计算并缓存的 token，和用户已经收到的 token 数可能存在一步时序差。做容量分析应使用引擎状态，而不是只数流式响应字符。

## 从形状推导公式

设：

- `L`：Transformer 层数；
- `H_kv`：KV Head 数量；
- `D_head`：每个 Head 的维度；
- `B`：每个元素字节数。

每层、每 token 的 K 或 V 都包含 `H_kv * D_head` 个元素。因为同时保存 K 和 V：

```text
KV bytes/token = 2 * L * H_kv * D_head * B
```

在**没有 Prefix Sharing、Copy-on-Write 或其他物理去重**时，一组请求的逻辑总量为：

```text
total KV bytes = bytes/token * sum(cached tokens of every request)
```

这只是容量上界，不一定等于物理显存。启用前缀共享时，应按唯一的物理 KV Block（包括 Block 向上取整和对齐）计数；共享 Block 只算一份，并通过引用计数管理生命周期。

### 算例

模型有 32 层、8 个 KV Head、Head Dim 为 128，KV 使用 BF16（2 bytes）：

```text
2 * 32 * 8 * 128 * 2 = 131072 bytes = 128 KiB/token
```

如果 20 个请求各缓存 4096 token：

```text
128 KiB * 20 * 4096 = 10 GiB
```

这还没有包含任何权重、临时激活和运行时内存。

## MHA、GQA 与 MQA

在 MHA 中通常 `H_kv = H_q`。GQA 让多个 Query Head 共用一个 KV Head 组，MQA 则让所有 Query Head 共用一组 K/V。因此在层数、Head Dim 与精度相同时，KV 大小按 `H_kv` 线性变化。

不要无条件用 hidden size 替代 `H_kv * D_head`：对 GQA/MQA 模型这会高估 KV Cache。

## 分布式场景不能机械除法

Tensor Parallel 往往会沿 Head 维分片 KV，但实际每 Rank 占用是否近似除以 TP，取决于：

- KV Head 数能否被 TP Size 合理分片；
- 实现是否复制某些 KV Head；
- Attention Backend 的布局约束；
- 模型是否使用 MLA 等不同缓存表示。

Pipeline Parallel 的每个 Stage 只保存该 Stage 层的 KV，但请求生命周期和负载可能在 Stage 间耦合。任何 per-rank 估算都应对照当前框架和模型配置验证。

## KV Cache 生命周期

```text
admit request
  -> allocate blocks on demand
  -> Prefill writes prompt K/V
  -> Decode reads history and appends K/V
  -> optional prefix sharing / copy-on-write
  -> finish, cancel or evict
  -> release references and return blocks to pool
```

Prefix Cache 可以让相同前缀共享物理 KV Block，但命中通常只跳过重复 Prefill；后续 Decode 仍需读取对应历史 KV。

KV 量化把 K/V 换成更窄的数据格式，可以增加容量、降低带宽压力，但会引入量化/反量化开销与精度风险。必须在目标模型和真实请求分布上验证。

## 完整显存账本

| 项目 | 是否固定 | 主要影响因素 |
|---|---|---|
| 模型权重 | 近似固定 | 参数量、精度、并行分片 |
| KV Pool | 通常预留 | 可用显存比例、Block 配置 |
| 激活 | 动态 | Batched token 数、Kernel 实现 |
| Workspace | 动态或缓存 | GEMM/Attention Backend |
| CUDA Graph | 配置相关 | 捕获的 Batch Shape 数量 |
| 通信 Buffer | 配置相关 | TP/PP/EP、NCCL |
| 分配器预留 | 动态 | 历史峰值和碎片 |
| Runtime 上下文 | 近似固定 | CUDA、框架与驱动 |

所以“权重占 16 GiB，GPU 有 24 GiB，还剩 8 GiB 给 KV”通常是过于乐观的估算。

## 理论 Token 与实际 Block

分页式 KV Manager 以固定大小 Block 分配。若一个 Block 容纳 16 token，请求缓存 17 token 时可能已经占用两个 Block，即按 32 token 的空间计费。尾块浪费、内部元数据和对齐都会让物理占用高于理论 KV bytes。

## 自测

- 为什么缓存 Q 的收益通常很小？
- 把 BF16 KV 改为 FP8，理论容量发生什么变化？还必须验证什么？
- 一个模型能在空闲 GPU 上成功加载，为何压测时仍可能 OOM？

## 参考资料

- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)（论文）
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)（论文）
- [大模型推理优化技术：KV Cache](https://zhuanlan.zhihu.com/p/700197845)（社区解读）
