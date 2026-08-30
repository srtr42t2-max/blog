---
title: "一个 Token 如何产生：Prefill、Decode 与请求时间线"
published: 2026-08-12
description: "沿着数据流拆开自回归生成的两个阶段，并用 Roofline 理解二者不同的性能瓶颈。"
tags: ["AI Infra", "LLM 推理", "概念", "Prefill", "Decode", "Roofline", "Continuous Batching"]
category: "AI Infra · 系统学习"
draft: true
lang: zh_CN
comment: true
---

| 类型 | 难度 | 预计时间 | 前置知识 | 核验日期 |
|---|---|---:|---|---|
| 概念 | 基础 | 38 分钟 | 在线推理指标与测量契约；Decoder-only Transformer；矩阵乘法 | 2026-08-13 |

在线推理不是一次普通的模型前向。它先并行处理完整 Prompt，然后进入逐 token 的循环；两阶段的矩阵形状、数据复用和调度行为明显不同。

> 30 秒结论：Prefill 处理一批未缓存的输入 token 并写入 KV Cache，最后位置的 logits 可以直接采样出首个 token。Decode 每轮读取历史 KV、追加新 KV，通常为每个序列产生一个 token。Prefill 常偏计算受限、Decode 常偏带宽受限，但这是由负载和硬件共同决定的条件性结论。

## 从文本到流式输出

```text
prompt
  -> tokenize
  -> admission / queue
  -> Prefill(all uncached prompt tokens)
       -> write K/V for every layer
       -> sample first output token
  -> Decode(new token per sequence per normal step)
       -> read historical K/V
       -> append K/V
       -> sample and stream
  -> stop condition
  -> release KV blocks
```

一个常见误区是认为 Prefill 结束后还需要一次独立 Decode forward 才能得到首 token。实际上，Prefill 最后位置的 hidden state 已经能够计算 logits 并采样首 token。

## Prefill 做了什么

假设 Prompt 长度为 `S`。在每层 Attention 中，全部 `S` 个位置可以并行计算 Q、K、V。因果 Mask 限制每个位置只能看见自己之前的 token，但不妨碍 GPU 并行处理这些位置。

Prefill 的主要工作包括：

- 对 `S` 个 token 执行 QKV 投影、Attention、输出投影和 MLP；
- 把每层每个位置的 K/V 写入 KV Cache；
- 用最后一个位置的 logits 采样第一个输出 token。

较长序列会形成更大的矩阵乘法，通常能更好利用 Tensor Core。Attention 的理论计算量随序列长度二次增长，但 FlashAttention 等 IO-aware Kernel 会显著改变实际访存和中间结果占用。

## Decode 做了什么

标准 Decode 的每个序列每轮通常只有一个新 token。新 token 的 Q 需要与历史所有 K 做 Attention，再用权重聚合历史 V。每层还要读取模型权重，并追加当前 token 的 K/V。

与 Prefill 相比，Decode 的矩阵 `M` 维往往很小，单请求数据复用有限。模型权重和不断增长的 KV Cache 需要重复从 HBM 读取，因此常见瓶颈是显存带宽。

但是“Decode 一定 memory-bound”是错误说法。更大的 Batch、很短的上下文、不同量化方式、MoE 路由、特殊硬件与 Kernel 都会移动工作点。

## 用 Roofline 说清条件

算术强度定义为：

```text
Arithmetic Intensity = FLOPs / bytes moved from the measured memory level
```

Roofline 给出的性能上界近似为：

```text
attainable FLOP/s = min(peak compute, bandwidth * arithmetic intensity)
```

判断瓶颈至少要回答：

1. 字节数相对哪一层存储计算，是 HBM、L2 还是共享内存？
2. 权重和 KV 在一次 Step 内复用了多少次？
3. Kernel 是否达到足够 Occupancy，是否被 Launch 开销限制？
4. Batch、序列长度和数据类型是什么？

Roofline 是性能模型，不是看到 Decode 就自动贴上的标签。

## Continuous Batching

静态批处理会等一批请求全部结束，短请求完成后留下空位。Iteration-level Scheduling 在每个生成 Step 重新组合活跃序列：完成的请求退出，新请求及时加入。这就是 Continuous Batching 的基础。

它提高了利用率，但也引入新的调度权衡：

- 本轮接纳多少 Prefill token？
- Decode 是否优先，以保护 ITL？
- KV Cache 不足时等待、抢占还是拒绝？
- 一个超长 Prompt 是否能独占一次很长的 GPU Step？

## 长 Prefill 为什么干扰 Decode

如果一个 32K Prompt 被作为一个完整 Prefill 执行，同一 Worker 上等待 Decode 的请求要等这个长 Step 结束。它们的 ITL 会出现尖峰，即使平均吞吐看起来很好。

Chunked Prefill 把长 Prompt 切成多个 token 块，在统一 token budget 下与 Decode 共同调度：

```text
without chunking: [--------- long prefill ---------][decode]
with chunking:    [chunk][decode][chunk][decode][chunk]
```

它能缩短 Decode 被连续阻塞的时间，但会增加调度轮次和部分开销；长请求自己的 TTFT 也不一定改善。优化目标应由 SLO 决定。

## 例外与边界

- Prefix Cache 命中时，Prefill 只处理未缓存的后缀。
- Speculative Decoding 的一次目标模型验证可能接受多个 token。
- 多模态模型还包含视觉编码器等额外阶段。
- PD 分离把 Prefill 与 Decode 放到不同实例，阶段间需要传输 KV。

## 自测

- 首 token 在哪个阶段产生？
- 为什么增大 Decode Batch 可能同时提高吞吐并恶化某些请求的 ITL？
- Chunked Prefill 解决了什么问题，又引入什么代价？

## 参考资料

- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)（论文）
- [Sarathi-Serve: Taming Throughput-Latency Tradeoff in LLM Inference](https://www.usenix.org/conference/osdi24/presentation/agrawal)（论文）
- [Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://doi.org/10.1145/1498765.1498785)（论文）
- [聊聊大模型推理服务中的优化问题](https://zhuanlan.zhihu.com/p/677650022)（社区解读）
