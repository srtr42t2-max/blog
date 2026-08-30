---
title: "构建 KV Cache 容量计算器并验证 Block 浪费"
published: 2026-08-12
description: "从模型配置推导理论 KV 占用，再模拟分页 Block 分配对并发容量的影响。"
tags: ["AI Infra", "LLM 推理", "实验", "KV Cache", "容量规划", "Block Size"]
category: "AI Infra · 实验"
draft: true
lang: zh_CN
comment: true
---

| 类型 | 难度 | 预计时间 | 前置知识 | 核验日期 |
|---|---|---:|---|---|
| 实验 | 进阶 | 60 分钟 | KV Cache 显存账本；PagedAttention；Python 基础 | 2026-08-13 |

本实验把“模型能部署多少并发”拆成一笔透明账目，不需要 GPU。

## 第一部分：理论 KV 大小

实现：

```python
def kv_bytes_per_token(num_layers, num_kv_heads, head_dim, dtype_bytes):
    return 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
```

用以下输入验证：

```text
layers = 32
kv_heads = 8
head_dim = 128
dtype = bf16 (2 bytes)
expected = 128 KiB/token
```

固定 `query_heads = 32`，再比较 `kv_heads = 32 / 8 / 1`，分别观察 MHA、GQA、MQA 的理论容量比例。

## 第二部分：Block 分配

给定每个请求的 cached token 数与 `block_size`，实际分配 token 槽位为：

```text
allocated_slots = ceil(cached_tokens / block_size) * block_size
```

模拟一组长度：

```python
lengths = [17, 31, 128, 257, 1023, 4096]
block_sizes = [8, 16, 32, 64]
```

对每种 Block Size 输出：总理论 token、总分配槽位、尾块浪费 token、`waste / allocated_slots`（并可同时报告 `waste / theoretical_tokens`）和 Block Table 项数。

## 第三部分：容量预算

输入总 GPU 显存后，不要直接减去权重就结束。至少列出：

```yaml
total_gpu_gib: 80
weights_gib: 16
runtime_reserve_gib: 3
cuda_graph_gib: 2
workspace_gib: 2
communication_gib: 1
kv_budget_gib: 56
```

约定 `cached_tokens` 是已经写入 KV Cache 的 Prompt + Decode token 总数；根据请求长度分布估算最大并发。为使结果可复现，使用一个固定的 100 请求批次，分别计算：

1. 所有请求固定 4096 token；
2. 90 个请求为 1024 token、10 个请求为 16384 token，不共享；
3. 使用同一组长度，其中 45 个 1024-token 请求和 5 个 16384-token 请求共享一个 1024-token、按当前 Block Size 对齐的公共前缀，剩余 45 个短请求和 5 个长请求不共享；公共前缀只计一份物理 Block。

第三项要区分“共享的完整 Block”与“理论上相同的 token”，不要把未对齐后缀算入共享。最大并发先按该固定批次的平均物理占用估算，再说明真实 Scheduler 需要按请求逐个 admission，不能只靠平均值。

## 需要回答

- 更小的 Block 是否总能提高最大并发？
- 计算器为何只能给出估算，而不能替代真实压测？
- TP=4 时能否把单卡未分片的全模型理论 KV 直接除以 4？需要核对哪些实现条件？

## 完成标准

- 所有单位在 bytes、KiB、MiB、GiB 间显式转换。
- 输入非法模型形状时拒绝计算。
- 报告理论占用与 Block 分配占用两组数字。
- 所有结论附带模型、精度、并行和版本假设。

## 参考资料

- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)（论文）
