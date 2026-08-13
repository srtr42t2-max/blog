---
title: "给定层数、KV Head、Head Dim、精度、上下文和并发，如何计算 KV Cache 显存？"
published: 2026-08-12
description: "用公式、单位换算和分布式边界完成一笔可审计的 KV 显存估算。"
tags: ["AI Infra", "LLM 推理", "面试题", "KV Cache", "显存估算", "GQA"]
category: "AI Infra · 面试题"
draft: false
lang: zh_CN
comment: true
---

| 类型 | 难度 | 预计时间 | 前置知识 | 核验日期 |
|---|---|---:|---|---|
| 面试题 | 进阶 | 13 分钟 | KV Cache 显存账本 | 2026-08-13 |

## 30 秒回答

标准 Decoder 模型每 token 的理论 KV 大小是 `2 * L * H_kv * D_head * bytes_per_element`，前一个 2 分别代表 K 和 V。在没有 Prefix Sharing 时，总量乘以所有请求实际 cached token 数。举例 `L=32, H_kv=8, D_head=128, BF16`，得到 `128 KiB/token`；20 个请求各 4096 token 时约 10 GiB。若启用共享，实际物理显存应按唯一物理 Block 计数。然后还要考虑 TP 的实际分片/复制、Block 向上取整、KV Pool 预留和其他显存项目。

## 评分点

- **基础**：公式正确，单位不混乱。
- **合格**：区分 MHA/GQA/MQA，能说明总 token 是并发请求的总和。
- **优秀**：补充 per-rank、Block Size、prefix sharing、KV 量化和完整显存账本。

不要无条件写成 `2 * layers * hidden_size * bytes`；GQA/MQA 下 KV Head 数可能远小于 Query Head 数。

关联：[KV Cache 显存账本](/blog/posts/ai-infra/concepts/03-kv-cache-ledger/)。

## 参考资料

- [GQA 原论文](https://arxiv.org/abs/2305.13245)（论文）
