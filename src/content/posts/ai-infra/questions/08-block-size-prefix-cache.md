---
title: "Block Size 越小越好吗？Prefix Cache 命中后为什么 Decode 不一定更快？"
published: 2026-08-12
description: "分析分页粒度、元数据开销和前缀复用的真实收益。"
tags: ["AI Infra", "LLM 推理", "面试题", "Block Size", "Prefix Cache", "性能权衡"]
category: "AI Infra · 面试题"
draft: false
lang: zh_CN
comment: true
---

| 类型 | 难度 | 预计时间 | 前置知识 | 核验日期 |
|---|---|---:|---|---|
| 面试题 | 高级 | 13 分钟 | PagedAttention、KV Cache 显存账本 | 2026-08-13 |

## 30 秒回答

更小的 Block 能降低尾块浪费并提高前缀匹配粒度，但会增加 Block Table 长度、元数据、管理和 Kernel 访问开销。Prefix Cache 命中主要跳过共享前缀的 Prefill；Decode 仍要读取历史 KV 和模型权重，因此 Decode 不一定变快。比较时要说明命中率按 token、Block 还是 request 统计，并用真实请求分布测量 TTFT、ITL、吞吐和 CPU 开销。

## 评分点

优秀回答还会讨论当前 vLLM APC 只缓存完整 Block、引用计数、Cache Salt/LoRA/多模态身份，以及命中收益取决于公共前缀长度和后缀比例。其他实现的缓存粒度可能不同，不能把这一实现约束当成普遍定律。

关联：[PagedAttention：块表、分配、共享与抢占](/blog/posts/ai-infra/concepts/04-paged-attention/)。

## 参考资料

- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/design/prefix_caching/)（官方文档）
