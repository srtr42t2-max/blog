---
title: "LLM 推理 Infra 系统学习路线"
published: 2026-08-12
description: "从在线请求指标出发，沿 Prefill、Decode、KV Cache、GPU、推理引擎、分布式与生产 Serving 建立完整知识链。"
tags: ["AI Infra", "LLM 推理", "学习路线", "知识地图"]
category: "AI Infra · 导航"
draft: false
lang: zh_CN
pinned: true
priority: 1
comment: true
---

这个知识库面向 **LLM 推理 Infra**，目标不是背一套孤立题目，而是能沿着一次在线请求解释数据流、资源账本、性能瓶颈和工程权衡。

主线固定为：

```text
请求进入服务
-> 指标与负载模型
-> Prefill / Decode
-> KV Cache
-> GPU / Kernel
-> 推理引擎调度
-> 分布式推理
-> 生产 Serving
```

系统基础会在主线需要时进入，例如用 Roofline 解释计算与访存、用 OS 分页理解 PagedAttention、用排队和背压解释尾延迟。这样既不跳过底座，也不会先陷入一套和推理脱节的通用百科。

:::important
每个主题都按“概念 -> 手算或实验 -> 面试表达”闭环。面试题用于检验理解，不替代概念文章；社区面经只决定内容权重，技术结论由论文、官方文档、源码和实验确认。
:::

## 当前可学习内容

### 第一阶段：先建立性能测量契约

1. [在线推理指标与测量契约](/blog/posts/ai-infra/concepts/01-metrics-contract/)
2. [实验：从流式日志计算 TTFT、ITL 与 Goodput](/blog/posts/ai-infra/labs/01-metrics-from-stream/)
3. 面试检验：
   - [如何严格定义 TTFT、TPOT、ITL、E2E、Throughput 和 Goodput？](/blog/posts/ai-infra/questions/01-inference-metrics/)
   - [吞吐更高但 P99 TTFT 超标，系统更好吗？](/blog/posts/ai-infra/questions/02-throughput-vs-slo/)

完成标准：能写清测量边界、到达过程和 SLO，能解释为什么只报 output tok/s 或平均延迟不足以判断在线系统。

### 第二阶段：沿一次请求理解 Prefill 与 Decode

1. [Prefill、Decode 与请求时间线](/blog/posts/ai-infra/concepts/02-prefill-decode/)
2. 面试检验：
   - [为什么常说 Prefill compute-bound、Decode memory-bound？](/blog/posts/ai-infra/questions/03-prefill-vs-decode/)
   - [长 Prompt 为什么会让其他请求的 ITL 抖动？](/blog/posts/ai-infra/questions/04-long-prompt-itl/)

完成标准：能解释首个输出 token 在哪里产生，能用算术强度而非口号分析 Prefill 与 Decode，并知道 Chunked Prefill 的收益和代价。

### 第三阶段：建立 KV Cache 显存账本

1. [KV Cache 显存账本：形状、容量与生命周期](/blog/posts/ai-infra/concepts/03-kv-cache-ledger/)
2. [实验：构建 KV Cache 容量计算器并验证 Block 浪费](/blog/posts/ai-infra/labs/02-kv-capacity-planner/)
3. 面试检验：
   - [如何计算 KV Cache 显存？](/blog/posts/ai-infra/questions/05-kv-cache-calculation/)
   - [模型权重能装进 GPU，为什么在线服务仍会 OOM？](/blog/posts/ai-infra/questions/06-weights-fit-but-oom/)

完成标准：能从 `num_layers`、`num_kv_heads`、`head_dim` 和精度推导 bytes/token，并能把权重、KV Pool、Workspace、CUDA Graph、通信 Buffer 和分配器预留放进同一份账本。

### 第四阶段：理解分页 KV 管理与调度耦合

1. [PagedAttention：块表、分配、共享与抢占](/blog/posts/ai-infra/concepts/04-paged-attention/)
2. 面试检验：
   - [PagedAttention 减少了什么，又没有减少什么？](/blog/posts/ai-infra/questions/07-explain-paged-attention/)
   - [Block Size 越小越好吗？Prefix Cache 命中后为何 Decode 不一定更快？](/blog/posts/ai-infra/questions/08-block-size-prefix-cache/)

完成标准：能在白板上画出 Logical Block、Physical Block 和 Block Table，明确它不是 OS Page Fault，也不减少理论 KV bytes/token。

## 后续建设顺序

| 优先级 | 模块 | 核心问题 | 预期实践 |
|---|---|---|---|
| P0 | GPU 与性能模型 | HBM/L2/Shared/Register、Warp、GEMM、Roofline、FlashAttention | 用 Profiler 证明瓶颈位置 |
| P0 | 推理引擎 | Scheduler、Continuous Batching、Chunked Prefill、Prefix Cache、抢占 | 阅读并改动 vLLM V1 调度路径 |
| P0 | 量化与投机解码 | W/A/KV 量化、校准、精度与性能权衡 | 对比延迟、吞吐、显存和精度 |
| P1 | 分布式推理 | TP/PP/DP/EP、NCCL、拓扑、PD 分离、KV 传输 | 建立计算通信成本模型 |
| P1 | 生产 Serving | 队列、流式输出、取消、超时、限流、SLO、可观测性 | 多副本压测与故障注入 |
| P2 | 前沿专题 | MLA、EPLB、FP4、TMA、Mooncake/LMCache | 标注版本并按季度复核 |

## 面经如何影响这条路线

对核验的 10 份 2026 年推理 Infra 定向样本按“是否明确提及该能力”编码（同一份面经可命中多个标签）后，现场编码、项目深挖、Attention/KV、量化、GPU/CUDA、Profiling/Roofline、引擎调度与缓存、并行通信与 PD 分离都反复出现。下面保留样本清单，便于复核；它是岗位方向的排序信号，不是行业总体统计。

这说明目标能力不是“会调用 vLLM API”，而是能把项目、性能模型、源码、CUDA、量化、调度和系统能力组合起来。样本来自匿名个人经历，只用于排序，不代表公司固定题库或录用标准。

样本清单：

- S1 [AI infra 推理方向日常实习面经总结](https://www.nowcoder.com/feed/main/detail/ebeea95fa44a4eceb1c2890022a6bb2e)
- S2 [AI infra 小厂实习面经](https://www.nowcoder.com/feed/main/detail/c7eee5b04fb8424aa4847f0e21fab875)
- S3 [AI Infra 面经 攒人品版](https://www.nowcoder.com/feed/main/detail/d695614a06424c148c04586ac3a66e78)
- S4 [快手实习 AI Infra 一面面经](https://www.nowcoder.com/feed/main/detail/b77d789783d04541ab39972a59832bb2)
- S5 [智谱 AI infra 一面面经](https://www.nowcoder.com/feed/main/detail/846a09e34fea4fe9a7e14da2a88e3f72)
- S6 [快手校招 AI infra 面经分享](https://www.nowcoder.com/feed/main/detail/c582fdfbc29d4c93ac9044005ad0a311)
- S7 [字节校招 AI infra 后端面经](https://www.nowcoder.com/feed/main/detail/45814e935b894c4fb13d5a72c81e23a6)
- S8 [字节 AI infra 校招一面 1h](https://www.nowcoder.com/feed/main/detail/bb24ea24216242ae8fb57557cfbc1bbd)
- S9 [快手 AI Infra 校招面经](https://www.nowcoder.com/feed/main/detail/eccb5cafdfce452c8d56374ef070685d)
- S10 [摩尔线程 AI infra 二面实习面经](https://www.nowcoder.com/feed/main/detail/e1ac5e7fccf243c5898394d80911b3c0)

## 内容质量约定

- 稳定原理与快速变化的框架实现分开写，版本敏感内容记录核验日期。
- 社区文章负责帮助理解；论文和官方文档负责技术定论；源码和可复现实验负责验证。
- 性能结论必须同时给出模型、硬件、精度、负载、测量边界和统计分布。
- 面试答案必须包含假设、机制、权衡和失效边界，避免只有背诵要点。
- 每学完一个模块，至少留下一个可运行实验或一份可审计计算。

## 参考资料

### 岗位与社区

- [入局 AI Infra：程序员必须了解的 AI 系统设计与挑战](https://zhuanlan.zhihu.com/p/1929555810737955192)
- [聊聊大模型推理服务中的优化问题](https://zhuanlan.zhihu.com/p/677650022)
- [开源大模型推理引擎现状及常见推理优化方法](https://zhuanlan.zhihu.com/p/755874470)
- [SGLang：LLM 推理引擎发展新方向](https://zhuanlan.zhihu.com/p/711378550)
- [AI Infra 推理方向日常实习面经总结](https://www.nowcoder.com/feed/main/detail/ebeea95fa44a4eceb1c2890022a6bb2e)
- [AI Infra 小厂实习面经](https://www.nowcoder.com/feed/main/detail/c7eee5b04fb8424aa4847f0e21fab875)

社区内容可能随版本过时，阅读后应继续回到文章所列的一手资料和实验。
