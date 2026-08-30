---
title: "先定义再优化：在线推理指标与测量契约"
published: 2026-08-12
description: "统一 TTFT、ITL、TPOT、吞吐与 Goodput 的定义，建立所有性能讨论都能复现的测量边界。"
tags: ["AI Infra", "LLM 推理", "概念", "TTFT", "TPOT", "ITL", "Goodput", "Benchmark"]
category: "AI Infra · 系统学习"
draft: true
lang: zh_CN
comment: true
---

| 类型 | 难度 | 预计时间 | 前置知识 | 核验日期 |
|---|---|---:|---|---|
| 概念 | 基础 | 32 分钟 | 基础统计；HTTP 请求的基本过程 | 2026-08-13 |

性能优化的第一步不是改参数，而是约定**测量的对象、边界和负载**。如果两份报告对 TTFT 的起点定义不同，后面的数字即使精确到微秒，也无法比较。

> 30 秒结论：延迟指标描述单个请求，吞吐描述系统完成工作的速率，Goodput 描述满足 SLO 的有效速率。任何性能数字都必须和负载、模型、硬件、输入输出分布及计时边界一起出现。

## 一条请求的时间线

先定义事件：

```text
t_arrival    t_scheduled       t_first    t_2       t_3       t_done
    |              |               |       |         |           |
    |---- queue ---|--- prefill ----|-- decode / stream ----------|
    |<---------------------- E2E latency ------------------------->|
    |<----------- TTFT ----------->|
                                    |<-ITL_2->|<-ITL_3->|
```

- `t_arrival`：服务端接受请求，或客户端开始发起请求。二者必须选一个并写明。
- `t_scheduled`：请求首次被调度进一次模型执行。
- `t_first`：第一个输出 token 在测量边界可见。
- `t_i`：第 `i` 个输出 token 可见。
- `t_done`：最后一个 token 或终止事件可见。

网络、反向代理和客户端流式缓冲都会改变“可见”的时刻。因此客户端 TTFT 和引擎内部 TTFT 是两个指标，不能混写。

## 延迟指标

### TTFT

Time to First Token：

```text
TTFT = t_first - t_arrival
```

它通常包含排队和 Prefill，但是否包含 tokenization、网络和反向代理取决于测量边界。TTFT 对交互式应用非常重要，但不能单独描述后续生成是否流畅。

### ITL 与 TPOT

第 `i` 个 token 的 Inter-token Latency：

```text
ITL_i = t_i - t_(i-1),  i >= 2
```

本站把 TPOT 定义为首 token 之后的平均每 token 时间：

```text
TPOT = (t_last - t_first) / (N_out - 1),  N_out >= 2
```

当输出只有一个 token 时，TPOT 未定义，而不是零。TPOT 是平均值，会隐藏卡顿；分析流式体验时还要观察 ITL 的 P50、P95、P99 和时间序列。

在相同计时边界、没有额外收尾开销并采用上述 TPOT 定义时，可以近似写成：

```text
E2E ~= TTFT + (N_out - 1) * TPOT
```

这是一条核对账目的近似式，不是普遍定律。

## 吞吐、并发与负载

常见吞吐至少有四种：

| 指标 | 定义 | 容易误解的地方 |
|---|---|---|
| 请求吞吐 | 完成请求数 / 秒 | 请求长度不同，工作量不可直接比较 |
| 输入吞吐 | 处理的输入 token / 秒 | 偏向衡量 Prefill 工作 |
| 输出吞吐 | 生成的输出 token / 秒 | 不能反映单请求卡顿 |
| 总 token 吞吐 | 输入与输出 token 之和 / 秒 | 两类 token 的计算特征不同 |

`concurrency` 是系统内同时存在的请求数，`request rate` 是到达速率。封闭负载通常固定并发：请求完成后客户端再发一个；开放负载按时间独立地产生请求。前者会自我限速，容易低估过载时的排队和尾延迟。

## Goodput：满足目标的有效吞吐

为了让 Goodput 能从逐请求数据直接计算，本文采用**逐请求硬阈值**作为通过条件：例如请求的 `TTFT <= 800 ms`，且该请求的 `ITL P99 <= 80 ms`。其中请求内 ITL 至少有两个样本；输出只有一个 token 时，ITL 条件必须按业务策略明确记为“不适用”或单独放宽。

```text
Goodput = 统计窗口内满足全部逐请求阈值的完成请求数 / 窗口时长
```

例如在 `TTFT <= 800 ms`、请求内 `ITL P99 <= 80 ms` 的口径下，100 个请求/秒中有 40 个请求超出任一阈值，则 Goodput 为 60 req/s。

这和常见的**窗口级百分位 SLO**不是一回事。`TTFT P99 <= 800 ms` 描述整个统计窗口的分布，不能据此把某一个请求标记为“违反 P99”；窗口级 SLO 应单独报告达标/未达标，或报告在该约束下的最大可承载请求率。发布性能结果时必须注明采用哪一种口径。

Goodput 必须与以下信息同时记录：

- SLO 条件及判断粒度；
- 统计窗口和预热时间；
- 超时、取消和错误请求如何处理；
- 输入、输出长度分布和到达过程。

## 一份最小测量契约

```yaml
model: Qwen/Qwen3-8B
revision: <commit-or-release>
engine: vLLM
engine_version: <version>
hardware: <gpu-model-and-count>
precision: bf16
workload:
  mode: open-loop
  request_rate: 8 req/s
  input_tokens: {distribution: fixed, value: 1024}
  output_tokens: {distribution: fixed, value: 256}
measurement:
  boundary: client-observed
  warmup_seconds: 60
  duration_seconds: 600
  percentiles: [50, 95, 99]
slo:
  ttft_p99_ms: 800
  request_itl_p99_ms: 80
```

至少重复三次，并保存原始逐请求数据。只有均值而没有分布，无法判断尾延迟；只有服务端指标而没有客户端观测，可能遗漏网络和缓冲。

## 常见错误

1. 在不同输入输出长度下直接比较 token/s。
2. 把 TPOT 和某个 token 的 ITL 当作同一件事。
3. 只报告平均延迟，忽略 P99 和错误请求。
4. 用最大并发压满系统，再宣称这是线上可持续吞吐。
5. 测试客户端本身已满 CPU，却把瓶颈归因于服务端。

## 自测

- 为什么封闭负载可能掩盖排队崩溃？
- 输出一个 token 的请求该如何报告 TPOT？
- 两个系统的 output tok/s 相同，还需要哪些信息才能判断谁更适合在线聊天？

## 参考资料

- [GenAI-Perf Metrics](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html)（官方文档）
- [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)（论文）
- [vLLM Metrics](https://docs.vllm.ai/en/latest/design/metrics/)（官方文档）
