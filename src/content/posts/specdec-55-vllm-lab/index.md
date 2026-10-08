---
title: "5.5 动手实验：vLLM 实测代码生成 vs 开放对话的接受率"
published: 2026-10-07T19:00:00
description: "本章收官实验：用 vLLM 搭建投机解码基准，同一台机器上对照代码生成与开放对话两类负载的接受率、接受长度与端到端加速，附 n-gram 草稿对照与在线服务 Prometheus 观测方案。"
tags: [投机解码, 推理优化, vLLM, 实验, AIInfraGuide]
category: AIInfraGuide·投机解码
author: pplk
draft: false
---

# 5.5 动手实验：vLLM 实测代码生成 vs 开放对话的接受率

> **系列导航｜《AIInfraGuide》模块四·第 5 章：Speculative Decoding**
> 1. [5.1 Speculative Decoding 核心原理：先猜后验的无损加速](../specdec-51-speculative-sampling/)
> 2. [5.2 Draft 模型方案：独立小模型的选型、对齐与接受率](../specdec-52-draft-model/)
> 3. [5.3 Self-Draft 方案：Medusa 多头预测与 EAGLE-2 动态草稿树](../specdec-53-medusa-eagle2/)
> 4. [5.4 收益边界与限制：什么时候投机解码不划算](../specdec-54-limits/)
> 5. **5.5 动手实验：vLLM 实测代码生成 vs 开放对话的接受率（本篇）**

## 本章简介

本篇是《AIInfraGuide》模块四"推理优化"第 5 章"Speculative Decoding"的第 5 篇（共 5 篇），也是本章收官。前四篇建立了全部理论工具：Draft–Verify 框架、无损的拒绝采样、α/γ/c 加速比公式、draft 选型、Self-Draft、收益边界。本篇把第 5.4 篇最核心的论断——**"接受率是任务属性"**——变成你自己机器上的数字。

实验假设（来自 5.4 第 1 节）：**代码生成负载的接受率与加速比显著高于开放对话负载；n-gram 草稿在代码负载上能拿到接近模型草稿的收益。**

读完你应该能：在自己机器上搭起一个可复现的投机解码基准；用离线 API 与在线服务两条路径测出接受率、接受长度与端到端加速；并能把同一套流程搬到自己业务的流量上，为"要不要开投机"提供数据支撑。

## 1. 实验设计

| 要素 | 设计 |
|---|---|
| Target 模型 | Qwen2.5-7B-Instruct（FP16，单卡可跑；有 GPU 条件可换 32B/72B 复测） |
| Draft | ① Qwen2.5-0.5B-Instruct（同族小模型，5.2 路线一）② n-gram（免模型，5.2 路线三） |
| 负载 | 代码生成 6 题 vs 开放对话 6 题，均贪心解码、max_tokens=256 |
| 变量 | 草稿类型（无 / 小模型 / n-gram）× 负载类型（代码 / 对话），共 6 组 |
| 指标 | 接受率 α、平均接受长度 τ、输出吞吐 tok/s、相对基线加速比 |
| 控制变量 | 同机、同卡、同 vLLM 版本、固定输出长度、先 warmup 再计时 |

两条提醒：贪心解码下输出应与基线**逐 token 一致**（5.1 篇第 3.4 节），实验脚本会顺带验证这一点；所有数字都只对"这台机器 + 这个版本 + 这个负载"负责——结论的可迁移性来自方法，不来自数字本身。

## 2. 环境准备

```bash
# 测试环境：单卡 H100/A100（≥24GB 显存即可跑 7B+0.5B），CUDA 12.4，
# vLLM 0.10.x，torch 2.5.x（核验于 2026-10-07；接口随版本演进，以所用版本文档为准）
pip install "vllm>=0.10" transformers

# 模型（首次运行自动下载，也可提前 huggingface-cli download）
# target: Qwen/Qwen2.5-7B-Instruct   draft: Qwen/Qwen2.5-0.5B-Instruct
```

## 3. 实验一：离线批量对比（核心实验）

```python
# spec_bench.py —— 每个配置独立进程运行（核验于 2026-10-07）
# 注意：vLLM 引擎在进程退出前不释放显存与 worker 资源，
# 同一进程内串行实例化多个 LLM 几乎必然 OOM/挂起，必须逐配置起独立进程
import json, sys, time
from vllm import LLM, SamplingParams

CODE_PROMPTS = [
    "用 Python 实现快速排序，要求原地排序，附类型注解与 docstring。",
    "写一个 Python 装饰器，统计函数运行时间并打印日志。",
    "用 Python 实现 LRU Cache，get/put 均 O(1)。",
    "写一个函数，把嵌套 JSON 扁平化成一层字典，key 用点号连接。",
    "用 Python 实现二分查找，处理边界并返回插入位置。",
    "写一个生成器，逐行读取大文件并按行号 yield。",
]

CHAT_PROMPTS = [
    "给我讲讲你为什么喜欢秋天，随便聊聊。",
    "如果明天不用工作，你理想的周末会怎么过？",
    "给我三个提高工作效率的原创建议，并解释背后的原因。",
    "写一段关于深夜便利店的短文，风格随意。",
    "和朋友争论'咖啡好还是茶好'，帮我各想两个论点。",
    "讲讲一个你最近想到的有趣点子，什么都可以。",
]

def main():
    # payload: {"prompts": "code"|"chat", "spec": null|{...}, "tag": "...", "dump": "out.txt"}
    payload = json.loads(sys.argv[1])
    prompts = CODE_PROMPTS if payload["prompts"] == "code" else CHAT_PROMPTS
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        dtype="float16",
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        speculative_config=payload["spec"],   # None = 关闭投机解码
        disable_log_stats=False,               # 保留 SpecDecoding 统计日志
    )
    sp = SamplingParams(temperature=0.0, max_tokens=256)  # 贪心：输出应与基线逐 token 一致
    llm.generate(prompts[:2], sp)              # warmup，不计时
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp)
    dt = time.perf_counter() - t0
    n_tok = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"[{payload['tag']}] {n_tok} tokens in {dt:.2f}s -> {n_tok/dt:.1f} tok/s")
    if payload.get("dump"):                    # 导出 token 序列，供无损性 diff
        with open(payload["dump"], "w") as f:
            for o in outs:
                f.write(" ".join(map(str, o.outputs[0].token_ids)) + "\n")

if __name__ == "__main__":
    main()
```

用 shell 逐配置起独立进程驱动（每组都是全新进程，显存随进程退出释放）：

```bash
python spec_bench.py '{"prompts":"code","spec":null,"tag":"code/基线","dump":"base_code.txt"}'
python spec_bench.py '{"prompts":"code","spec":{"model":"Qwen/Qwen2.5-0.5B-Instruct","num_speculative_tokens":5},"tag":"code/小模型draft","dump":"spec_code.txt"}'
python spec_bench.py '{"prompts":"code","spec":{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":1},"tag":"code/ngram"}'
python spec_bench.py '{"prompts":"chat","spec":null,"tag":"chat/基线","dump":"base_chat.txt"}'
python spec_bench.py '{"prompts":"chat","spec":{"model":"Qwen/Qwen2.5-0.5B-Instruct","num_speculative_tokens":5},"tag":"chat/小模型draft","dump":"spec_chat.txt"}'

# 贪心无损校验：基线与投机的 token 序列应逐 token 一致（diff 无输出即通过）
diff base_code.txt spec_code.txt && echo "code 逐 token 一致"
diff base_chat.txt spec_chat.txt && echo "chat 逐 token 一致"
```

运行期间关注日志中的 SpecDecoding 统计行（V1 引擎周期性打印，格式随版本变化，示例如下）：

```text
# 典型输出（数字为示意，参考值，实测随硬件/版本/负载变化）：
# —— vLLM 周期性统计行（混在引擎日志里）：
... Draft acceptance rate: ~65%, Mean acceptance length: ~2.8   # code/小模型draft 进程
... Draft acceptance rate: ~45%, Mean acceptance length: ~2.1   # chat/小模型draft 进程
# —— 脚本自身的计时输出：
[code/基线] 1536 tokens in 21.0s -> 73.1 tok/s
[code/小模型draft] 1536 tokens in 11.2s -> 137.1 tok/s
[code/ngram] 1536 tokens in 12.5s -> 122.9 tok/s
[chat/基线] 1536 tokens in 21.0s -> 73.1 tok/s
[chat/小模型draft] 1536 tokens in 15.8s -> 97.2 tok/s
# —— diff 无损校验：
code 逐 token 一致
chat 逐 token 一致
# —— 手工对照 tok/s：code=1.88x（ngram=1.68x），chat=1.33x
```

如果数字长成这个样子，实验假设成立：代码负载 α 高、加速接近 2×，连零成本的 n-gram 都能拿到 1.7× 左右；开放对话 α 明显低，加速缩到 1.3× 上下。**你的数字几乎必然不同，但"代码 > 对话、n-gram 在代码上有效"这两个相对关系应该在绝大多数环境复现**——不复现本身就是值得排查的信号（draft 没对齐？负载太短？温度没归零？）。

## 4. 实验二：在线服务与 Prometheus 观测

离线脚本能回答"快多少"，回答不了"在线负载下接受率怎么变"。上服务：

```bash
# 启动带投机解码的在线服务（核验于 2026-10-07；旧版本用 --speculative-model 等独立参数，
# 新版本统一为 --speculative-config JSON，以所用版本文档为准）
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --speculative-config '{"model": "Qwen/Qwen2.5-0.5B-Instruct", "num_speculative_tokens": 5}' \
  --max-model-len 4096 \
  --port 8000

# 另开一个对照服务（同机换端口或先后运行），不带 --speculative-config
```

接受率的在线观测走 Prometheus（`/metrics` 端点）：

```text
# V1 引擎暴露的累计计数（指标名随版本可能调整，以所用版本 /metrics 实际输出为准）：
vllm:spec_decode_num_drafts_total            # 草稿轮数（投机迭代次数）
vllm:spec_decode_num_draft_tokens_total      # 草稿 token 总数
vllm:spec_decode_num_accepted_tokens_total   # 被接受 token 总数

# 接受率 α = accepted_tokens / draft_tokens
# 平均接受长度 τ = 1 + accepted_tokens / drafts
#   （每轮必有 1 个 bonus/修正 token + 平均每轮接受的草稿 token 数）
```

压测用 vLLM 自带脚本，两类负载分开打（`--dataset-name sharegpt` 模拟对话、自建代码 prompt 集模拟代码），对照两个服务的 TTFT/TPOT/吞吐：

```bash
python benchmarks/benchmark_serving.py \
  --backend openai-chat --model Qwen/Qwen2.5-7B-Instruct \
  --dataset-name sharegpt --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 200 --request-rate 4 --port 8000
```

观测要点（对应 5.4 篇的边界条件）：

1. **低并发（request-rate 1~4）**：投机服务的 TPOT/ITL 应明显低于对照，接受率与离线实验一致；
2. **拉高并发（request-rate 16+）**：盯两件事——验证税是否让 TPOT 优势收窄甚至反转（batch 撞上 Roofline 的 B*），以及吞吐曲线是否被拖低；
3. **动态降档**：配置 `disable_by_batch_size`（如 32）重跑高压组，验证引擎在超阈值后自动退回普通 decode，TPOT 应回到基线水平——这是 5.4 决策清单里"按负载自动降档"的落地验证。

## 5. 进阶：换 EAGLE 头复测

有额外算力时，把 draft 换成社区训练好的 EAGLE 头（HuggingFace 检索 `EAGLE` / `EAGLE3` + 你的模型名，如 SafeAILab 为 Llama 系发布的草稿头；Qwen 系以社区实际发布为准），`speculative_config` 改为 `{"method": "eagle", "model": "<草稿头仓库>", "num_speculative_tokens": 5}`（树规模等参数随方法不同，以版本文档为准）。预期：τ 从 2.5~3 抬到 4 上下，同负载加速比再上一档——这就是 5.3 篇"候选构造质量"差异在你机器上的具象化。

## 6. 结果记录模板

实验报告建议按这个模板归档（与第 4 章量化实验同构，方便跨章对比）：

```text
日期 / 机器 / GPU / vLLM 版本 / CUDA 版本
target / draft / γ（或树配置）/ 温度
负载：任务类型、prompt 数、输入输出长度分布
基线 tok/s ｜ 投机 tok/s ｜ 加速比
α（分任务、分温度）｜ τ ｜ draft 单步耗时实测（算 c）
贪心逐 token 一致性：通过 / 不通过（不通过必须排查）
并发扫描：request-rate × TPOT/吞吐/接受率 曲线，标注 B* 失效点
结论：开 or 不开、γ 取值、disable_by_batch_size 阈值
```

## 7. 常见问题（FAQ）

**Q1：日志里看不到 SpecDecoding 统计？** 检查三处：是否设了 `disable_log_stats=True`（离线 API 默认可能关闭统计，显式设 False）；`speculative_config` 是否真的生效（启动日志会打印 draft 加载信息，没有就是配置没传对）；vLLM 版本是否支持该方法的 V1 投机路径（老版本部分方法只在 V0 可用，升级或按文档切换）。

**Q2：加速比明显低于理论公式，先查什么？** 先实测 c：单独跑 draft 模型一次前向的延迟，除以 target 单步延迟。c 如果远大于参数量比（0.5B/7B 理论约 0.07），多半是小模型的 kernel launch/调度固定开销占比过高——小模型前向只有零点几毫秒时，框架开销能把它放大两三倍。缓解：开 CUDA Graph（`enforce_eager=False` 默认即启用）、确认 draft 也走了高效 kernel。

**Q3：贪心校验失败（输出不一致）？** 这违反 5.1 的数学，必是工程问题：常见根因是 draft 与 target 词表/chat template 不一致（拿 base 版 draft 配 Instruct target 是高频踩坑），或数值精度配置（dtype、attention backend）在两组运行间不同。先对齐配置再复测。

**Q4：为什么输出短（<128 token）时加速比打折？** 每个请求的开头是接受率最差的区域（draft 还没"进入状态"），且 prefill 占比大——投机只加速 decode。短输出请求的加速比天然偏低，评估时按业务真实的输出长度分布加权。

**Q5：能直接用 ShareGPT 当"开放对话"、HumanEval 当"代码"吗？** 可以，而且比手写 prompt 更接近真实分布；本文手写 6+6 是为了零依赖可复现。正式评估换真实数据集或线上采样流量（5.2 篇的离线 replay 方法）。

## 本章小结

1. **实验验证了全章主线**：接受率是任务属性——代码负载 α 高、加速近 2×，开放对话 α 低、加速约 1.3×（示意值）；n-gram 在代码负载上以零成本拿到接近模型草稿的收益。
2. **贪心无损可工程验证**：开启投机前后输出逐 token 比对，应完全相等；不等就是配置 bug，不是数学近似。
3. **在线观测三件套**：Prometheus 的草稿/接受计数、分并发的 TPOT-吞吐曲线、`disable_by_batch_size` 降档行为——分别对应 5.4 的 α、B* 边界与调度复杂度。
4. **方法可迁移**：把 prompt 集换成你的业务流量，同一套流程就是"我的业务要不要开投机、开多大 γ"的决策依据。
5. 至此第 5 章闭环：原理（5.1）→ draft 选型（5.2）→ Self-Draft（5.3）→ 边界（5.4）→ 实测（5.5）。投机解码与第 4 章量化是 decode 优化的两条腿，叠加规则在 5.4；后续模块的 PD 分离与生产 Serving 会再次用到本章的负载边界分析。

## 延伸阅读

- **vLLM Speculative Decoding 文档**：`speculative_config` 的全部方法与参数，以及所用版本的支持矩阵——动手前必读；
- **vLLM benchmark_serving.py / benchmark_throughput.py**：本篇压测脚本的出处，参数细节以仓库版本为准；
- **Prompt Lookup Decoding**：n-gram 草稿的原始说明，理解 `prompt_lookup_max/min` 的语义；
- **本系列**：第 4.6 篇《量化选型与 vLLM 实战》（实验记录模板与压测方法同源）；第 5.1~5.4 篇（本实验全部预期的理论出处）。

## 参考文献

- vLLM Speculative Decoding 文档 — https://docs.vllm.ai/en/latest/features/spec_decode.html
- vLLM benchmark_serving.py — https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py
- vLLM Metrics 文档 — https://docs.vllm.ai/en/latest/usage/metrics.html
- Fast Inference from Transformers via Speculative Decoding — https://arxiv.org/abs/2211.17192
- EAGLE 官方仓库（预训练草稿头检索入口） — https://github.com/SafeAILab/EAGLE
- Qwen2.5 模型家族 — https://huggingface.co/Qwen
