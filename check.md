# 意图识别服务 — 参数调优手册

## 一、参数总览

### 阶段 1 — Milvus 粗排（向量召回）

| 参数 | 位置 | 默认值 | 取值范围 | 说明 |
|---|---|---|---|---|
| `recall_limit` | Intent.py:545/687 | `max(20, top_k*2)` | 整数，最小 20，最大 40 | 粗排候选集大小，决定给精排提供多少候选 |
| `ef`（dense 搜索） | Intent.py:550/692 | `64` | `16~512`（越大越准但越慢） | HNSW 搜索宽度，影响稠密向量的召回质量 |
| `drop_ratio_search`（sparse） | Intent.py:555/697 | `0.2` | `0.0~0.9` | 稀疏向量搜索时丢弃低权重 token 的比例，越小越精确 |
| `drop_ratio_build`（索引构建） | HyBridSearch.py:281 | `0.2` | `0.0~0.5` | 构建稀疏索引时丢弃低权重 token，影响稀疏向量召回上限（改后需重建索引） |
| `M`（HNSW 图层数） | HyBridSearch.py:278 | `16` | `4~64` | 越大图连接越密，召回质量越高，内存/构建时间越多（改后需重建索引） |
| `efConstruction`（HNSW 构建） | HyBridSearch.py:278 | `256` | `64~512` | 构建时搜索宽度，越大索引质量越高（改后需重建索引） |

### 阶段 2 — RRF 融合

| 参数 | 位置 | 默认值 | 取值范围 | 说明 |
|---|---|---|---|---|
| `RRFRanker()` 的 `k` 值 | Intent.py:564/703 | `60`（Milvus 默认） | `1~∞`（通常 10~100） | RRF 公式中的调和常数，小 k 偏向高排名结果更极端。调整示例：`RRFRanker(k=20)` |

### 阶段 3 — Reranker 精排

| 参数 | 位置 | 默认值 | 取值范围 | 说明 |
|---|---|---|---|---|
| `top_k`（/compare） | Intent.py:193 | `4` | `1~20` | 精排后保留的最终候选数 |
| `RECOGNIZE_TOP_K`（/recognize） | Intent.py:669 | `2` | 整数 ≥ 1 | 生产接口固定精排 top-k |

### 阶段 4 — 置信度判决（Nacos 热更新）

| 参数 | Nacos key | 默认值 | 取值范围 | 说明 |
|---|---|---|---|---|
| `low_score_threshold` | `low_score_threshold` | `0.30` | `0.0~1.0` | Top-1 概率低于此值 → 返回 `intent_unknown` |
| `high_gap_threshold` | `high_gap_threshold` | `0.20` | `0.0~1.0` | Top-1 与 Top-2 概率差高于此值 → `HIGH_CONFIDENCE` |

### 阶段 5 — type 过滤（/recognize 专有）

| 参数 | 位置 | 说明 |
|---|---|---|
| `type_threshold` | IntentRequest 请求字段 | 文本字数 ≤ 阈值时用 `type=0`（短话术），否则用 `type=1`（长话术）。决定搜索哪一批语料 |

---

## 二、参数-现象对照表

| 现象 | 根因参数 | 调整方向 |
|---|---|---|
| 完全不相关的文本命中（误识别） | `low_score_threshold` 太低 | **调高**，如 `0.30 → 0.50` |
| 意图明明正确，却返回 `intent_unknown` | `low_score_threshold` 太高 | **调低**，如 `0.30 → 0.20` |
| Top-1 和 Top-2 概率很接近，结果不稳 | `high_gap_threshold` 太低，没有充分过滤模糊意图 | **调高**，如 `0.20 → 0.30` |
| 正确意图总是排第二位（排序不准） | `recall_limit` 太小，精排候选不足 | **调大** recall_limit；或**降低** `drop_ratio_search` |
| 稀疏向量（关键词）匹配失效 | `drop_ratio_search=0.2` 过滤掉了重要 token | **调低**至 `0.05~0.1` |
| 向量搜索结果本身就不准（粗排质量差） | `ef=64` 太小 | **调大** ef，如 `64 → 128` |
| 短文本和长文本互相干扰 | `type_threshold` 设置不合理或语料 type 标注混乱 | 检查 `type_threshold` 与语料 type 字段是否对应 |
| 相似度高但不同意图的样本都被召回 | 语料设计问题，或 `recall_limit` 太大导致噪声太多 | **缩小** recall_limit；检查语料区分度 |

---

## 三、调参优先级

```
最快生效（Nacos 热更新，无需重启）：
  low_score_threshold  ← 最直接影响误识别/漏识别
  high_gap_threshold   ← 控制模糊拒识

需要重启服务才生效：
  ef / drop_ratio_search / recall_limit
  RECOGNIZE_TOP_K / top_k

需要重建索引才生效（影响最底层）：
  drop_ratio_build / M / efConstruction
```

**推荐第一步**：先把 `low_score_threshold` 从 `0.30` 提高到 `0.45~0.55` 来抑制完全不相关文本命中，再通过 `/compare` 接口观察 `probability` 分布，找到合适的分界线。
