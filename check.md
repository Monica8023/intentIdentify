1) 先调 low_score_threshold（收益最大，且可热更新）

  配置位置：run/NacosConfig.py:52
  生效逻辑：run/Intent.py:730-735

  建议从 0.30 往上试：0.35 → 0.40 → 0.45
  每次观察上述三指标。
  一般规律：阈值越高，误命中降、unknown 升。

  ---
  2) 再校准 type_threshold（只影响 /recognize）

  路由逻辑：run/Intent.py:677
  过滤条件：run/Intent.py:688

  如果短句被错分到长句库（或反过来），会明显误命中。
  建议按你真实文本长度分布，找一个分界点再试（比如 6/8/10 字）。

  ---
  3) 校验 modelId 传参是否稳定正确

  检索过滤：run/Intent.py:551-557, run/Intent.py:688
  modelId 一旦错，等于在错误语料空间里检索，误命中会很高。

  ---
  4) 只在必要时改召回参数（需要改代码）

  这些是硬编码，不在 Nacos 热更新里：
  - recall_limit = max(20, top_k*2)：run/Intent.py:545, 687
  - ef=64：run/Intent.py:550, 691
  - drop_ratio_search=0.2：run/Intent.py:555, 696

  若前两步后仍误命中高，再考虑缩召回范围或调 ANN 参数。

  ---
  5) 你要特别注意一个当前实现点

  /recognize 的 gap 判定写死了 0.1：run/Intent.py:739
  而配置里有 high_gap_threshold（run/NacosConfig.py:53），但这里没用到。
  所以你改 high_gap_threshold 目前主要影响 /compare 的状态展示，不会完整影响生产识别决策。

  ---
  6) 数据侧排查（常常比调参更关键）

  - 同一 intent_id 下是否混入“过泛文本”
  - type 标注是否一致
  - 无效语料是否仍 is_active=true

  ---