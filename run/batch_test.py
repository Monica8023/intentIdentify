"""
/compare 接口性能测试脚本
========================
用法:
  python batch_test.py                                # 默认 10 并发，内置语料
  python batch_test.py --concurrency 20               # 20 并发
  python batch_test.py --csv d:/work/table/意图真实数据.csv  # 从CSV读取测试文本
  python batch_test.py --url http://10.0.0.1:8000     # 指定服务地址

输出:
  1. 控制台实时打印每条请求的耗时和响应结果
  2. 测试结束后打印汇总统计
  3. 结果保存到 JSON 文件 (compare_test_result_时间戳.json)
"""

import argparse
import csv
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict

import requests

# ==================== 默认配置 ====================
DEFAULT_URL = "http://localhost:8000"
DEFAULT_CONCURRENCY = 10
DEFAULT_TOP_K = 4
DEFAULT_TIMEOUT = 30

# ==================== 内置测试语料 ====================
BUILTIN_TEXTS = [
    "不需要，谢谢",
    "不用了，我们不找",
    "不需要，别打了",
    "我们不用，谢谢",
    "喂",
    "你好",
    "嗯",
    "喂？你好",
    "你是哪家公司",
    "你们怎么收费",
    "我先了解一下",
    "你把资料发我",
    "我证书在单位用",
    "我证书还没到期",
    "我证书自己用",
    "暂时不需要，谢谢",
    "目前不用，以后再说",
    "现在不需要，谢谢",
    "您拨打的用户无法接听",
    "电话无人接听",
    "用户暂时无法接通",
    "你们具体做什么",
    "具体怎么收费",
    "你们有什么优势",
    "你帮我推荐一下",
    "帮我找个单位",
    "现在多少钱一年",
    "价格多少",
    "费用怎么算",
    "你们是哪里的公司",
    "你们在哪个城市",
    "我们已经合作了",
    "已经在做了",
    "我不太清楚",
    "我不确定",
    "我想咨询一下",
    "我咨询一下价格",
    "你发我资料看看",
    "把资料发我微信",
    "什么时候能办好",
    "多久能弄好",
    "以后再联系",
    "改天再说吧",
    "你好我想了解一下你们的证书价格",
    "我不太清楚你们做什么的但好像需要",
    "我证书在单位不过也可以看看",
    "先不办手续改天联系你",
    "你们和我之前合作那家有什么区别",
]


def load_csv(path: str) -> List[str]:
    """从 CSV 文件加载测试文本"""
    texts = []
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb2312"]:
        try:
            with open(path, "r", encoding=enc) as f:
                reader = csv.DictReader(f)
                if "text" in (reader.fieldnames or []):
                    for row in reader:
                        t = (row.get("text") or "").strip()
                        if t:
                            texts.append(t)
                else:
                    f.seek(0)
                    for row in csv.reader(f):
                        if row and row[0].strip():
                            texts.append(row[0].strip())
            break
        except UnicodeDecodeError:
            continue
    return texts


def do_request(base_url: str, text: str, top_k: int, timeout: int) -> Dict:
    """发送一次 /compare 请求，记录耗时与完整响应"""
    url = f"{base_url.rstrip('/')}/compare"
    record = {
        "text": text,
        "success": False,
        "status_code": None,
        "client_latency_ms": None,
        "response": None,
        "error": None,
    }
    start = time.perf_counter()
    try:
        resp = requests.post(url, json={"text": text, "top_k": top_k}, timeout=timeout)
        record["client_latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
        record["status_code"] = resp.status_code
        if resp.status_code == 200:
            record["success"] = True
            record["response"] = resp.json()
        else:
            record["error"] = resp.text
    except Exception as e:
        record["client_latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
        record["error"] = str(e)
    return record


def run_test(base_url: str, texts: List[str], concurrency: int, top_k: int, timeout: int):
    """执行批量测试并实时打印每条结果"""
    total = len(texts)
    print(f"\n{'=' * 72}")
    print(f"  /compare 性能测试")
    print(f"  服务地址: {base_url}   并发: {concurrency}   请求数: {total}   top_k: {top_k}")
    print(f"{'=' * 72}\n")

    results: List[Dict] = []
    batch_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(do_request, base_url, t, top_k, timeout): t for t in texts}
        for idx, future in enumerate(as_completed(futures), 1):
            r = future.result()
            results.append(r)

            if r["success"]:
                matches = r["response"].get("matches", [])
                confidence = r["response"].get("confidence_status", "")
                server_total = r["response"].get("time_cost_ms", {}).get("total", "-")
                print(
                    f"\n  [{idx:>3}/{total}] ✓ "
                    f"客户端={r['client_latency_ms']:>7.1f}ms  "
                    f"服务端={str(server_total):>7s}ms  "
                    f"置信度={confidence}"
                )
                print(f"           查询文本: {r['text']}")
                if matches:
                    print(f"           匹配结果:")
                    for rank, m in enumerate(matches, 1):
                        print(
                            f"             Top{rank}: "
                            f"label={m['label']}  "
                            f"score={m['score']:.4f}  "
                            f"text={m.get('text', '')}"
                        )
                else:
                    print(f"           匹配结果: 无")
            else:
                print(
                    f"\n  [{idx:>3}/{total}] ✗ "
                    f"客户端={r['client_latency_ms']:>7.1f}ms  "
                    f"错误={r['error'][:60]}"
                )
                print(f"           查询文本: {r['text']}")

    batch_ms = (time.perf_counter() - batch_start) * 1000

    # ==================== 汇总统计 ====================
    ok = [r for r in results if r["success"]]
    fail = [r for r in results if not r["success"]]
    ok_lats = [r["client_latency_ms"] for r in ok]
    server_totals = [
        r["response"]["time_cost_ms"]["total"]
        for r in ok if r["response"].get("time_cost_ms", {}).get("total") is not None
    ]

    def pct(arr, p):
        return round(sorted(arr)[int(len(arr) * p)], 2) if arr else 0

    def stats_block(name, arr):
        if not arr:
            return {}
        return {
            "count": len(arr),
            "min": round(min(arr), 2),
            "max": round(max(arr), 2),
            "mean": round(statistics.mean(arr), 2),
            "median": round(statistics.median(arr), 2),
            "p90": pct(arr, 0.9),
            "p95": pct(arr, 0.95),
            "p99": pct(arr, 0.99),
            "stdev": round(statistics.stdev(arr), 2) if len(arr) > 1 else 0,
        }

    client_stats = stats_block("客户端延迟", ok_lats)
    server_stats = stats_block("服务端延迟", server_totals)

    # 服务端各阶段
    phase_names = ["model_encode", "build_request", "milvus_search", "result_process"]
    phase_data = {}
    for pn in phase_names:
        vals = [
            r["response"]["time_cost_ms"][pn]
            for r in ok
            if r["response"].get("time_cost_ms", {}).get(pn) is not None
        ]
        if vals:
            phase_data[pn] = stats_block(pn, vals)

    # 置信度分布
    conf_dist = {}
    for r in ok:
        s = r["response"].get("confidence_status", "UNKNOWN")
        conf_dist[s] = conf_dist.get(s, 0) + 1

    qps = round(total / (batch_ms / 1000), 2) if batch_ms > 0 else 0

    # 打印报告
    print(f"\n{'─' * 72}")
    print(f"  📊 测试结果汇总")
    print(f"{'─' * 72}")
    print(f"  成功: {len(ok)}   失败: {len(fail)}   成功率: {len(ok)/total*100:.1f}%")
    print(f"  总耗时: {batch_ms:.1f}ms   吞吐量: {qps} QPS")

    if client_stats:
        cs = client_stats
        print(f"\n  ▸ 客户端延迟(ms)")
        print(f"    min={cs['min']}  max={cs['max']}  mean={cs['mean']}  "
              f"median={cs['median']}  P90={cs['p90']}  P95={cs['p95']}  P99={cs['p99']}")

    if server_stats:
        ss = server_stats
        print(f"\n  ▸ 服务端总延迟(ms)")
        print(f"    min={ss['min']}  max={ss['max']}  mean={ss['mean']}  "
              f"median={ss['median']}  P90={ss['p90']}  P95={ss['p95']}  P99={ss['p99']}")

    if phase_data:
        print(f"\n  ▸ 服务端各阶段平均耗时(ms)")
        print(f"    {'阶段':<20s} {'mean':>8s} {'median':>8s} {'P90':>8s} {'P95':>8s} {'max':>8s}")
        for pn in phase_names:
            if pn in phase_data:
                d = phase_data[pn]
                print(f"    {pn:<20s} {d['mean']:>8.2f} {d['median']:>8.2f} "
                      f"{d['p90']:>8.2f} {d['p95']:>8.2f} {d['max']:>8.2f}")

    if conf_dist:
        print(f"\n  ▸ 置信度分布")
        for k, v in sorted(conf_dist.items(), key=lambda x: -x[1]):
            print(f"    {k:<25s} {v:>4d}")

    print(f"{'─' * 72}\n")

    # ==================== 保存到 JSON ====================
    summary = {
        "test_time": datetime.now().isoformat(),
        "config": {"url": base_url, "concurrency": concurrency, "top_k": top_k, "total": total},
        "overview": {
            "success": len(ok), "fail": len(fail),
            "success_rate": f"{len(ok)/total*100:.1f}%",
            "batch_duration_ms": round(batch_ms, 2), "qps": qps,
        },
        "client_latency": client_stats,
        "server_latency": server_stats,
        "server_phases": phase_data,
        "confidence_distribution": conf_dist,
        "details": results,
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"compare_test_result_{ts}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"  💾 详细结果已保存: {out_file}")
    print(f"  🏁 测试完成!\n")


def main():
    parser = argparse.ArgumentParser(description="/compare 接口性能测试")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"服务地址 (默认 {DEFAULT_URL})")
    parser.add_argument("--concurrency", "-c", type=int, default=DEFAULT_CONCURRENCY, help="并发数 (默认 10)")
    parser.add_argument("--top_k", "-k", type=int, default=DEFAULT_TOP_K, help="top_k (默认 4)")
    parser.add_argument("--csv", default=None, help="从CSV文件读取测试文本 (需含text列)")
    parser.add_argument("--repeat", "-r", type=int, default=1, help="每条文本重复次数 (默认 1)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="请求超时秒数 (默认 30)")
    args = parser.parse_args()

    # 加载测试数据
    if args.csv:
        texts = load_csv(args.csv)
        if not texts:
            print(f"  ❌ 无法从 {args.csv} 读取有效数据"); sys.exit(1)
        print(f"  📂 从CSV加载 {len(texts)} 条测试文本")
    else:
        texts = BUILTIN_TEXTS
        print(f"  📦 使用内置 {len(texts)} 条测试语料")

    if args.repeat > 1:
        texts = texts * args.repeat
        print(f"  🔁 重复{args.repeat}次，总计 {len(texts)} 条")

    # 健康检查
    print(f"  ⏳ 检查服务 {args.url} ...")
    try:
        r = requests.get(f"{args.url.rstrip('/')}/health", timeout=5)
        if r.status_code == 200:
            print(f"  ✅ 服务正常")
        else:
            print(f"  ⚠️ 服务返回 {r.status_code}")
    except Exception as e:
        print(f"  ❌ 无法连接服务: {e}"); sys.exit(1)

    run_test(args.url, texts, args.concurrency, args.top_k, args.timeout)


if __name__ == "__main__":
    main()
