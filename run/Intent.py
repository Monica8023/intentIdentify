"""
Intent — AI 意图识别微服务（唯一服务入口）
===========================================
整合了 HyBridSearch 混合检索与 BGE Reranker 精排能力，对外暴露全部 API：
  - GET  /health           健康检查
  - GET  /ready            就绪检查
  - POST /insert           批量写入意图语料
  - DELETE /delete           根据 intent_id 批量删除
  - POST /update           批量更新（先删后增）
  - POST /upload           CSV 批量导入语料
  - POST /compare          混合检索+精排（调试/测试用）
  - POST /api/v1/recognize 意图识别（生产接口，含缓存+防击穿）

启动：
  uvicorn Intent:app --host 0.0.0.0 --port 8808
"""

import asyncio
import csv
import io
import logging
import os
import sys
import re
import time

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel, Field, ConfigDict, validator
from typing import Optional
import redis.asyncio as redis
from pymilvus import AnnSearchRequest, RRFRanker

# 确保同级目录可被 import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import HyBridSearch
from HyBridSearch import BatchInsertRequest, DeleteRequest, BatchUpdateRequest, CompareRequest
import NacosConfig
from NacosConfig import config as _cfg

# --- 基础配置 ---
MAX_TEXT_LENGTH = 500  # 防止恶意长文本

# 阈值与 Redis URL 均从 _cfg 读取（支持 Nacos 热更新），不在此硬编码

# --- 日志与应用初始化 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("intent_service")

app = FastAPI(
    title="AI Intent Recognition Microservice",
    description="基于 bge-m3 混合检索与 bge-reranker-v2-m3 精排的意图识别服务"
)

# --- Redis 异步连接池 ---
redis_pool = redis.ConnectionPool.from_url(_cfg.redis_url, decode_responses=True)
redis_client = redis.Redis(connection_pool=redis_pool)


# --- 并发控制：Singleflight (请求合并机制) ---
class SingleFlight:
    def __init__(self):
        self._futures = {}
        self._lock = asyncio.Lock()

    async def do(self, key: str, coro_func, *args, **kwargs):
        async with self._lock:
            if key in self._futures:
                logger.info(f"[SingleFlight] 请求合并命中，等待结果... Key: {key}")
                fut = self._futures[key]
                is_leader = False
            else:
                fut = asyncio.get_event_loop().create_future()
                self._futures[key] = fut
                is_leader = True

        if is_leader:
            try:
                result = await coro_func(*args, **kwargs)
                fut.set_result(result)
                return result
            except Exception as e:
                fut.set_exception(e)
                raise
            finally:
                async with self._lock:
                    self._futures.pop(key, None)
        else:
            return await asyncio.shield(fut)


singleflight = SingleFlight()


# --- 请求与响应模型 ---
class IntentRequest(BaseModel):
    text: str
    call_id: str
    model_id: int
    word_count: Optional[int] = Field(default=None, description="字数阈值，不传时跳过短文本拦截直接走相似度匹配")
    question_similarity: float = Field(default=0.0, description="问法相似度阈值，覆盖 low_score_threshold 做置信度拦截，0.0 表示使用全局配置")


class IntentResponse(BaseModel):
    intent_id: str
    call_id: Optional[str] = Field(default=None)


# ================= 生命周期管理 =================

@app.on_event("startup")
async def startup_event():
    logger.info("[启动] 正在初始化 Nacos 配置...")
    await NacosConfig.init_config()
    logger.info("[启动] 正在初始化 HyBridSearch 组件（模型 + Milvus + 批处理器 + 重排器）...")
    HyBridSearch.init_components()
    logger.info("[启动] 所有组件初始化完成，服务就绪")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("[关停] 正在释放资源...")
    HyBridSearch.cleanup_components()
    await redis_pool.disconnect()
    logger.info("[关停] 所有资源已释放")


# ================= 通用 API =================
# ... (保持原样的接口：/health, /ready, /insert, /delete, /update, /upload 省略修改，与原代码一致)
@app.get("/health")
async def health_check(): return {"status": "ok"}


@app.get("/ready")
async def readiness_check():
    if HyBridSearch.bge_model is None or HyBridSearch.collection is None or HyBridSearch.batcher is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"status": "ready", "collection": _cfg.collection_name}


@app.post("/insert")
async def insert_data(req: BatchInsertRequest):
    try:
        total_start = time.perf_counter()
        items = req.items
        logger.info(f"[新增] 接收数据{items}")
        texts = [item.text for item in items]
        t_encode = time.perf_counter()
        encoded = HyBridSearch.bge_model.encode(texts, return_dense=True, return_sparse=True)
        encode_cost = (time.perf_counter() - t_encode) * 1000
        model_ids = [item.model_id for item in items]
        intent_ids = [item.intent_id for item in items]
        text_list = [item.text for item in items]
        types = [item.type for item in items]
        is_actives = [item.active for item in items]
        dense_vecs = [encoded['dense_vecs'][j].tolist() for j in range(len(items))]
        sparse_vecs = [encoded['lexical_weights'][j] for j in range(len(items))]
        entities = [model_ids, intent_ids, text_list, types, is_actives, dense_vecs, sparse_vecs]
        logger.info(f"[新增] 新增数据{entities}")
        res = HyBridSearch.collection.insert(entities)
        total_cost = (time.perf_counter() - total_start) * 1000
        logger.info(f"[新增] 批量写入 {len(items)} 条, 编码={encode_cost:.1f}ms, 总耗时={total_cost:.1f}ms")
        return {
            "code": 200,
            "msg": "success",
            "data": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量写入 Milvus 失败: {str(e)}")


@app.delete("/delete")
async def delete_data(req: DeleteRequest):
    """根据 intent_id 批量逻辑删除意图语料"""
    try:
        total_start = time.perf_counter()
        deleted_count = 0

        # 分批处理避免 Milvus expr 表达式超长
        batch_size = 100
        for i in range(0, len(req.intent_ids), batch_size):
            batch_ids = req.intent_ids[i:i + batch_size]

            id_list_str = ",".join([str(int(x)) for x in batch_ids])
            # 只查询当前还是”活跃”且匹配 intent_id 的全量数据
            expr = f"intent_id in [{id_list_str}] and is_active == true and model_id == {req.model_id}"

            query_results = HyBridSearch.collection.query(
                expr=expr,
                output_fields=["id", "model_id", "intent_id", "text", "type", "dense_vector", "sparse_vector"],
                limit=16384
            )

            if not query_results:
                continue

            # 1. 依据主键(PK)物理删除
            pks = [str(r["id"]) for r in query_results]
            HyBridSearch.collection.delete(expr=f"id in [{', '.join(pks)}]")

            # 2. 数据原样组装，只将 is_active 设为 False，然后重新插入 (逻辑删除)
            ins_models = [r["model_id"] for r in query_results]
            ins_intents = [r["intent_id"] for r in query_results]
            ins_texts = [r["text"] for r in query_results]
            ins_types = [r["type"] for r in query_results]
            ins_actives = [False] * len(query_results)
            ins_dense = [r["dense_vector"] for r in query_results]
            ins_sparse = [r["sparse_vector"] for r in query_results]

            HyBridSearch.collection.insert([
                ins_models, ins_intents, ins_texts, ins_types, ins_actives, ins_dense, ins_sparse
            ])
            deleted_count += len(query_results)

        total_cost = (time.perf_counter() - total_start) * 1000
        logger.info(
            f"[删除] intent_ids={len(req.intent_ids)}个, "
            f"逻辑删除 {deleted_count} 条, 耗时={total_cost:.1f}ms"
        )

        return {
            "code": 200,
            "msg": "success",
            "data": True
        }
    except Exception as e:
        logger.error(f"[删除失败] {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"逻辑删除失败: {str(e)}")


@app.post("/update")
async def update_data(req: BatchUpdateRequest):
    """
    批量更新意图语料（先删后增）。
    Java 侧传入 List<MilvusIntentDTO>，每条包含 intentId/modelId/type/text。
    按 (intentId, modelId, type) 分组聚合后：
      1. 对每组先逻辑删除该 (intentId, modelId) 下所有活跃记录
      2. 再将聚合后的新 texts 插入
    """
    try:
        total_start = time.perf_counter()

        # 按 (intent_id, model_id, type) 分组聚合 texts
        from collections import defaultdict
        groups = defaultdict(lambda: {"texts": [], "active": True})
        for item in req.items:
            key = (item.intent_id, item.model_id, item.type)
            groups[key]["texts"].append(item.text)
            groups[key]["active"] = item.active

        results = []
        total_deleted = 0
        total_inserted = 0

        for (intent_id, model_id, item_type), group in groups.items():
            item_result = {
                "intent_id": intent_id,
                "model_id": model_id,
                "type": item_type,
                "deleted": 0,
                "inserted": 0,
                "status": "success"
            }
            try:
                # === Step 1: 逻辑删除该 intent_id+model_id 下的原有活跃记录 ===
                expr = f"intent_id == {intent_id} and is_active == true and model_id == {model_id}"
                qs = HyBridSearch.collection.query(
                    expr=expr,
                    output_fields=["id", "model_id", "intent_id", "text", "type", "dense_vector", "sparse_vector"],
                    limit=16384
                )
                if qs:
                    pks = [str(r["id"]) for r in qs]
                    HyBridSearch.collection.delete(expr=f"id in [{', '.join(pks)}]")

                    ins_models = [r["model_id"] for r in qs]
                    ins_intents = [r["intent_id"] for r in qs]
                    ins_texts = [r["text"] for r in qs]
                    ins_types = [r["type"] for r in qs]
                    ins_actives = [False] * len(qs)
                    ins_dense = [r["dense_vector"] for r in qs]
                    ins_sparse = [r["sparse_vector"] for r in qs]

                    HyBridSearch.collection.insert([
                        ins_models, ins_intents, ins_texts, ins_types, ins_actives, ins_dense, ins_sparse
                    ])

                    item_result["deleted"] = len(qs)
                    total_deleted += len(qs)

                # === Step 2: 编码新的 texts 并插入 ===
                texts_to_insert = group["texts"]
                t_encode = time.perf_counter()
                encoded = HyBridSearch.bge_model.encode(
                    texts_to_insert, return_dense=True, return_sparse=True
                )
                encode_cost = (time.perf_counter() - t_encode) * 1000

                n = len(texts_to_insert)
                entities = [
                    [model_id] * n,
                    [intent_id] * n,
                    texts_to_insert,
                    [item_type] * n,
                    [group["active"]] * n,
                    [encoded['dense_vecs'][j].tolist() for j in range(n)],
                    [encoded['lexical_weights'][j] for j in range(n)],
                ]
                res = HyBridSearch.collection.insert(entities)

                item_result["inserted"] = len(res.primary_keys)
                total_inserted += len(res.primary_keys)

                logger.info(
                    f"[更新] intent_id={intent_id} model_id={model_id} type={item_type}: "
                    f"逻辑删除旧记录={item_result['deleted']}, "
                    f"新增={item_result['inserted']}, 编码耗时={encode_cost:.1f}ms"
                )
            except Exception as e:
                item_result["status"] = f"failed: {str(e)}"
                logger.error(f"[更新] intent_id={intent_id} 失败: {e}", exc_info=True)

            results.append(item_result)

        total_cost = (time.perf_counter() - total_start) * 1000
        logger.info(
            f"[更新] 批量完成: 处理 {len(groups)} 个意图组, "
            f"逻辑删除原记录总计={total_deleted}, 新增记录总计={total_inserted}, "
            f"总耗时={total_cost:.1f}ms"
        )

        return {
            "code": 200,
            "msg": "success",
            "data": True
        }
    except Exception as e:
        logger.error(f"[更新失败] {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"批量逻辑更新失败: {str(e)}")


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """批量上传 CSV 文件导入意图语料（格式：intent_id,text）"""
    try:
        total_start = time.perf_counter()

        # 1. 读取并解析 CSV
        content = await file.read()
        for encoding in ['utf-8-sig', 'utf-8', 'gbk', 'gb2312']:
            try:
                text_content = content.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise HTTPException(status_code=400, detail="无法解析文件编码，请使用 UTF-8 或 GBK 编码")

        reader = csv.DictReader(io.StringIO(text_content))

        # 校验表头
        required_fields = {'intent_id', 'text', 'model_id'}
        if not required_fields.issubset(set(reader.fieldnames or [])):
            raise HTTPException(
                status_code=400,
                detail=f"CSV 表头缺少必要字段，需要: {required_fields}，实际: {reader.fieldnames}"
            )

        # 收集所有行
        rows = []
        for row in reader:
            text = (row.get('text') or '').strip()
            if text:
                # model_id 必填，这里已在表头校验，且如果数据行中没有，强行转会报错或取默认值
                # 为了健壮性，如果为空可以抛错，或者转为int
                model_id_val = row.get('model_id')
                if not model_id_val:
                    raise HTTPException(status_code=400, detail="model_id 不能为空")
                rows.append({
                    'model_id': int(model_id_val),
                    'intent_id': int(row.get('intent_id')),
                    'text': text,
                    'type': int(row.get('type') or 1)  # 缺省默认为 1（具体问法）
                })

        if not rows:
            raise HTTPException(status_code=400, detail="CSV 中没有有效数据行")

        logger.info(f"[上传] 文件: {file.filename}, 有效行数: {len(rows)}")

        # 2. 分批编码并写入
        BATCH_SIZE = 32
        success_count = 0
        fail_count = 0
        fail_details = []

        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i:i + BATCH_SIZE]
            texts = [r['text'] for r in batch]

            try:
                t_encode = time.perf_counter()
                encoded = HyBridSearch.bge_model.encode(texts, return_dense=True, return_sparse=True)
                encode_cost = (time.perf_counter() - t_encode) * 1000

                model_ids = [r['model_id'] for r in batch]
                intent_ids = [r['intent_id'] for r in batch]
                text_list = [r['text'] for r in batch]
                types = [r['type'] for r in batch]
                is_actives = [True] * len(batch)
                dense_vecs = [encoded['dense_vecs'][j].tolist() for j in range(len(batch))]
                sparse_vecs = [encoded['lexical_weights'][j] for j in range(len(batch))]

                entities = [model_ids, intent_ids, text_list, types, is_actives, dense_vecs, sparse_vecs]
                HyBridSearch.collection.insert(entities)

                success_count += len(batch)
                logger.info(
                    f"[上传] 批次 {i // BATCH_SIZE + 1}: "
                    f"写入 {len(batch)} 条, 编码耗时={encode_cost:.1f}ms"
                )
            except Exception as e:
                fail_count += len(batch)
                fail_details.append(f"批次{i // BATCH_SIZE + 1}失败: {str(e)}")
                logger.error(f"[上传] 批次 {i // BATCH_SIZE + 1} 写入失败: {e}")

        total_cost = (time.perf_counter() - total_start) * 1000
        logger.info(
            f"[上传] 完成! 成功={success_count}, 失败={fail_count}, "
            f"总耗时={total_cost:.1f}ms"
        )

        return {
            "code": 200,
            "msg": "上传完成",
            "filename": file.filename,
            "total": len(rows),
            "success": success_count,
            "fail": fail_count,
            "fail_details": fail_details if fail_details else None,
            "time_cost_ms": round(total_cost, 2)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV 导入失败: {str(e)}")


@app.get("/list")
async def list_data(
        model_id: int,
        intent_id: Optional[str] = None,
        type: Optional[int] = None,
        is_active: Optional[bool] = True,
        page: int = 1,
        page_size: int = 20
):
    """查询向量库中的意图语料，支持按 model_id / intent_id / is_active 过滤，分页返回"""
    try:
        if page < 1:
            raise HTTPException(status_code=400, detail="page 最小为 1")
        if not (1 <= page_size <= 200):
            raise HTTPException(status_code=400, detail="page_size 范围 1~200")

        expr_parts = [f"model_id == {model_id}"]
        if intent_id is not None:
            expr_parts.append(f'intent_id == "{intent_id}"')
        if type is not None:
            expr_parts.append(f"type == {type}")
        if is_active is not None:
            expr_parts.append(f"is_active == {'true' if is_active else 'false'}")
        expr = " and ".join(expr_parts)

        offset = (page - 1) * page_size

        import asyncio as _asyncio

        def _query_page():
            return HyBridSearch.collection.query(
                expr=expr,
                output_fields=["id", "model_id", "intent_id", "text", "type"],
                limit=page_size,
                offset=offset
            )

        def _query_total():
            res = HyBridSearch.collection.query(
                expr=expr,
                output_fields=["count(*)"],
            )
            return res[0]["count(*)"] if res else 0

        loop = _asyncio.get_event_loop()
        results, total = await _asyncio.gather(
            loop.run_in_executor(None, _query_page),
            loop.run_in_executor(None, _query_total),
        )

        items = [
            {
                "id": r["id"],
                "model_id": r["model_id"],
                "intent_id": r["intent_id"],
                "text": r["text"],
                "type": r["type"]
            }
            for r in results
        ]

        import math
        total_pages = math.ceil(total / page_size) if total > 0 else 1

        logger.info(
            f"[列表] model_id={model_id} intent_id={intent_id} is_active={is_active} page={page} 返回 {len(items)}/{total} 条")
        return {
            "code": 200,
            "msg": "success",
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": total_pages,
            "count": len(items),
            "items": items
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[列表查询失败] {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"列表查询失败: {str(e)}")


@app.post("/compare")
async def compare_data(req: CompareRequest):
    """混合检索 + 重排精排 核心逻辑（异步 + 批处理 + 分段耗时监控）"""
    try:
        total_start = time.perf_counter()

        # -------- Phase 1: 模型编码 --------
        t1 = time.perf_counter()
        encoded = await HyBridSearch.batcher.encode(req.text)
        query_dense = [encoded['dense_vec']]
        query_sparse = [encoded['sparse_vec']]
        t1_cost = (time.perf_counter() - t1) * 1000

        # -------- Phase 2: 构建粗排请求 --------
        # 【修改点】放大召回候选集，给精排提供更多弹药
        recall_limit = max(20, req.top_k * 2)

        t2 = time.perf_counter()
        req_dense = AnnSearchRequest(
            data=query_dense, anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=recall_limit, expr=f"is_active == true and model_id == {req.model_id}"
        )
        req_sparse = AnnSearchRequest(
            data=query_sparse, anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
            limit=recall_limit, expr=f"is_active == true and model_id == {req.model_id}"
        )
        t2_cost = (time.perf_counter() - t2) * 1000

        # -------- Phase 3: Milvus 混合检索 (粗排) --------
        t3 = time.perf_counter()
        results = HyBridSearch.collection.hybrid_search(
            reqs=[req_dense, req_sparse],
            rerank=RRFRanker(),
            limit=recall_limit,
            output_fields=["text", "intent_id"]
        )
        t3_cost = (time.perf_counter() - t3) * 1000

        # 提取候选集
        candidates = []
        if results and results[0]:
            for hit in results[0]:
                candidates.append({
                    "intent_id": hit.entity.get("intent_id"),
                    "text": hit.entity.get("text")
                })

        # -------- Phase 4: Cross-Encoder 精排 --------
        t4 = time.perf_counter()
        # 【新增】调用异步重排方法
        reranked_results = await HyBridSearch.async_rerank_candidates(req.text, candidates)
        t4_cost = (time.perf_counter() - t4) * 1000

        # -------- Phase 5: 截取 Top-K 与 置信度计算 --------
        t5 = time.perf_counter()
        final_matches = reranked_results[:req.top_k]

        confidence_status = "UNKNOWN"
        debug_gap = None

        if len(final_matches) > 0:
            top1_conf = final_matches[0].get("raw_confidence", 0.0)
            top1_prob = final_matches[0]["probability"]

            # 低置信判断使用绝对置信度，避免 softmax 相对概率误伤
            if top1_conf < _cfg.low_score_threshold:
                confidence_status = "LOW_CONFIDENCE"
            elif len(final_matches) > 1:
                top2_prob = final_matches[1]["probability"]
                # 断层判断继续使用相对概率差值
                debug_gap = round(top1_prob - top2_prob, 4)

                if debug_gap >= _cfg.high_gap_threshold:
                    confidence_status = "HIGH_CONFIDENCE"
                else:
                    confidence_status = "NEEDS_CLARIFICATION"
            else:
                confidence_status = "HIGH_CONFIDENCE"
        t5_cost = (time.perf_counter() - t5) * 1000

        total_cost = (time.perf_counter() - total_start) * 1000

        # 日志输出
        logger.info(
            f"[查询] text='{req.text}' | 编码={t1_cost:.1f}ms 粗排={t3_cost:.1f}ms 精排={t4_cost:.1f}ms 总计={total_cost:.1f}ms | "
            f"{confidence_status}"
        )
        if len(final_matches) >= 2:
            logger.debug(
                f"[评分] Top1={final_matches[0]['probability']:.4f}({final_matches[0]['intent_id']}) "
                f"Top2={final_matches[1]['probability']:.4f}({final_matches[1]['intent_id']}) "
                f"Gap={debug_gap}"
            )

        return {
            "code": 200,
            "msg": "success",
            "query_text": req.text,
            "confidence_status": confidence_status,
            "gap_score": debug_gap,
            "matches": final_matches,  # 此时返回的列表里包含了 probability 分数
            "time_cost_ms": {
                "total": round(total_cost, 2),
                "model_encode": round(t1_cost, 2),
                "build_request": round(t2_cost, 2),
                "milvus_search": round(t3_cost, 2),
                "reranker": round(t4_cost, 2),
                "result_process": round(t5_cost, 2)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"混合检索+重排失败: {str(e)}")

class InterruptRequest(BaseModel):
    call_id: int


class CallbackRequest(BaseModel):
    call_id: int
    intent_id: Optional[str] = None
    uuid: Optional[int] = None
    event: Optional[str] = None
    timestamp: Optional[int] = None
    transcript: Optional[str] = None


@app.post("/interrupt")
async def interrupt(req: InterruptRequest):
    logger.info(f"这是打断请求 : {req}")


@app.post("/callback")
async def callback(req: CallbackRequest):
    logger.info(f"回调事件: {req.event} 回调请求 : {req}")



# ================= 意图识别（生产接口）=================

RECOGNIZE_TOP_K = 2


async def _vector_search(text: str, model_id: int, query_type: int) -> list:
    """Milvus 混合检索 + Reranker 精排，返回排序后的候选列表"""
    encoded = await HyBridSearch.batcher.encode(text)
    query_dense = [encoded['dense_vec']]
    query_sparse = [encoded['sparse_vec']]

    recall_limit = max(20, RECOGNIZE_TOP_K * 2)
    search_expr = f"is_active == true and model_id == {model_id} and type == {query_type}"
    req_dense = AnnSearchRequest(
        data=query_dense, anns_field="dense_vector",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=recall_limit, expr=search_expr
    )
    req_sparse = AnnSearchRequest(
        data=query_sparse, anns_field="sparse_vector",
        param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
        limit=recall_limit, expr=search_expr
    )

    results = HyBridSearch.collection.hybrid_search(
        reqs=[req_dense, req_sparse], rerank=RRFRanker(),
        limit=recall_limit, output_fields=["text", "intent_id"]
    )

    candidates = []
    if results and results[0]:
        for hit in results[0]:
            candidates.append({
                "intent_id": hit.entity.get("intent_id"),
                "text": hit.entity.get("text")
            })

    if not candidates:
        return []

    return await HyBridSearch.async_rerank_candidates(text, candidates)


async def fetch_intent_from_vector_db(
    text: str, cache_key: str, model_id: int,
    word_count: Optional[int], question_similarity: float
) -> str:
    """
    意图识别主流程：
    - word_count 为 None：跳过关键字匹配和短文本拦截，直接走向量相似度匹配
    - 短文本（len <= word_count）：仅 Redis 关键字匹配，未命中直接返回 intent_unknown
    - 长文本（len > word_count） ：先 Redis 关键字匹配，未命中再走向量相似度匹配
    置信度阈值优先使用请求参数 question_similarity，为 0.0 时回退到全局 low_score_threshold
    """
    total_start = time.perf_counter()
    score_threshold = question_similarity if question_similarity > 0.0 else _cfg.low_score_threshold

    # ---- word_count 有值时走关键字匹配分支 ----
    if word_count is not None:
        is_short = len(text) < word_count

        # Phase 1: Redis 关键字精确匹配（短文本和长文本都走）
        # key 格式：dolphin:intent:keyword:{model_id}:{text}
        keyword_key = f"dolphin:intent:keyword:{model_id}:{text}"
        keyword_intent = await redis_client.get(keyword_key)
        if keyword_intent:
            logger.info(f"[Keyword] 关键字命中: text='{text}' → intent_id={keyword_intent}")
            return str(keyword_intent)

        # 短文本：关键字未命中直接返回未知
        if is_short:
            logger.info(f"[Keyword] 短文本未命中关键字，直接返回 intent_unknown: text='{text}'")
            return "intent_unknown"

    # ---- 向量相似度匹配（长文本 或 word_count 未传）----
    t_vec = time.perf_counter()
    reranked_results = await _vector_search(text, model_id, query_type=1)
    vec_cost = (time.perf_counter() - t_vec) * 1000

    if not reranked_results:
        logger.warning(f"[VectorDB] 粗排未找到任何候选: text='{text}'")
        return "intent_unknown"

    final_matches = reranked_results[:4]

    debug_gap = None
    top_hit = reranked_results[0]
    top_conf = top_hit.get("raw_confidence", 0.0)
    intent_id = top_hit["intent_id"]

    if len(final_matches) > 0:
        top1_prob = final_matches[0]["probability"]

        # 低置信判断使用绝对置信度，避免 softmax 相对概率误伤
        if top_conf < score_threshold:
            logger.warning(
                f"[VectorDB] 置信度拦截(低于{score_threshold}): text='{text}' | "
                f"Top1={intent_id}, conf={top_conf:.4f}"
            )
            return "intent_unknown"
        elif len(final_matches) > 1:
            top2_prob = final_matches[1]["probability"]
            # 断层判断继续使用相对概率差值
            debug_gap = round(top1_prob - top2_prob, 4)

            if debug_gap >= _cfg.high_gap_threshold:
                return str(intent_id)
        else:
            return str(intent_id)

    total_cost = (time.perf_counter() - total_start) * 1000
    logger.info(
        f"[VectorDB] 识别完成: text='{text}' | word_count={word_count} | "
        f"命中意图: {intent_id}, confidence={top_conf:.4f}, 阈值={score_threshold} | "
        f"向量耗时={vec_cost:.1f}ms, 总耗时={total_cost:.1f}ms"
    )

    return str(intent_id)


# --- 异步缓存写入 ---
async def set_cache_background(key: str, intent_id: str):
    await redis_client.set(key, intent_id, ex=86400)


# --- 意图识别接口 ---
@app.post("/api/v1/recognize", response_model=IntentResponse)
async def recognize_intent(request: IntentRequest):
    # (此部分保持不变，请求清洗、SingleFlight 和 Redis 逻辑非常完善)
    clean_text = re.sub(r'<[^>]+>', '', request.text)
    clean_text = re.sub(r'[^\w\s\u4e00-\u9fa5,.?!，。？！]', '', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    logger.info(f"request: {request}")

    if not clean_text:
        raise HTTPException(status_code=400, detail="Text is empty after preprocessing.")

    if len(clean_text) > MAX_TEXT_LENGTH:
        raise HTTPException(status_code=400, detail="Text exceeds maximum length limit.")
    current_node_key = f"dolphin:current:node:{request.call_id}"  # 查询当前call_id对应的节点
    current_node_id = await redis_client.get(current_node_key)
    if not current_node_id:
        raise HTTPException(status_code=400, detail="current  callId have not been set node id")
    logger.info(f"当前节点id:{current_node_id}  查询key:{current_node_key}")

    # 根据当前节点id查询热键  命中直接返回 未命中走向量匹配
    hot_key_selector = f"dolphin:intent:match:{current_node_id}:{clean_text}"
    intent_id = await redis_client.get(hot_key_selector)
    logger.info(f"是否匹配到热数据 -> {intent_id} 当前intent_id:{intent_id}")

    if not intent_id:
        try:
            intent_id = await singleflight.do(
                hot_key_selector,
                fetch_intent_from_vector_db,
                clean_text,
                hot_key_selector,
                request.model_id,
                request.word_count,
                request.question_similarity
            )
        except Exception as e:
            logger.error(f"[Error] 向量检索失败: {e}")
            intent_id = "intent_unknown"
    logger.info(f"callId : {request.call_id} intentId : {intent_id}")
    return IntentResponse(
        intent_id=intent_id,
        call_id=request.call_id
    )
