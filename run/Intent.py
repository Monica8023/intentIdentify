"""
Intent — AI 意图识别微服务（唯一服务入口）
===========================================
整合了 HyBridSearch 混合检索与 BGE Reranker 精排能力，对外暴露全部 API：
  - GET  /health           健康检查
  - GET  /ready            就绪检查
  - POST /insert           批量写入意图语料
  - POST /delete           根据 intent_id 批量删除
  - POST /update           批量更新（先删后增）
  - POST /upload           CSV 批量导入语料
  - POST /compare          混合检索+精排（调试/测试用）
  - POST /api/v1/recognize 意图识别（生产接口，含缓存+防击穿）

启动：
  uvicorn Intent:app --host 0.0.0.0 --port 8808
"""

import asyncio
import csv
import hashlib
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
    call_id: Optional[str] = Field(default=None, alias="callId")

    @validator('call_id')
    def call_id_must_not_be_empty(cls, v):
        if not v:
            raise ValueError('call_id不能为空')


class IntentResponse(BaseModel):
    intent_id: str
    call_id: Optional[str] = Field(default=None, alias="callId")


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
        is_actives = [item.active for item in items]
        dense_vecs = [encoded['dense_vecs'][j].tolist() for j in range(len(items))]
        sparse_vecs = [encoded['lexical_weights'][j] for j in range(len(items))]
        entities = [model_ids, intent_ids, text_list, is_actives, dense_vecs, sparse_vecs]
        logger.info(f"[新增] 新增数据{entities}")
        res = HyBridSearch.collection.insert(entities)
        total_cost = (time.perf_counter() - total_start) * 1000
        logger.info(f"[新增] 批量写入 {len(items)} 条, 编码={encode_cost:.1f}ms, 总耗时={total_cost:.1f}ms")
        return {
            "code": 200, "msg": "success", "inserted_count": len(res.primary_keys),
            "inserted_ids": list(res.primary_keys), "time_cost_ms": round(total_cost, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量写入 Milvus 失败: {str(e)}")


@app.post("/delete")
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
                output_fields=["id", "model_id", "intent_id", "text", "dense_vector", "sparse_vector"],
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
            ins_actives = [False] * len(query_results)
            ins_dense = [r["dense_vector"] for r in query_results]
            ins_sparse = [r["sparse_vector"] for r in query_results]

            HyBridSearch.collection.insert([
                ins_models, ins_intents, ins_texts, ins_actives, ins_dense, ins_sparse
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
            "deleted_count": deleted_count,
            "intent_ids": req.intent_ids,
            "time_cost_ms": round(total_cost, 2)
        }
    except Exception as e:
        logger.error(f"[删除失败] {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"逻辑删除失败: {str(e)}")


@app.post("/update")
async def update_data(req: BatchUpdateRequest):
    """
    批量更新意图语料（先删后增）。
    对指定的 intent_id 进行逻辑删除：先通过 ID 查出 is_active=True 的全量原数据，将其
    以 is_active=False 的状态重新入库，并物理删除原有 ID。然后再插入更新的新 texts。
    """
    try:
        total_start = time.perf_counter()

        results = []
        total_deleted = 0
        total_inserted = 0

        for item in req.items:
            item_result = {
                "intent_id": item.intent_id,
                "deleted": 0,  # 这里的 deleted 是指逻辑删除的数量
                "inserted": 0,
                "status": "success"
            }
            try:
                # === Step 1: 逻辑删除该 intent_id 下的原有活跃记录 ===
                expr = f'intent_id == "{item.intent_id}" and is_active == true and model_id == {item.model_id}'
                qs = HyBridSearch.collection.query(
                    expr=expr,
                    output_fields=["id", "model_id", "intent_id", "text", "dense_vector", "sparse_vector"],
                    limit=16384
                )
                if qs:
                    # 1.1 依据主键(PK)物理删除存量活跃记录
                    pks = [str(r["id"]) for r in qs]
                    HyBridSearch.collection.delete(expr=f"id in [{', '.join(pks)}]")

                    # 1.2 以 is_active=False 重新插入实现逻辑删除
                    ins_models = [r["model_id"] for r in qs]
                    ins_intents = [r["intent_id"] for r in qs]
                    ins_texts = [r["text"] for r in qs]
                    ins_actives = [False] * len(qs)
                    ins_dense = [r["dense_vector"] for r in qs]
                    ins_sparse = [r["sparse_vector"] for r in qs]

                    HyBridSearch.collection.insert([
                        ins_models, ins_intents, ins_texts, ins_actives, ins_dense, ins_sparse
                    ])

                    item_result["deleted"] = len(qs)
                    total_deleted += len(qs)

                # === Step 2: 编码新的 texts 并插入新配置 ===
                t_encode = time.perf_counter()
                encoded = HyBridSearch.bge_model.encode(
                    item.texts, return_dense=True, return_sparse=True
                )
                encode_cost = (time.perf_counter() - t_encode) * 1000

                n = len(item.texts)
                model_ids = [item.model_id] * n
                intent_ids = [item.intent_id] * n
                text_list = list(item.texts)
                is_actives = [item.active] * n
                dense_vecs = [encoded['dense_vecs'][j].tolist() for j in range(n)]
                sparse_vecs = [encoded['lexical_weights'][j] for j in range(n)]

                entities = [model_ids, intent_ids, text_list, is_actives, dense_vecs, sparse_vecs]
                res = HyBridSearch.collection.insert(entities)

                item_result["inserted"] = len(res.primary_keys)
                total_inserted += len(res.primary_keys)

                logger.info(
                    f"[更新] intent_id={item.intent_id}: "
                    f"逻辑删除的旧记录数 {item_result['deleted']}, "
                    f"新增记录数 {item_result['inserted']}, "
                    f"新记录编码耗时 {encode_cost:.1f}ms"
                )
            except Exception as e:
                item_result["status"] = f"failed: {str(e)}"
                logger.error(f"[更新] intent_id={item.intent_id} 失败: {e}", exc_info=True)

            results.append(item_result)

        total_cost = (time.perf_counter() - total_start) * 1000
        logger.info(
            f"[更新] 批量完成: 处理 {len(req.items)} 个意图, "
            f"逻辑删除原记录总计={total_deleted}, 新增记录总计={total_inserted}, "
            f"总耗时={total_cost:.1f}ms"
        )

        return {
            "code": 200,
            "msg": "success",
            "total_deleted": total_deleted,
            "total_inserted": total_inserted,
            "details": results,
            "time_cost_ms": round(total_cost, 2)
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
        required_fields = {'intent_id', 'text'}
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
                rows.append({
                    'model_id': int(row.get('model_id') or 1),
                    'intent_id': (row.get('intent_id') or '').strip(),
                    'text': text
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
                is_actives = [True] * len(batch)
                dense_vecs = [encoded['dense_vecs'][j].tolist() for j in range(len(batch))]
                sparse_vecs = [encoded['lexical_weights'][j] for j in range(len(batch))]

                entities = [model_ids, intent_ids, text_list, is_actives, dense_vecs, sparse_vecs]
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
        model_id: int = 1,
        intent_id: Optional[str] = None,
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
        if is_active is not None:
            expr_parts.append(f"is_active == {'true' if is_active else 'false'}")
        expr = " and ".join(expr_parts)

        offset = (page - 1) * page_size

        import asyncio as _asyncio

        def _query_page():
            return HyBridSearch.collection.query(
                expr=expr,
                output_fields=["id", "model_id", "intent_id", "text"],
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
                "text": r["text"]
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
            top1_prob = final_matches[0]["probability"]

            if top1_prob < _cfg.low_score_threshold:
                confidence_status = "LOW_CONFIDENCE"
            elif len(final_matches) > 1:
                top2_prob = final_matches[1]["probability"]
                # 计算概率断层差值
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


# ================= 意图识别（生产接口）=================

RECOGNIZE_TOP_K = 4


async def fetch_intent_from_vector_db(text: str, cache_key: str) -> str:
    """两阶段检索：Milvus 召回 + Reranker 精排"""
    total_start = time.perf_counter()

    # ---- Phase 1: 编码文本 ----
    t1 = time.perf_counter()
    encoded = await HyBridSearch.batcher.encode(text)
    query_dense = [encoded['dense_vec']]
    query_sparse = [encoded['sparse_vec']]
    t1_cost = (time.perf_counter() - t1) * 1000

    # ---- Phase 2: 构建混合检索请求 (放大召回) ----
    recall_limit = max(20, RECOGNIZE_TOP_K * 2)
    req_dense = AnnSearchRequest(
        data=query_dense, anns_field="dense_vector",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=recall_limit, expr="is_active == true"
    )
    req_sparse = AnnSearchRequest(
        data=query_sparse, anns_field="sparse_vector",
        param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
        limit=recall_limit, expr="is_active == true"
    )

    # ---- Phase 3: Milvus 粗排 ----
    t3 = time.perf_counter()
    results = HyBridSearch.collection.hybrid_search(
        reqs=[req_dense, req_sparse], rerank=RRFRanker(),
        limit=recall_limit, output_fields=["text", "intent_id"]
    )
    t3_cost = (time.perf_counter() - t3) * 1000

    candidates = []
    if results and results[0]:
        for hit in results[0]:
            candidates.append({
                "intent_id": hit.entity.get("intent_id"),
                "text": hit.entity.get("text")
            })

    if not candidates:
        logger.warning(f"[VectorDB] 粗排未找到任何候选: text='{text}'")
        return "intent_unknown"

    # ---- Phase 4: Reranker 精排 ----
    t4 = time.perf_counter()
    reranked_results = await HyBridSearch.async_rerank_candidates(text, candidates)
    t4_cost = (time.perf_counter() - t4) * 1000

    # ---- Phase 5: 置信度拦截与提取 ----
    top_hit = reranked_results[0]
    top_prob = top_hit["probability"]
    intent_id = top_hit["intent_id"]

    if top_prob < _cfg.low_score_threshold:
        logger.warning(
            f"[VectorDB] 置信度拦截(低于{_cfg.low_score_threshold}): text='{text}' | "
            f"Top1={intent_id}, Prob={top_prob}"
        )
        return "intent_unknown"

    if len(reranked_results) > 1:
        gap = top_prob - reranked_results[1]["probability"]
        if gap < 0.1:
            logger.warning(
                f"[VectorDB] 意图模糊(Gap={gap:.4f}): text='{text}' | Top1={intent_id}, Top2={reranked_results[1]['intent_id']}")

    total_cost = (time.perf_counter() - total_start) * 1000
    logger.info(
        f"[VectorDB] 识别完成: text='{text}' | "
        f"命中意图: {intent_id}, Prob={top_prob:.4f} | "
        f"编码={t1_cost:.1f}ms, 粗排={t3_cost:.1f}ms, 精排={t4_cost:.1f}ms, 总耗时={total_cost:.1f}ms"
    )

    # 写入缓存（后台，不阻塞返回）
    asyncio.ensure_future(set_cache_background(cache_key, intent_id))

    return intent_id


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

    if not clean_text:
        raise HTTPException(status_code=400, detail="Text is empty after preprocessing.")

    if len(clean_text) > MAX_TEXT_LENGTH:
        raise HTTPException(status_code=400, detail="Text exceeds maximum length limit.")
    current_node_key = f"dolphin:current:node:{request.call_id}"  # 查询当前call_id对应的节点
    current_node_id = await redis_client.get(current_node_key)
    logger.info(f"当前节点id:{current_node_id}  查询key:{current_node_key}")

    # 根据当前节点id查询热键  命中直接返回 未命中走向量匹配
    hot_key_selector = f"dolphin:intent:match:{current_node_id}:{clean_text}"
    intent_id = await redis_client.get(hot_key_selector)
    logger.info(f"是否匹配到热数据 -> {not intent_id} 当前intent_id:{intent_id}")

    if not intent_id:
        try:
            intent_id = await singleflight.do(
                hot_key_selector,
                fetch_intent_from_vector_db,
                clean_text,
                hot_key_selector
            )
        except Exception as e:
            logger.error(f"[Error] 向量检索失败: {e}")
            intent_id = "intent_unknown"

    return IntentResponse(
        intent_id=intent_id,
        call_id=request.call_id
    )
