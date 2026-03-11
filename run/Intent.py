"""
Intent — AI 意图识别微服务（唯一服务入口）
===========================================
整合了 HyBridSearch 混合检索能力，对外暴露全部 API：
  - GET  /health           健康检查
  - GET  /ready            就绪检查
  - POST /insert           批量写入意图语料
  - POST /delete           根据 intent_id 批量删除
  - POST /update           批量更新（先删后增）
  - POST /upload           CSV 批量导入语料
  - POST /compare          混合检索（调试/测试用）
  - POST /api/v1/recognize 意图识别（生产接口，含缓存+防击穿）

启动：
  uvicorn Intent:app --host 0.0.0.0 --port 8000
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
from pydantic import BaseModel
import redis.asyncio as redis
from pymilvus import AnnSearchRequest, RRFRanker

# 确保同级目录可被 import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import HyBridSearch
from HyBridSearch import BatchInsertRequest, DeleteRequest, BatchUpdateRequest, CompareRequest

# --- 基础配置 ---
MAX_TEXT_LENGTH = 500  # 防止恶意长文本
REDIS_URL = "redis://localhost:6379/0"

# --- 日志与应用初始化 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("intent_service")

app = FastAPI(
    title="AI Intent Recognition Microservice",
    description="基于 bge-m3 与 Milvus 的混合检索意图识别服务"
)

# --- Redis 异步连接池 ---
redis_pool = redis.ConnectionPool.from_url(REDIS_URL, decode_responses=True)
redis_client = redis.Redis(connection_pool=redis_pool)


# --- 并发控制：Singleflight (请求合并机制) ---
class SingleFlight:
    """
    用于防止缓存击穿。
    当缓存未命中时，如果瞬间有1000个对同一个未知文本的并发请求，
    只有第一个请求会执行查询大模型的操作，其余999个请求会等待第一个请求的结果，从而保护大模型和向量数据库。
    """

    def __init__(self):
        self._futures = {}
        self._lock = asyncio.Lock()

    async def do(self, key: str, coro_func, *args, **kwargs):
        async with self._lock:
            if key in self._futures:
                logger.info(f"[SingleFlight] 请求合并命中，等待结果... Key: {key}")
                return await self._futures[key]

            future = asyncio.Future()
            self._futures[key] = future

        try:
            result = await coro_func(*args, **kwargs)
            future.set_result(result)
            return result
        except Exception as e:
            future.set_exception(e)
            raise e
        finally:
            async with self._lock:
                self._futures.pop(key, None)


singleflight = SingleFlight()


# --- 请求与响应模型 ---
class IntentRequest(BaseModel):
    user_id: str
    text: str


class IntentResponse(BaseModel):
    intent_id: str
    voice_url: str


# ================= 生命周期管理 =================

@app.on_event("startup")
def startup_event():
    """
    启动时初始化所有组件：
    1. 加载 bge-m3 模型
    2. 连接 Milvus 并加载 Collection
    3. 模型预热
    4. 初始化动态批处理器 (DynamicBatcher)
    """
    logger.info("[启动] 正在初始化 HyBridSearch 组件（模型 + Milvus + 批处理器）...")
    HyBridSearch.init_components()
    logger.info("[启动] 所有组件初始化完成，服务就绪")


@app.on_event("shutdown")
async def shutdown_event():
    """关停时释放所有资源"""
    logger.info("[关停] 正在释放资源...")
    HyBridSearch.cleanup_components()
    await redis_pool.disconnect()
    logger.info("[关停] 所有资源已释放")


# ================= 通用 API（从 HyBridSearch 迁入）=================

@app.get("/health")
async def health_check():
    """健康检查（负载均衡器探活）"""
    return {"status": "ok"}


@app.get("/ready")
async def readiness_check():
    """就绪检查（K8s readiness probe）"""
    if HyBridSearch.bge_model is None or HyBridSearch.collection is None or HyBridSearch.batcher is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"status": "ready", "collection": HyBridSearch.COLLECTION_NAME}


@app.post("/insert")
async def insert_data(req: BatchInsertRequest):
    """批量写入意图语料（含双向量编码）"""
    try:
        total_start = time.perf_counter()
        items = req.items
        texts = [item.text for item in items]

        # 批量编码（直接调用 bge_model，提高吹量）
        t_encode = time.perf_counter()
        encoded = HyBridSearch.bge_model.encode(texts, return_dense=True, return_sparse=True)
        encode_cost = (time.perf_counter() - t_encode) * 1000

        # 组装批量数据
        intent_ids = [item.intent_id for item in items]
        text_list = [item.text for item in items]
        is_actives = [item.is_active for item in items]
        dense_vecs = [encoded['dense_vecs'][j].tolist() for j in range(len(items))]
        sparse_vecs = [encoded['lexical_weights'][j] for j in range(len(items))]

        entities = [intent_ids, text_list, is_actives, dense_vecs, sparse_vecs]
        res = HyBridSearch.collection.insert(entities)

        total_cost = (time.perf_counter() - total_start) * 1000
        logger.info(
            f"[新增] 批量写入 {len(items)} 条, "
            f"编码={encode_cost:.1f}ms, 总耗时={total_cost:.1f}ms"
        )

        return {
            "code": 200,
            "msg": "success",
            "inserted_count": len(res.primary_keys),
            "inserted_ids": res.primary_keys,
            "time_cost_ms": round(total_cost, 2)
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
            batch_ids = req.intent_ids[i:i+batch_size]
            
            id_list_str = ", ".join([f'"{x}"' for x in batch_ids])
            # 只查询当前还是“活跃”且匹配 intent_id 的全量数据
            expr = f"intent_id in [{id_list_str}] and is_active == true"

            query_results = HyBridSearch.collection.query(
                expr=expr,
                output_fields=["id", "intent_id", "text", "dense_vector", "sparse_vector"],
                limit=16384
            )

            if not query_results:
                continue

            # 1. 依据主键(PK)物理删除
            pks = [str(r["id"]) for r in query_results]
            HyBridSearch.collection.delete(expr=f"id in [{', '.join(pks)}]")

            # 2. 数据原样组装，只将 is_active 设为 False，然后重新插入 (逻辑删除)
            ins_intents = [r["intent_id"] for r in query_results]
            ins_texts = [r["text"] for r in query_results]
            ins_actives = [False] * len(query_results)
            ins_dense = [r["dense_vector"] for r in query_results]
            ins_sparse = [r["sparse_vector"] for r in query_results]

            HyBridSearch.collection.insert([
                ins_intents, ins_texts, ins_actives, ins_dense, ins_sparse
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
                expr = f'intent_id == "{item.intent_id}" and is_active == true'
                qs = HyBridSearch.collection.query(
                    expr=expr,
                    output_fields=["id", "intent_id", "text", "dense_vector", "sparse_vector"],
                    limit=16384
                )
                if qs:
                    # 1.1 依据主键(PK)物理删除存量活跃记录
                    pks = [str(r["id"]) for r in qs]
                    HyBridSearch.collection.delete(expr=f"id in [{', '.join(pks)}]")
                    
                    # 1.2 以 is_active=False 重新插入实现逻辑删除
                    ins_intents = [r["intent_id"] for r in qs]
                    ins_texts = [r["text"] for r in qs]
                    ins_actives = [False] * len(qs)
                    ins_dense = [r["dense_vector"] for r in qs]
                    ins_sparse = [r["sparse_vector"] for r in qs]
                    
                    HyBridSearch.collection.insert([
                        ins_intents, ins_texts, ins_actives, ins_dense, ins_sparse
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
                intent_ids = [item.intent_id] * n
                text_list = list(item.texts)
                is_actives = [item.is_active] * n
                dense_vecs = [encoded['dense_vecs'][j].tolist() for j in range(n)]
                sparse_vecs = [encoded['lexical_weights'][j] for j in range(n)]

                entities = [intent_ids, text_list, is_actives, dense_vecs, sparse_vecs]
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

                intent_ids = [r['intent_id'] for r in batch]
                text_list = [r['text'] for r in batch]
                is_actives = [True] * len(batch)
                dense_vecs = [encoded['dense_vecs'][j].tolist() for j in range(len(batch))]
                sparse_vecs = [encoded['lexical_weights'][j] for j in range(len(batch))]

                entities = [intent_ids, text_list, is_actives, dense_vecs, sparse_vecs]
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


@app.post("/compare")
async def compare_data(req: CompareRequest):
    """混合检索核心逻辑（异步 + 批处理 + 分段耗时监控）"""
    try:
        total_start = time.perf_counter()

        # -------- Phase 1: 模型编码（通过批处理器）--------
        t1 = time.perf_counter()
        encoded = await HyBridSearch.batcher.encode(req.text)
        query_dense = [encoded['dense_vec']]
        query_sparse = [encoded['sparse_vec']]
        t1_cost = (time.perf_counter() - t1) * 1000

        # -------- Phase 2: 构建检索请求 --------
        t2 = time.perf_counter()
        req_dense = AnnSearchRequest(
            data=query_dense,
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=req.top_k,
            expr="is_active == true"
        )
        req_sparse = AnnSearchRequest(
            data=query_sparse,
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
            limit=req.top_k,
            expr="is_active == true"
        )
        t2_cost = (time.perf_counter() - t2) * 1000

        # -------- Phase 3: Milvus 混合检索（只检索启用的语料）--------
        t3 = time.perf_counter()
        results = HyBridSearch.collection.hybrid_search(
            reqs=[req_dense, req_sparse],
            rerank=RRFRanker(),
            limit=req.top_k,
            output_fields=["text", "intent_id"]
        )
        t3_cost = (time.perf_counter() - t3) * 1000

        # -------- Phase 4: 结果格式化 + 置信度计算 --------
        t4 = time.perf_counter()
        match_list = []
        for hit in results[0]:
            match_list.append({
                "intent_id": hit.entity.get("intent_id"),
                "text": hit.entity.get("text"),
                "score": round(hit.distance, 4)
            })

        # ---- 置信度判断（比值策略，适配 RRF 评分）----
        LOW_SCORE_THRESHOLD = 0.010
        HIGH_RATIO_THRESHOLD = 1.05

        confidence_status = "UNKNOWN"
        debug_gap = None
        debug_ratio = None

        if len(match_list) > 0:
            top1_score = match_list[0]["score"]

            if top1_score < LOW_SCORE_THRESHOLD:
                confidence_status = "LOW_CONFIDENCE"
            elif len(match_list) > 1:
                top2_score = match_list[1]["score"]
                debug_gap = round(top1_score - top2_score, 6)
                debug_ratio = round(top1_score / top2_score, 4) if top2_score > 0 else float('inf')

                if debug_ratio >= HIGH_RATIO_THRESHOLD:
                    confidence_status = "HIGH_CONFIDENCE"
                else:
                    confidence_status = "NEEDS_CLARIFICATION"
            else:
                confidence_status = "HIGH_CONFIDENCE"
        t4_cost = (time.perf_counter() - t4) * 1000

        total_cost = (time.perf_counter() - total_start) * 1000

        # 日志输出
        logger.info(
            f"[查询] text='{req.text}' | 编码={t1_cost:.1f}ms 检索={t3_cost:.1f}ms 总计={total_cost:.1f}ms | "
            f"{confidence_status}"
        )
        if len(match_list) >= 2:
            logger.debug(
                f"[评分] Top1={match_list[0]['score']:.4f}({match_list[0]['intent_id']}) "
                f"Top2={match_list[1]['score']:.4f}({match_list[1]['intent_id']}) "
                f"Gap={debug_gap} Ratio={debug_ratio}"
            )

        return {
            "code": 200,
            "msg": "success",
            "query_text": req.text,
            "confidence_status": confidence_status,
            "matches": match_list,
            "time_cost_ms": {
                "total": round(total_cost, 2),
                "model_encode": round(t1_cost, 2),
                "build_request": round(t2_cost, 2),
                "milvus_search": round(t3_cost, 2),
                "result_process": round(t4_cost, 2)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"混合检索失败: {str(e)}")


# ================= 意图识别（生产接口）=================

# --- 向量检索核心方法：直接调用 HyBridSearch 组件 ---
RECOGNIZE_TOP_K = 4

async def fetch_intent_from_vector_db(text: str) -> str:
    """
    直接调用 HyBridSearch 模块中的 batcher（模型编码）和 collection（Milvus 混合检索），
    无需 HTTP 请求，本地方法调用。
    """
    total_start = time.perf_counter()

    # ---- Phase 1: 通过 DynamicBatcher 编码文本 ----
    t1 = time.perf_counter()
    encoded = await HyBridSearch.batcher.encode(text)
    query_dense = [encoded['dense_vec']]
    query_sparse = [encoded['sparse_vec']]
    t1_cost = (time.perf_counter() - t1) * 1000

    # ---- Phase 2: 构建混合检索请求 ----
    req_dense = AnnSearchRequest(
        data=query_dense,
        anns_field="dense_vector",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=RECOGNIZE_TOP_K,
        expr="is_active == true"
    )
    req_sparse = AnnSearchRequest(
        data=query_sparse,
        anns_field="sparse_vector",
        param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
        limit=RECOGNIZE_TOP_K,
        expr="is_active == true"
    )

    # ---- Phase 3: Milvus 混合检索 ----
    t3 = time.perf_counter()
    results = HyBridSearch.collection.hybrid_search(
        reqs=[req_dense, req_sparse],
        rerank=RRFRanker(),
        limit=RECOGNIZE_TOP_K,
        output_fields=["text", "intent_id"]
    )
    t3_cost = (time.perf_counter() - t3) * 1000

    # ---- Phase 4: 提取 Top1 的 intent_id ----
    intent_id = "intent_unknown"
    if results and results[0]:
        top_hit = results[0][0]
        intent_id = top_hit.entity.get("intent_id", "intent_unknown")
        top_score = round(top_hit.distance, 4)
        logger.info(
            f"[VectorDB] 检索完成: text='{text}' | "
            f"Top1: intent_id={intent_id}, score={top_score} | "
            f"编码={t1_cost:.1f}ms, 检索={t3_cost:.1f}ms, "
            f"总计={(time.perf_counter() - total_start) * 1000:.1f}ms"
        )
    else:
        logger.warning(f"[VectorDB] 未找到匹配结果: text='{text}'")

    return intent_id


# --- 异步缓存写入 ---
async def set_cache_background(key: str, intent_id: str):
    """缓存意图识别结果，例如缓存24小时"""
    await redis_client.set(key, intent_id, ex=86400)


# --- 意图识别接口 ---
@app.post("/api/v1/recognize", response_model=IntentResponse)
async def recognize_intent(request: IntentRequest, background_tasks: BackgroundTasks):
    # === 步骤 1: 请求接收与预处理 ===
    clean_text = re.sub(r'<[^>]+>', '', request.text)
    clean_text = re.sub(r'[^\w\s\u4e00-\u9fa5,.?!，。？！]', '', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    if not clean_text:
        raise HTTPException(status_code=400, detail="Text is empty after preprocessing.")

    if len(clean_text) > MAX_TEXT_LENGTH:
        raise HTTPException(status_code=400, detail="Text exceeds maximum length limit.")

    # 计算文本 Hash，作为缓存 Key
    text_hash = hashlib.md5(clean_text.encode('utf-8')).hexdigest()
    match_cache_key = f"intent:match:{text_hash}"

    # === 步骤 2: L1 缓存层 (精确匹配) ===
    intent_id = await redis_client.get(match_cache_key)

    # === 步骤 3: 向量化检索 (未命中缓存时触发) ===
    if not intent_id:
        try:
            # 使用 SingleFlight 机制防止高并发导致的缓存击穿
            intent_id = await singleflight.do(
                match_cache_key,
                fetch_intent_from_vector_db,
                clean_text
            )
            # 获取结果后，通过后台任务异步写入 Redis，不阻塞当前 HTTP 响应
            background_tasks.add_task(set_cache_background, match_cache_key, intent_id)

        except Exception as e:
            logger.error(f"[Error] 向量检索失败: {e}")
            # 熔断/降级策略：如果底层服务出问题，返回默认意图
            intent_id = "intent_unknown"

    # === 步骤 4: 根据 Intent ID 获取提前生成的语音 ===
    voice_cache_key = f"voice:intent:{intent_id}"
    voice_url = await redis_client.get(voice_cache_key)

    if not voice_url:
        logger.warning(f"[Voice Cache Miss] Intent: {intent_id} 未找到对应语音，使用默认兜底。")
        voice_url = "https://your-cdn.com/fallback_voice.mp3"

    return IntentResponse(
        intent_id=intent_id,
        voice_url=voice_url
    )