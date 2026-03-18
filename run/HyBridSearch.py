"""
HyBridSearch — 混合检索核心组件模块（纯工具模块，不启动服务）
================================================================
提供：
  - DynamicBatcher：动态批处理器（攒批 GPU 推理）
  - InsertRequest / CompareRequest：Pydantic 校验模型
  - init_components()：加载模型、连接 Milvus、初始化批处理器
  - cleanup_components()：释放模型、断开 Milvus
  - 全局变量：collection, bge_model, reranker_model, batcher

使用方式（由 Intent.py 调用）：
  import HyBridSearch
  HyBridSearch.init_components()   # 启动时调用
  await HyBridSearch.batcher.encode(text)
  HyBridSearch.collection.hybrid_search(...)
  await HyBridSearch.async_rerank_candidates(text, candidates) # 【新增】重排调用
  HyBridSearch.cleanup_components() # 关闭时调用
"""

import time
import asyncio
import threading
import logging
import math # 【新增】用于 sigmoid 计算
from collections import deque

from typing import List, Optional
from pydantic import BaseModel, validator, Field, ConfigDict
from pymilvus import (
    connections, Collection, FieldSchema, CollectionSchema,
    DataType, utility
)
# 【修改】引入 FlagReranker
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from NacosConfig import config as _cfg

# ================= 1. 日志配置 =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("hybrid-search")

# ================= 2. 配置参数 =================
# 静态常量（不可热更新）
DIM = 1024
MODEL_WORKERS = 4

# 动态配置通过 _cfg 访问（支持 Nacos 热更新）
# _cfg.model_path, _cfg.reranker_model_path, _cfg.milvus_host, _cfg.milvus_port
# _cfg.collection_name, _cfg.max_batch_size, _cfg.max_wait_ms

# COLLECTION_NAME 保持向后兼容（Intent.py /ready 接口读取），实际值从 _cfg 获取
def _get_collection_name() -> str:
    return _cfg.collection_name

# ================= 3. 全局变量 =================
collection = None
bge_model = None
reranker_model = None  # 【新增】重排模型全局变量
batcher = None


# ================= 4. 动态批处理器 =================
class DynamicBatcher:
    # ... (此部分代码保持你的原样，无需修改) ...
    def __init__(self, model, max_batch_size=32, max_wait_ms=5):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms / 1000
        self.queue = deque()
        self.lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._batch_loop, daemon=True, name="batcher")
        self._thread.start()

    def _batch_loop(self):
        while self._running:
            while self._running and not self.queue:
                time.sleep(0.001)
            if not self._running:
                break
            batch = []
            deadline = time.perf_counter() + self.max_wait_ms
            while len(batch) < self.max_batch_size and self._running:
                with self.lock:
                    if self.queue:
                        batch.append(self.queue.popleft())
                if time.perf_counter() >= deadline or not self.queue:
                    if batch:
                        break
                    time.sleep(0.0005)
            if not batch:
                continue
            texts = [item[0] for item in batch]
            t_start = time.perf_counter()
            try:
                encoded = self.model.encode(texts, return_dense=True, return_sparse=True)
                cost = (time.perf_counter() - t_start) * 1000
                logger.info(f"[批处理] batch_size={len(texts)}, 推理耗时={cost:.2f}ms")
                for i, (_, loop, future) in enumerate(batch):
                    result = {
                        'dense_vec': encoded['dense_vecs'][i].tolist(),
                        'sparse_vec': encoded['lexical_weights'][i]
                    }
                    loop.call_soon_threadsafe(future.set_result, result)
            except Exception as e:
                logger.error(f"[批处理] 推理异常: {e}")
                for _, loop, future in batch:
                    loop.call_soon_threadsafe(future.set_exception, e)

    async def encode(self, text: str) -> dict:
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        with self.lock:
            self.queue.append((text, loop, future))
        return await future

    def shutdown(self):
        self._running = False
        self._thread.join(timeout=5)


# ================= 5. Pydantic 数据模型（含校验）=================

# --- 新增 / 批量新增 ---
class InsertItem(BaseModel):
    """单条意图语料"""
    text: str
    intent_id: Optional[int] = Field(default=1, alias="intentId")
    model_id: Optional[int] = Field(default=1, alias="modelId")
    active: bool = True

    @validator('text')
    def text_must_not_be_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError('文本不能为空')
        if len(v) > 500:
            raise ValueError('文本不能超过500字')
        return v


class BatchInsertRequest(BaseModel):
    """批量新增请求"""
    items: List[InsertItem]

    @validator('items')
    def items_not_empty(cls, v):
        if not v:
            raise ValueError('items 不能为空')
        return v


# --- 批量删除 ---
class DeleteRequest(BaseModel):
    """根据 intent_id 批量删除"""
    intent_ids: List[int]
    model_id: Optional[int] = Field(default=1, alias="modelId")

    @validator('intent_ids')
    def ids_not_empty(cls, v):
        v = [x for x in v]
        if not v:
            raise ValueError('intent_ids 不能为空')
        return v


# --- 批量更新（先删后增）---
class UpdateItem(BaseModel):
    """单个意图的更新数据：指定 intent_id + 新的 texts 列表"""
    intent_id: int
    texts: List[str]
    model_id: Optional[int] = Field(default=1, alias="modelId")
    active: bool = True

    @validator('intent_id')
    def id_not_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError('intent_id 不能为空')
        return v

    @validator('texts')
    def texts_not_empty(cls, v):
        v = [t.strip() for t in v if t.strip()]
        if not v:
            raise ValueError('texts 不能为空')
        return v


class BatchUpdateRequest(BaseModel):
    """批量更新请求"""
    items: List[UpdateItem]

    @validator('items')
    def items_not_empty(cls, v):
        if not v:
            raise ValueError('items 不能为空')
        return v


# --- 混合检索 ---
class CompareRequest(BaseModel):
    text: str
    model_id: Optional[int] = Field(default=1, alias="modelId")
    top_k: Optional[int] = Field(default=4, alias="topK")

    @validator('text')
    def text_must_not_be_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError('查询文本不能为空')
        return v
    @validator('top_k')
    def top_k_range(cls, v):
        if v < 1 or v > 20:
            raise ValueError('top_k 必须在 1~20 之间')
        return v

# ================= 6. 重排辅助工具 =================
# 【新增】将分数转为 0~1 的概率
def _sigmoid(x):
    return 1 / (1 + math.exp(-x))

# 【新增】提供给 Intent.py 调用的异步重排方法
async def async_rerank_candidates(query: str, candidates: List[dict]) -> List[dict]:
    """
    异步对 Milvus 召回的候选集进行精排打分。
    要求 candidates 列表中的每个字典必须包含 "text" 字段。
    """
    global reranker_model
    if not candidates or not reranker_model:
        return candidates

    # 构造交叉打分对
    sentence_pairs = [[query, item["text"]] for item in candidates]

    # 使用 asyncio.to_thread 放入线程池执行，防止阻塞主异步事件循环
    t_start = time.perf_counter()
    scores = await asyncio.to_thread(reranker_model.compute_score, sentence_pairs)
    cost = (time.perf_counter() - t_start) * 1000
    logger.info(f"[精排处理] 候选数量={len(candidates)}, 推理耗时={cost:.2f}ms")

    # 处理单返回值格式
    if isinstance(scores, float):
        scores = [scores]

    # 将分数回填并计算概率
    for i, item in enumerate(candidates):
        item["rerank_raw_score"] = scores[i]
        item["probability"] = round(_sigmoid(scores[i]), 4)

    # 降序排列
    sorted_candidates = sorted(candidates, key=lambda x: x["probability"], reverse=True)
    return sorted_candidates

# ================= 7. 初始化 / 清理 =================
def init_components():
    """加载模型、连接 Milvus、初始化批处理器（由服务启动时调用）"""
    global collection, bge_model, reranker_model, batcher # 【修改】引入 reranker_model

    # 1. 加载模型
    logger.info("正在加载 bge-m3 embedding 模型...")
    bge_model = BGEM3FlagModel(_cfg.model_path, use_fp16=True)
    logger.info("Embedding 模型加载完成")

    # 2. 加载 bge-reranker-v2-m3 模型
    logger.info("正在加载 bge-reranker-v2-m3 重排模型...")
    reranker_model = FlagReranker(_cfg.reranker_model_path, use_fp16=True)
    logger.info("重排模型加载完成")

    # 3. 连接 Milvus
    logger.info("正在连接 Milvus...")
    connections.connect(alias="default", host=_cfg.milvus_host, port=_cfg.milvus_port)

    col_name = _cfg.collection_name
    if not utility.has_collection(col_name):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="model_id", dtype=DataType.INT32),
            FieldSchema(name="intent_id", dtype=DataType.INT32),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="is_active", dtype=DataType.BOOL),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=DIM),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR)
        ]
        schema = CollectionSchema(fields=fields, description="意图识别双向量混合库 v2")
        collection = Collection(name=col_name, schema=schema)

        dense_index = {"metric_type": "COSINE", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 256}}
        collection.create_index(field_name="dense_vector", index_params=dense_index)

        sparse_index = {"metric_type": "IP", "index_type": "SPARSE_INVERTED_INDEX", "params": {"drop_ratio_build": 0.2}}
        collection.create_index(field_name="sparse_vector", index_params=sparse_index)

        logger.info("新建 Collection 并创建双索引成功")
    else:
        collection = Collection(col_name)
        logger.info("已连接到现有 Collection")

    collection.load()

    # 4. 模型预热
    logger.info("正在预热模型...")
    warmup_start = time.perf_counter()
    _ = bge_model.encode(["模型预热测试"], return_dense=True, return_sparse=True)
    _ = reranker_model.compute_score([["预热测试", "预热测试"]])
    warmup_cost = (time.perf_counter() - warmup_start) * 1000
    logger.info(f"模型预热完成，总耗时: {warmup_cost:.2f}ms")

    # 5. 初始化动态批处理器
    batcher = DynamicBatcher(bge_model, max_batch_size=_cfg.max_batch_size, max_wait_ms=_cfg.max_wait_ms)
    logger.info(f"动态批处理器已启动 (max_batch={_cfg.max_batch_size}, max_wait={_cfg.max_wait_ms}ms)")

    logger.info("===== HyBridSearch 组件初始化完成 =====")


def cleanup_components():
    """释放批处理器、断开 Milvus（由服务关停时调用）"""
    logger.info("正在清理 HyBridSearch 资源...")
    if batcher:
        batcher.shutdown()
    connections.disconnect("default")
    logger.info("HyBridSearch 资源已释放")