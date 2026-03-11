"""
HyBridSearch — 混合检索核心组件模块（纯工具模块，不启动服务）
================================================================
提供：
  - DynamicBatcher：动态批处理器（攒批 GPU 推理）
  - InsertRequest / CompareRequest：Pydantic 校验模型
  - init_components()：加载模型、连接 Milvus、初始化批处理器
  - cleanup_components()：释放模型、断开 Milvus
  - 全局变量：collection, bge_model, batcher

使用方式（由 Intent.py 调用）：
  import HyBridSearch
  HyBridSearch.init_components()   # 启动时调用
  await HyBridSearch.batcher.encode(text)
  HyBridSearch.collection.hybrid_search(...)
  HyBridSearch.cleanup_components() # 关闭时调用
"""

import time
import asyncio
import threading
import logging
from collections import deque

from typing import List
from pydantic import BaseModel, validator
from pymilvus import (
    connections, Collection, FieldSchema, CollectionSchema,
    DataType, utility
)
from FlagEmbedding import BGEM3FlagModel

# ================= 1. 日志配置 =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("hybrid-search")

# ================= 2. 配置参数 =================
MODEL_PATH = "/home/zhulie/mf/model_list/bge-m3"
MILVUS_HOST = "release.milvus.com"
MILVUS_PORT = "19530"
COLLECTION_NAME = "intent_hybrid_recognition"  # 新 Schema 用 v2 表名
DIM = 1024

# 批处理参数
MAX_BATCH_SIZE = 32      # 单次 GPU 推理最大批量
MAX_WAIT_MS = 5          # 攒批等待时间（毫秒），越小延迟越低，越大吞吐越高
MODEL_WORKERS = 4        # 线程池大小，按 GPU 显存调整（8GB→2~4, 16GB→4~8）

# ================= 3. 全局变量 =================
collection = None
bge_model = None
batcher = None  # 动态批处理器


# ================= 4. 动态批处理器 =================
class DynamicBatcher:
    """将并发请求攒批后统一 GPU 推理，大幅提升吞吐量"""

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
            # 等待请求到达
            while self._running and not self.queue:
                time.sleep(0.001)
            if not self._running:
                break

            # 攒批：等 max_wait_ms 或凑够 max_batch_size
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

            # 批量推理
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
    intent_id: str = ""
    is_active: bool = True

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
    intent_ids: List[str]

    @validator('intent_ids')
    def ids_not_empty(cls, v):
        v = [x.strip() for x in v if x.strip()]
        if not v:
            raise ValueError('intent_ids 不能为空')
        return v


# --- 批量更新（先删后增）---
class UpdateItem(BaseModel):
    """单个意图的更新数据：指定 intent_id + 新的 texts 列表"""
    intent_id: str
    texts: List[str]
    is_active: bool = True

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
    top_k: int = 4

    @validator('text')
    def text_must_not_be_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError('查询文本不能为空')
        if len(v) > 500:
            raise ValueError('查询文本不能超过500字')
        return v

    @validator('top_k')
    def top_k_range(cls, v):
        if v < 1 or v > 20:
            raise ValueError('top_k 必须在 1~20 之间')
        return v


# ================= 6. 初始化 / 清理 =================
def init_components():
    """加载 bge-m3 模型、连接 Milvus、初始化批处理器（由服务启动时调用）"""
    global collection, bge_model, batcher

    # 1. 加载模型
    logger.info("正在加载 bge-m3 模型...")
    bge_model = BGEM3FlagModel(MODEL_PATH, use_fp16=True)
    logger.info("模型加载完成")

    # 2. 连接 Milvus
    logger.info("正在连接 Milvus...")
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

    if not utility.has_collection(COLLECTION_NAME):
        # ---- 优化后的 Schema（含 intent_id、is_active）----
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="intent_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="is_active", dtype=DataType.BOOL),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=DIM),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR)
        ]
        schema = CollectionSchema(fields=fields, description="意图识别双向量混合库 v2")
        collection = Collection(name=COLLECTION_NAME, schema=schema)

        # 稠密向量索引
        dense_index = {"metric_type": "COSINE", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 256}}
        collection.create_index(field_name="dense_vector", index_params=dense_index)

        # 稀疏向量索引
        sparse_index = {"metric_type": "IP", "index_type": "SPARSE_INVERTED_INDEX", "params": {"drop_ratio_build": 0.2}}
        collection.create_index(field_name="sparse_vector", index_params=sparse_index)

        logger.info("新建 Collection 并创建双索引成功")
    else:
        collection = Collection(COLLECTION_NAME)
        logger.info("已连接到现有 Collection")

    collection.load()

    # 3. 模型预热
    logger.info("正在预热模型...")
    warmup_start = time.perf_counter()
    _ = bge_model.encode(["模型预热测试"], return_dense=True, return_sparse=True)
    warmup_cost = (time.perf_counter() - warmup_start) * 1000
    logger.info(f"模型预热完成，耗时: {warmup_cost:.2f}ms")

    # 4. 初始化动态批处理器
    batcher = DynamicBatcher(bge_model, max_batch_size=MAX_BATCH_SIZE, max_wait_ms=MAX_WAIT_MS)
    logger.info(f"动态批处理器已启动 (max_batch={MAX_BATCH_SIZE}, max_wait={MAX_WAIT_MS}ms)")

    logger.info("===== HyBridSearch 组件初始化完成 =====")


def cleanup_components():
    """释放批处理器、断开 Milvus（由服务关停时调用）"""
    logger.info("正在清理 HyBridSearch 资源...")
    if batcher:
        batcher.shutdown()
    connections.disconnect("default")
    logger.info("HyBridSearch 资源已释放")