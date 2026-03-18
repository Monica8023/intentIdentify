from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

connections.connect(host='release.milvus.com', port='19530')
collection_name = "intent_hybrid_recognition_v3"

# 如果存在老表，先删掉 (演示用，生产环境慎用！)
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)

# 1. 定义 Schema（v2：含 intent_id、is_active）
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    # 模型ID
    FieldSchema(name="model_id", dtype=DataType.INT32),
    # 意图ID：同一意图下的多条语料共享相同 intent_id
    FieldSchema(name="intent_id", dtype=DataType.INT32),
    # 语料文本（扩展到 500 字符）
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500),
    # 是否启用（支持软删除）
    FieldSchema(name="is_active", dtype=DataType.BOOL),
    # 稠密向量 (1024维)
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
    # 稀疏向量
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR)
]
schema = CollectionSchema(fields=fields, description="意图识别-混合检索库 v2")
collection = Collection(name=collection_name, schema=schema)

# 2. 分别创建两种索引
# 稠密向量索引 (HNSW + COSINE)
dense_index = {"metric_type": "COSINE", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 256}}
collection.create_index("dense_vector", dense_index)

# 稀疏向量索引 (SPARSE_INVERTED_INDEX + IP)
sparse_index = {"metric_type": "IP", "index_type": "SPARSE_INVERTED_INDEX", "params": {"drop_ratio_build": 0.2}}
collection.create_index("sparse_vector", sparse_index)

print("混合检索 Collection v3 初始化完毕！")