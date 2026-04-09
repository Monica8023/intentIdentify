"""
NacosConfig — 配置中心客户端（支持 Nacos 热更新 + 本地降级）
=============================================================
使用方式：
  from NacosConfig import config   # 全局配置单例

  config.milvus_host               # 当前 Milvus 主机
  config.redis_url                 # 当前 Redis URL
  config.low_score_threshold       # 置信度阈值（可热更新）

环境变量：
  NACOS_DISABLED=true   禁用 Nacos，使用默认值（本地开发用）
  NACOS_SERVER_ADDR     Nacos 服务地址，默认 nacos.register.service.com:8848
  NACOS_NAMESPACE       命名空间 ID，默认 intent_test
  NACOS_DATA_ID         Data ID，默认 intent-service.yaml
  NACOS_GROUP           Group，默认 dolphin
  NACOS_USERNAME        认证用户名
  NACOS_PASSWORD        认证密码
"""

import asyncio
import logging
import os
import threading

import yaml

logger = logging.getLogger("nacos-config")


class AppConfig:
    """应用配置，字段与 Nacos YAML 保持一致；支持线程安全热更新"""

    def __init__(self):
        # --- 模型路径 ---
        self.model_path: str = "D:\\model\\bge"
        self.reranker_model_path: str = "D:\\model\\bge-reranker-v2-m3"

        # --- Milvus ---
        self.milvus_host: str = "release.milvus.com"
        self.milvus_port: str = "19530"
        self.collection_name: str = "intent_hybrid_recognition_test"

        # --- Redis ---
        self.redis_url: str = "redis://dev.redis.service.com:6379/9"

        # --- 批处理 ---
        self.max_batch_size: int = 32
        self.max_wait_ms: int = 5

        # --- 阈值 ---
        self.low_score_threshold: float = 0.78
        self.high_gap_threshold: float = 0.10
        self.raw_score_min: float = 1.5

        self._lock = threading.RLock()

    def update_from_dict(self, data: dict):
        """从字典热更新配置（线程安全）"""
        with self._lock:
            for key, value in data.items():
                if hasattr(self, key):
                    target_type = type(getattr(self, key))
                    try:
                        setattr(self, key, target_type(value))
                    except (ValueError, TypeError) as e:
                        logger.warning(f"[Config] 字段 '{key}' 类型转换失败: {e}")
                else:
                    logger.debug(f"[Config] 未知字段，跳过: '{key}'")
        logger.info("[Config] 配置已更新")

    def __getattr__(self, name):
        # 防止 _lock 初始化前递归
        raise AttributeError(f"AppConfig 没有字段 '{name}'")


# ─── 全局单例 ───
config = AppConfig()


def _flatten(data: dict, prefix: str = "") -> dict:
    """将嵌套 YAML dict 展平为 snake_case key（一层嵌套）"""
    result = {}
    for k, v in data.items():
        flat_key = f"{prefix}{k}".replace("-", "_").replace(".", "_")
        if isinstance(v, dict):
            result.update(_flatten(v, f"{flat_key}_"))
        else:
            result[flat_key] = v
    return result


def _apply_yaml(content: str):
    try:
        data = yaml.safe_load(content)
        if isinstance(data, dict):
            config.update_from_dict(_flatten(data))
    except Exception as e:
        logger.error(f"[Config] YAML 解析失败: {e}")


async def init_config(
    nacos_server: str = "nacos.register.service.com:8848",
    nacos_namespace: str = "intent_test",
    nacos_data_id: str = "intent-server-dev.yaml",
    nacos_group: str = "dolphin",
    poll_interval_s: int = 30,
) -> None:
    """
    初始化配置（async）：
    - NACOS_DISABLED=true → 仅用内置默认值，不连 Nacos
    - 否则从 Nacos 拉取初始配置，注册 async 热更新监听，并启动轮询兜底
    兼容 nacos-sdk-python 3.x（包路径 v2.nacos，全异步 API）
    """
    if os.environ.get("NACOS_DISABLED", "").lower() == "true":
        logger.info("[Config] NACOS_DISABLED=true，使用默认配置")
        return

    try:
        from v2.nacos import NacosConfigService, ClientConfigBuilder, ConfigParam  # type: ignore
    except ImportError:
        logger.warning("[Config] nacos-sdk-python 未安装，使用默认配置（pip install nacos-sdk-python）")
        return

    try:
        client_config = (
            ClientConfigBuilder()
            .server_address(nacos_server)
            .namespace_id(nacos_namespace)
            .build()
        )
        client_config.disable_use_config_cache = True

        svc = await NacosConfigService.create_config_service(client_config)

        # 拉取初始配置
        content = await svc.get_config(ConfigParam(data_id=nacos_data_id, group=nacos_group))
        if content:
            _apply_yaml(content)
            logger.info(f"[Config] 已从 Nacos 加载配置: {nacos_server} / {nacos_namespace} / {nacos_data_id}")
        else:
            logger.warning("[Config] Nacos 返回空配置，使用默认值")

        # 注册 async 热更新监听（与 dolphin-asr 保持一致）
        async def _on_change(tenant, nacos_data_id, nacos_group, content):
            if not content:
                return
            try:
                logger.info(f"[Config] 收到 Nacos 配置变更，正在热更新... data_id={nacos_data_id}")
                _apply_yaml(content)
            except Exception as e:
                logger.error(f"[Config] 热更新失败: {e}")

        await svc.add_listener(nacos_data_id, nacos_group, _on_change)
        logger.info("[Config] Nacos 热更新监听已注册")

        # 轮询兜底：防止 gRPC 推送不可达时配置无法更新
        async def _poll_loop():
            while True:
                await asyncio.sleep(poll_interval_s)
                try:
                    latest = await svc.get_config(ConfigParam(data_id=nacos_data_id, group=nacos_group))
                    if latest:
                        _apply_yaml(latest)
                        logger.debug("[Config] Nacos 轮询拉取完成")
                except Exception as e:
                    logger.warning(f"[Config] Nacos 轮询失败: {e}")

        asyncio.create_task(_poll_loop())

    except Exception as e:
        logger.error(f"[Config] Nacos 初始化失败，使用默认配置: {e}", exc_info=True)
