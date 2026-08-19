"""
AnySearch 搜索服务 — 通过 AnySearch API 提供通用搜索能力

AnySearch (https://anysearch.com) 是一个统一的搜索 API，可聚合多源搜索结果。
支持无 API Key 使用（留空则不传 Authorization 头）。

环境变量:
  ANYSEARCH_API_KEY  - API Key（可选，留空则不传）
  ANYSEARCH_API_BASE - API 地址（默认 https://api.anysearch.com/v1/search）
  ANYSEARCH_TIMEOUT  - 超时秒数（默认 30）
  ANYSEARCH_MAX_RESULTS - 最大结果数（默认 10）
  ANYSEARCH_TAG      - 子域能力标签，如 "code.doc"
  ANYSEARCH_ZONE     - 地区，cn 或 intl
  ANYSEARCH_LANGUAGE - 偏好语言，如 zh-CN 或 en
  ANYSEARCH_FORMAT   - 输出格式，json 或 markdown
  ANYSEARCH_PARAMS   - 额外参数的 JSON 字符串，如 '{"source": "web"}'
"""

import json
import logging
import os
from typing import Dict, Any, List, Optional

import requests

from .general_search_base import GeneralSearchBase

logger = logging.getLogger(__name__)

_global_anysearch_service: Optional['AnySearchService'] = None


class AnySearchService(GeneralSearchBase):
    """AnySearch 搜索服务"""

    name = "anysearch"
    BASE_URL = "https://api.anysearch.com/v1/search"

    def __init__(self, api_key: str = "", api_base: str = "",
                 timeout: int = 30, max_results: int = 10,
                 tag: str = "", zone: str = "", language: str = "",
                 output_format: str = "", params: Optional[Dict[str, Any]] = None):
        self.api_key = api_key or os.environ.get("ANYSEARCH_API_KEY", "")
        self.api_base = api_base or os.environ.get("ANYSEARCH_API_BASE", self.BASE_URL)
        self.timeout = timeout
        self.max_results = max_results
        self.tag = tag or os.environ.get("ANYSEARCH_TAG", "")
        self.zone = zone or os.environ.get("ANYSEARCH_ZONE", "")
        self.language = language or os.environ.get("ANYSEARCH_LANGUAGE", "")
        self.output_format = output_format or os.environ.get("ANYSEARCH_FORMAT", "")
        # params 支持从环境变量读取 JSON 字符串
        self.params = params
        if self.params is None and os.environ.get("ANYSEARCH_PARAMS"):
            try:
                self.params = json.loads(os.environ.get("ANYSEARCH_PARAMS", "{}"))
            except json.JSONDecodeError:
                self.params = None

    def is_available(self) -> bool:
        """AnySearch 支持无 Key 使用，始终可用"""
        return True

    def search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        try:
            headers = {"Content-Type": "application/json"}
            # 只有配置了 API Key 才传 Authorization
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            payload = {
                "query": query,
                "max_results": max_results or self.max_results,
            }
            # 可选参数：仅当有值时传入
            if self.tag:
                payload["tag"] = self.tag
            if self.zone:
                payload["zone"] = self.zone
            if self.language:
                payload["language"] = self.language
            if self.output_format:
                payload["format"] = self.output_format
            if self.params:
                payload["params"] = self.params

            logger.info(f"🌐 使用 AnySearch 搜索: {query}"
                        f"{' (无 Key)' if not self.api_key else ''}")
            response = requests.post(
                self.api_base, json=payload, headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            results = self._parse_results(data)
            summary = self._generate_summary_from_results(results)

            logger.info(f"AnySearch 搜索完成: {len(results)} 条结果")
            return {
                "success": True,
                "results": results,
                "summary": summary,
                "error": None,
            }

        except Exception as e:
            logger.error(f"AnySearch 搜索失败: {e}")
            return {
                "success": False, "results": [], "summary": "",
                "error": f"AnySearch 搜索失败: {str(e)}",
            }

    def _parse_results(self, data: Dict) -> List[Dict[str, Any]]:
        """解析 AnySearch 响应为统一格式

        兼容多种响应结构:
        1. { "data": { "results": [ {...}, ... ] } }       (anysearch.com 实际格式)
        2. { "data": [ {...}, ... ] }
        3. { "results": [ {...}, ... ] }  /  { "items": [...] }
        """
        results = []

        # 收集原始条目列表（兼容多层嵌套）
        raw_items: List[Any] = []
        for key in ("results", "data", "items"):
            value = data.get(key)
            if isinstance(value, list):
                raw_items.extend(value)
            elif isinstance(value, dict):
                # 嵌套：{ data: { results: [...] } }
                for sub_key in ("results", "items", "data", "list"):
                    sub = value.get(sub_key)
                    if isinstance(sub, list):
                        raw_items.extend(sub)
                        break

        # 兜底：解析失败时按顶层 dict 直接遍历
        if not raw_items:
            for item in data.values():
                if isinstance(item, dict) and ("title" in item or "url" in item or "link" in item):
                    raw_items.append(item)

        for item in raw_items:
            if not isinstance(item, dict):
                continue
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", item.get("link", "")),
                "content": item.get("content", item.get("snippet", "")),
                "source": "AnySearch",
            })

        return results


def init_anysearch_service(config: Dict[str, Any] = None) -> AnySearchService:
    """初始化 AnySearch 搜索服务（AnySearch 支持无 Key 使用）"""
    global _global_anysearch_service

    params_raw = os.environ.get("ANYSEARCH_PARAMS", "")
    params = None
    if params_raw:
        try:
            params = json.loads(params_raw)
        except json.JSONDecodeError:
            logger.warning(f"ANYSEARCH_PARAMS JSON 解析失败: {params_raw}")

    _global_anysearch_service = AnySearchService(
        api_key=os.environ.get("ANYSEARCH_API_KEY", ""),
        api_base=os.environ.get("ANYSEARCH_API_BASE", AnySearchService.BASE_URL),
        timeout=int(os.environ.get("ANYSEARCH_TIMEOUT", "30")),
        max_results=int(os.environ.get("ANYSEARCH_MAX_RESULTS", "10")),
        tag=os.environ.get("ANYSEARCH_TAG", ""),
        zone=os.environ.get("ANYSEARCH_ZONE", ""),
        language=os.environ.get("ANYSEARCH_LANGUAGE", ""),
        output_format=os.environ.get("ANYSEARCH_FORMAT", ""),
        params=params,
    )
    logger.info("AnySearch 搜索服务已初始化")
    return _global_anysearch_service


def get_anysearch_service() -> Optional[AnySearchService]:
    """获取 AnySearch 搜索服务实例"""
    return _global_anysearch_service