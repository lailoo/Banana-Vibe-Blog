"""
Tavily 搜索服务 — 通过 Tavily Search API 提供通用搜索能力

Tavily (https://tavily.com) 专为 AI Agent 设计，支持高质量搜索结果。
配置环境变量 TAVILY_API_KEY 即可使用。
"""

import logging
import os
from typing import Dict, Any, List, Optional

from .general_search_base import GeneralSearchBase

logger = logging.getLogger(__name__)

_global_tavily_service: Optional['TavilySearchService'] = None


class TavilySearchService(GeneralSearchBase):
    """Tavily 搜索服务"""

    name = "tavily"

    def __init__(self, api_key: str = "", timeout: int = 30, max_results: int = 10):
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY", "")
        self.timeout = timeout
        self.max_results = max_results

    def is_available(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        if not self.api_key:
            return {
                "success": False, "results": [], "summary": "",
                "error": "Tavily API Key 未配置",
            }

        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=self.api_key)

            logger.info(f"🌐 使用 Tavily 搜索: {query}")
            response = client.search(
                query=query,
                max_results=max_results or self.max_results,
                search_depth="advanced",
                include_answer="basic",
            )

            results = self._parse_results(response)
            summary = self._generate_summary_from_results(results)

            logger.info(f"Tavily 搜索完成: {len(results)} 条结果")
            return {
                "success": True,
                "results": results,
                "summary": summary,
                "error": None,
            }

        except Exception as e:
            logger.error(f"Tavily 搜索失败: {e}")
            return {
                "success": False, "results": [], "summary": "",
                "error": f"Tavily 搜索失败: {str(e)}",
            }

    def _parse_results(self, response) -> List[Dict[str, Any]]:
        """解析 Tavily 响应为统一格式"""
        results = []

        # Tavily 的 answer 摘要
        answer = getattr(response, 'answer', None) or response.get('answer', '')
        if answer:
            results.append({
                "title": "Tavily Summary",
                "url": "",
                "content": answer,
                "source": "Tavily AI",
            })

        # 搜索结果
        raw_results = getattr(response, 'results', None) or response.get('results', [])
        for item in raw_results:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                "source": "Tavily",
            })

        return results


def init_tavily_service(config: Dict[str, Any] = None) -> Optional[TavilySearchService]:
    """初始化 Tavily 搜索服务"""
    global _global_tavily_service
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        logger.info("Tavily 搜索: TAVILY_API_KEY 未配置，跳过")
        _global_tavily_service = None
        return None

    _global_tavily_service = TavilySearchService(
        api_key=api_key,
        timeout=int(os.environ.get("TAVILY_TIMEOUT", "30")),
        max_results=int(os.environ.get("TAVILY_MAX_RESULTS", "10")),
    )
    logger.info("Tavily 搜索服务已初始化")
    return _global_tavily_service


def get_tavily_service() -> Optional[TavilySearchService]:
    """获取 Tavily 搜索服务实例"""
    return _global_tavily_service