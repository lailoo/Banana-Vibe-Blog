"""
搜索服务 - 智谱 Web Search API
参考 AI 绘本项目实现

GENERAL_SEARCH_PROVIDER 环境变量支持切换通用搜索源：
  - zhipu    (默认)  智谱 Web Search
  - tavily             Tavily Search API
  - anysearch          AnySearch API
  - doubao             豆包（FeedCoop Global Search）搜索
"""

import os
import json
import logging
import requests
from typing import Dict, Any, List, Optional

from .general_search_base import GeneralSearchBase

logger = logging.getLogger(__name__)

# 全局搜索服务实例（多态，类型为 GeneralSearchBase）
_search_service: Optional[GeneralSearchBase] = None


class SearchService(GeneralSearchBase):
    """
    搜索服务 - 智谱 Web Search API

    实现 GeneralSearchBase 接口，可作为 GENERAL_SEARCH_PROVIDER=zhipu 时的通用搜索源。
    """

    name = "zhipu"

    def __init__(self, api_key: str, config: Dict[str, Any] = None):
        """
        初始化搜索服务

        Args:
            api_key: 智谱 API Key
            config: 配置字典
        """
        self.api_key = api_key
        self.config = config or {}

    def is_available(self) -> bool:
        """检查服务是否可用"""
        return bool(self.api_key)

    def search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        搜索背景知识

        Args:
            query: 搜索关键词
            max_results: 最大结果数

        Returns:
            {
                'success': True/False,
                'results': [...],  # 搜索结果列表
                'summary': '...',  # 摘要
                'error': '...'     # 错误信息
            }
        """
        if not self.api_key:
            logger.warning("智谱 API Key 未配置，跳过搜索")
            return {
                'success': False,
                'results': [],
                'summary': '',
                'error': '智谱 API Key 未配置'
            }

        try:
            logger.info(f"使用智谱 Web Search 搜索: {query}")
            return self._search_zai(query, max_results)

        except Exception as e:
            logger.error(f"搜索失败: {e}", exc_info=True)
            return {
                'success': False,
                'results': [],
                'summary': '',
                'error': str(e)
            }

    def _search_zai(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """使用智谱 Web Search API 搜索"""
        try:
            # 从配置中获取 API 参数
            url = self.config.get('ZAI_SEARCH_API_BASE') or os.environ.get(
                'ZAI_SEARCH_API_BASE',
                'https://open.bigmodel.cn/api/paas/v4/web_search'
            )
            search_engine = self.config.get('ZAI_SEARCH_ENGINE') or os.environ.get('ZAI_SEARCH_ENGINE', 'search_std')
            max_count = int(self.config.get('ZAI_SEARCH_MAX_RESULTS') or os.environ.get('ZAI_SEARCH_MAX_RESULTS', '10'))
            content_size = self.config.get('ZAI_SEARCH_CONTENT_SIZE') or os.environ.get('ZAI_SEARCH_CONTENT_SIZE', 'medium')
            recency_filter = self.config.get('ZAI_SEARCH_RECENCY_FILTER') or os.environ.get('ZAI_SEARCH_RECENCY_FILTER', 'noLimit')

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "search_query": query,
                "search_engine": search_engine,
                "search_intent": False,
                "count": min(max_results, max_count, 50),
                "content_size": content_size,
                "search_recency_filter": recency_filter
            }

            logger.info(f"🌐 使用智谱 Web Search 搜索: {query}")
            logger.info(f"🌐 API URL: {url}")
            logger.info(f"🌐 请求参数: {json.dumps(payload, ensure_ascii=False)}")

            response = requests.post(url, json=payload, headers=headers, timeout=30)
            logger.info(f"API 响应状态码: {response.status_code}")
            response.raise_for_status()

            data = response.json()
            logger.info(f"智谱搜索完整响应: {json.dumps(data, ensure_ascii=False)}")

            # 解析搜索结果（统一格式）
            parsed_results = []
            search_results = data.get('search_result', [])
            logger.info(f"搜索结果数量: {len(search_results)}")

            for item in search_results:
                parsed_results.append({
                    'title': item.get('title', ''),
                    'url': item.get('link', ''),
                    'content': item.get('content', ''),
                    'source': item.get('media', ''),
                    'publish_date': item.get('publish_date', '')
                })

            # 生成摘要
            summary = self._generate_summary(parsed_results)

            logger.info(f"智谱搜索完成，获取 {len(parsed_results)} 条结果")

            return {
                'success': True,
                'results': parsed_results,
                'summary': summary,
                'error': None
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"智谱 API 请求失败: {e}")
            return {
                'success': False,
                'results': [],
                'summary': '',
                'error': f'智谱 API 请求失败: {str(e)}'
            }

    def _generate_summary(self, results: List[Dict[str, Any]]) -> str:
        """从搜索结果生成摘要"""
        if not results:
            return ''

        summary_parts = []
        for i, item in enumerate(results, 1):
            if item.get('content'):
                summary_parts.append(f"{i}. {item['content'][:2000]}")

        return '\n\n'.join(summary_parts)

    def search_for_topic(self, topic: str, article_type: str = '', target_audience: str = '') -> Dict[str, Any]:
        """
        针对技术主题搜索背景知识

        Args:
            topic: 技术主题，如 "LangGraph"
            article_type: 文章类型，如 "tutorial"
            target_audience: 目标受众，如 "intermediate"

        Returns:
            搜索结果
        """
        # 构建搜索查询
        query_parts = [topic]

        if article_type == 'tutorial':
            query_parts.append("教程 入门指南")
        elif article_type == 'problem-solution':
            query_parts.append("问题解决 最佳实践")
        elif article_type == 'comparison':
            query_parts.append("对比 选型")

        if target_audience == 'beginner':
            query_parts.append("入门 基础")
        elif target_audience == 'advanced':
            query_parts.append("高级 深入")

        query = ' '.join(query_parts)
        return self.search(query)


def _create_provider(provider: str, config: Dict[str, Any]) -> Optional[GeneralSearchBase]:
    """根据提供商名称创建对应的搜索服务实例"""
    if provider == 'zhipu':
        zai_api_key = config.get('ZAI_SEARCH_API_KEY') or os.environ.get('ZAI_SEARCH_API_KEY', '')
        if zai_api_key:
            return SearchService(api_key=zai_api_key, config=config)
        return None

    elif provider == 'tavily':
        from .tavily_search_service import TavilySearchService
        api_key = os.environ.get('TAVILY_API_KEY', '')
        if api_key:
            return TavilySearchService(
                api_key=api_key,
                timeout=int(os.environ.get('TAVILY_TIMEOUT', '30')),
                max_results=int(os.environ.get('TAVILY_MAX_RESULTS', '10')),
            )
        return None

    elif provider == 'anysearch':
        from .anysearch_service import AnySearchService
        import json as _json
        params_raw = os.environ.get('ANYSEARCH_PARAMS', '')
        params = None
        if params_raw:
            try:
                params = _json.loads(params_raw)
            except _json.JSONDecodeError:
                pass
        return AnySearchService(
            api_key=os.environ.get('ANYSEARCH_API_KEY', ''),
            api_base=os.environ.get('ANYSEARCH_API_BASE', ''),
            timeout=int(os.environ.get('ANYSEARCH_TIMEOUT', '30')),
            max_results=int(os.environ.get('ANYSEARCH_MAX_RESULTS', '10')),
            tag=os.environ.get('ANYSEARCH_TAG', ''),
            zone=os.environ.get('ANYSEARCH_ZONE', ''),
            language=os.environ.get('ANYSEARCH_LANGUAGE', ''),
            output_format=os.environ.get('ANYSEARCH_FORMAT', ''),
            params=params,
        )

    elif provider == 'doubao':
        from .doubao_search_service import DoubaoSearchService
        api_key = os.environ.get('DOUBAO_WEB_SEARCH_API_KEY', '')
        if api_key:
            return DoubaoSearchService(
                api_key=api_key,
                api_base= DoubaoSearchService.BASE_URL,
                timeout=int(os.environ.get('DOUBAO_WEB_SEARCH_TIMEOUT', '30')),
                max_results=int(os.environ.get('DOUBAO_WEB_SEARCH_MAX_RESULTS', '10')),
            )
        return None

    else:
        logger.warning(f"未知的通用搜索提供商: {provider}，回退到智谱")
        return None


def init_search_service(config: Dict[str, Any] = None) -> Optional[GeneralSearchBase]:
    """
    初始化搜索服务（工厂模式，支持切换通用搜索源）

    通过环境变量 GENERAL_SEARCH_PROVIDER 选择提供商：
      - zhipu  (默认)  智谱 Web Search
      - tavily           Tavily Search API
      - anysearch        AnySearch API
      - doubao           豆包（FeedCoop Global Search）搜索

    Args:
        config: Flask 配置字典
    """
    global _search_service

    config = config or {}
    provider = (config.get('GENERAL_SEARCH_PROVIDER')
                or os.environ.get('GENERAL_SEARCH_PROVIDER', 'zhipu')).lower().strip()

    _search_service = _create_provider(provider, config)

    if _search_service:
        logger.info(f"通用搜索服务已初始化 (提供商: {provider})")
    else:
        # 回退：如果指定提供商不可用，尝试其他可用提供商
        logger.warning(f"提供商 {provider} 不可用或无 API Key，尝试回退...")
        for fallback in ['zhipu', 'tavily', 'anysearch', 'doubao']:
            if fallback == provider:
                continue
            _search_service = _create_provider(fallback, config)
            if _search_service:
                logger.info(f"通用搜索服务已初始化 (回退至: {fallback})")
                break

        if not _search_service:
            logger.warning("搜索服务初始化: 未配置任何 API Key，搜索功能不可用")

    return _search_service


def get_search_service() -> Optional[GeneralSearchBase]:
    """获取搜索服务实例"""
    return _search_service
