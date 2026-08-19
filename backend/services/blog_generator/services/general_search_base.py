"""
通用搜索抽象基类 — 所有通用搜索源（Zhipu/Tavily/AnySearch）实现此接口

支持通过环境变量 GENERAL_SEARCH_PROVIDER 切换通用搜索源，平替智谱搜索。
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class GeneralSearchBase(ABC):
    """通用搜索抽象基类"""

    name: str = ""  # 搜索源名称标识，如 "zhipu" / "tavily" / "anysearch"

    @abstractmethod
    def is_available(self) -> bool:
        """检查服务是否可用（API Key 已配置等）"""
        ...

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        执行搜索，返回统一格式

        Returns:
            {
                'success': True/False,
                'results': [{'title': ..., 'url': ..., 'content': ..., 'source': ...}],
                'summary': '...',
                'error': '...'
            }
        """
        ...

    def search_for_topic(self, topic: str, article_type: str = '',
                         target_audience: str = '') -> Dict[str, Any]:
        """
        针对技术主题搜索（默认实现，可被子类覆盖）
        """
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

    @staticmethod
    def _generate_summary_from_results(results: List[Dict[str, Any]]) -> str:
        """从结果列表生成摘要文本"""
        if not results:
            return ''
        parts = []
        for i, item in enumerate(results, 1):
            content = item.get('content', '')[:2000]
            if content:
                parts.append(f"{i}. {content}")
        return '\n\n'.join(parts)