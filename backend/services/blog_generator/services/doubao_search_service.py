"""
豆包（Doubao）搜索服务 — 通过 FeedCoop Global Search API 提供通用搜索能力

API 文档: https://open.feedcoopapi.com/search_api/global_search
使用 POST JSON 直调搜索接口，返回结构化搜索结果。

配置环境变量:
  DOUBAO_WEB_SEARCH_API_KEY           - 豆包搜索 API Key（必填）
  DOUBAO_WEB_SEARCH_TIMEOUT           - 超时秒数（默认 30）
  DOUBAO_WEB_SEARCH_MAX_RESULTS       - 返回结果条数，默认 10，最大 20（对应 DocCount）
  DOUBAO_WEB_SEARCH_MAX_SNIPPET_LENGTH - 摘要最大 tokens，默认 500，最大 3000
  DOUBAO_IMAGE_SEARCH_API_KEY         - 豆包搜图 API Key（可选，默认复用 DOUBAO_WEB_SEARCH_API_KEY）
  DOUBAO_IMAGE_SEARCH_TIMEOUT         - 豆包搜图超时秒数（默认 30）
  DOUBAO_IMAGE_SEARCH_MAX_RESULTS     - 豆包搜图返回条数，默认 1，最大 20
"""

import logging
import os
from typing import Dict, Any, List, Optional

import requests

from .general_search_base import GeneralSearchBase

logger = logging.getLogger(__name__)

_global_doubao_service: Optional['DoubaoSearchService'] = None
_global_doubao_image_service: Optional['DoubaoImageSearchService'] = None


class DoubaoSearchService(GeneralSearchBase):
    """豆包搜索服务 — 通过 FeedCoop Global Search API 实现"""

    name = "doubao"
    BASE_URL = "https://open.feedcoopapi.com"

    def __init__(self, api_key: str = "", api_base: str = "",
                 timeout: int = 30, max_results: int = 10,
                 max_snippet_length: int = 500):
        self.api_key = api_key or os.environ.get("DOUBAO_WEB_SEARCH_API_KEY", "")
        self.api_base = (api_base if api_base else self.BASE_URL).rstrip("/")
        self.timeout = timeout
        self.max_results = max_results
        self.max_snippet_length = max_snippet_length

    def is_available(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        if not self.api_key:
            return {
                "success": False, "results": [], "summary": "",
                "error": "豆包搜索 API Key 未配置",
            }

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "Query": query,
                "SearchType": "web",
                "DocCount": min(max_results or self.max_results, 20),
                "MaxSnippetLength": min(self.max_snippet_length, 3000),
                "MaxImageCountPerDoc": 0,
            }

            url = f"{self.api_base}/search_api/global_search"
            logger.info(f"🌐 使用豆包搜索: {query}")
            response = requests.post(
                url, json=payload, headers=headers, timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            results = self._parse_results(data)
            summary = self._generate_summary_from_results(results)

            logger.info(f"豆包搜索完成: {len(results)} 条结果")
            return {
                "success": True,
                "results": results,
                "summary": summary,
                "error": None,
            }

        except Exception as e:
            logger.error(f"豆包搜索失败: {e}")
            return {
                "success": False, "results": [], "summary": "",
                "error": f"豆包搜索失败: {str(e)}",
            }

    def _parse_results(self, data: Dict) -> List[Dict[str, Any]]:
        """解析 FeedCoop Global Search 响应为统一格式"""
        results = []

        # 检查接口层错误
        metadata = data.get("ResponseMetadata", {})
        if metadata.get("Error"):
            logger.warning(f"豆包搜索接口错误: {metadata['Error']}")
            return results

        result = data.get("Result")
        if not result:
            return results

        if result.get("ErrorCode", 0) != 0:
            logger.warning(f"豆包搜索业务错误 [{result.get('ErrorCode')}]: {result.get('ErrorMsg')}")
            return results

        documents = result.get("Documents", [])
        for doc in documents:
            title = doc.get("Title", "")
            url = doc.get("Url", "")
            # 提取所有文本片段
            content_parts = []
            for snippet in doc.get("Snippet", []):
                if snippet.get("Type") == "text":
                    text = snippet.get("Text", "")
                    if text:
                        content_parts.append(text)
            content = "\n".join(content_parts)

            # 来源 = 站点名
            host_info = doc.get("HostInfo", {})
            source = host_info.get("Hostname", "Doubao")

            # 发布时间
            doc_info = doc.get("DocumentInfo", {})
            publish_time = doc_info.get("PublishTime", "")

            if title or content:
                results.append({
                    "title": title,
                    "url": url,
                    "content": content,
                    "source": source,
                    "publish_time": publish_time,
                })

        return results


def init_doubao_search_service(config: Dict[str, Any] = None) -> Optional[DoubaoSearchService]:
    """初始化豆包搜索服务"""
    global _global_doubao_service
    api_key = os.environ.get("DOUBAO_WEB_SEARCH_API_KEY", "")
    if not api_key:
        logger.info("豆包搜索: DOUBAO_WEB_SEARCH_API_KEY 未配置，跳过")
        _global_doubao_service = None
        return None

    _global_doubao_service = DoubaoSearchService(
        api_key=api_key,
        api_base=DoubaoSearchService.BASE_URL,
        timeout=int(os.environ.get("DOUBAO_WEB_SEARCH_TIMEOUT", "30")),
        max_results=int(os.environ.get("DOUBAO_WEB_SEARCH_MAX_RESULTS", "10")),
        max_snippet_length=int(os.environ.get("DOUBAO_WEB_SEARCH_MAX_SNIPPET_LENGTH", "500")),
    )
    logger.info("豆包搜索服务已初始化")
    return _global_doubao_service


def get_doubao_search_service() -> Optional[DoubaoSearchService]:
    """获取豆包搜索服务实例"""
    return _global_doubao_service


class DoubaoImageSearchService:
    """
    豆包搜图服务 — 通过 FeedCoop Global Search API (SearchType=image) 获取图片
    
    独立于文本搜索，使用独立的 DOUBAO_IMAGE_SEARCH_API_KEY。
    """

    BASE_URL = "https://open.feedcoopapi.com"

    def __init__(self, api_key: str = "", api_base: str = "", timeout: int = 30):
        self.api_key = (
             api_key
             or os.environ.get("DOUBAO_IMAGE_SEARCH_API_KEY", "")
        ) 
        self.api_base = (api_base or self.BASE_URL).rstrip("/")
        self.timeout = int(os.environ.get("DOUBAO_IMAGE_SEARCH_TIMEOUT", str(timeout)))

    def is_available(self) -> bool:
        return bool(self.api_key)

    def search_images(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        豆包搜图 — SearchType=image，返回图片 URL 列表

        Args:
            query: 搜索关键词
            max_results: 返回图片数量，默认 1，最大 20

        Returns:
            {
                "success": bool,
                "images": [{"url": str, "title": str, "source_url": str, "source": str}, ...],
                "error": Optional[str]
            }
        """
        if not self.api_key:
            return {
                "success": False, "images": [],
                "error": "豆包搜图 API Key 未配置（请设置 DOUBAO_IMAGE_SEARCH_API_KEY）",
            }

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "Query": query,
                "SearchType": "image",
                "DocCount": min(max_results, 20),
            }

            url = f"{self.api_base}/search_api/global_search"
            logger.info(f"🖼️ 使用豆包搜图: {query}")
            response = requests.post(
                url, json=payload, headers=headers, timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            images = self._parse_image_results(data)

            logger.info(f"豆包搜图完成: {len(images)} 张图片")
            return {
                "success": True,
                "images": images,
                "error": None,
            }

        except Exception as e:
            logger.error(f"豆包搜图失败: {e}")
            return {
                "success": False, "images": [],
                "error": f"豆包搜图失败: {str(e)}",
            }

    def _parse_image_results(self, data: Dict) -> List[Dict[str, str]]:
        """解析 FeedCoop Global Search image 响应为图片列表"""
        images = []

        metadata = data.get("ResponseMetadata", {})
        if metadata.get("Error"):
            logger.warning(f"豆包搜图接口错误: {metadata['Error']}")
            return images

        result = data.get("Result")
        if not result:
            return images

        if result.get("ErrorCode", 0) != 0:
            logger.warning(f"豆包搜图业务错误 [{result.get('ErrorCode')}]: {result.get('ErrorMsg')}")
            return images

        documents = result.get("Documents", [])
        for doc in documents:
            title = doc.get("Title", "")
            source_url = doc.get("Url", "")
            host_info = doc.get("HostInfo", {})
            source = host_info.get("Hostname", "")

            # 图片 URL 嵌在 Snippet[] 中 Type=image 的条目里
            for snippet in doc.get("Snippet", []):
                if snippet.get("Type") == "image":
                    img_data = snippet.get("Image", {})
                    img_url = img_data.get("ImageUrl", "") if isinstance(img_data, dict) else ""
                    if img_url:
                        images.append({
                            "url": img_url,
                            "title": title,
                            "source_url": source_url,
                            "source": source,
                        })

        return images


def init_doubao_image_search_service(config: Dict[str, Any] = None) -> Optional[DoubaoImageSearchService]:
    """
    初始化豆包搜图服务（独立实例）
    
    使用 DOUBAO_IMAGE_SEARCH_API_KEY，与文本搜索的 key 独立。
    """
    global _global_doubao_image_service
    api_key = os.environ.get("DOUBAO_IMAGE_SEARCH_API_KEY", "")
    if not api_key:
        logger.info("豆包搜图: DOUBAO_IMAGE_SEARCH_API_KEY 未配置，跳过")
        _global_doubao_image_service = None
        return None

    _global_doubao_image_service = DoubaoImageSearchService(
        api_key=api_key,
        api_base=DoubaoImageSearchService.BASE_URL,
        timeout=int(os.environ.get("DOUBAO_IMAGE_SEARCH_TIMEOUT", "30")),
    )
    logger.info("豆包搜图服务已初始化")
    return _global_doubao_image_service


def get_doubao_image_search_service() -> Optional[DoubaoImageSearchService]:
    """获取豆包搜图服务实例"""
    return _global_doubao_image_service