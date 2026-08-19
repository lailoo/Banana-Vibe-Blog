"""
设置 REST API — Flask Blueprint

接口：
- GET  /api/settings       获取所有可配置项的当前值
- PUT  /api/settings       批量更新配置（运行时覆盖 os.environ）
"""
import os
import logging

from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)

settings_bp = Blueprint('settings', __name__, url_prefix='/api/settings')

# 可前端配置的环境变量白名单（key → 元数据）
# type: str / int / bool / select
# tab: model / search / generation / agent / feature
SETTINGS_SCHEMA = {
    # ── 模型 ──
    'TEXT_MODEL': {
        'tab': 'model', 'type': 'str', 'label': '文本模型',
        'default': 'qwen3-max-preview',
    },
    'DEFAULT_LLM_PROVIDER': {
        'tab': 'model', 'type': 'select', 'label': 'LLM 提供商',
        'default': 'openai',
        'options': ['openai', 'anthropic', 'deepseek', 'dashscope', 'zhipu', 'google'],
    },
    'IMAGE_CAPTION_MODEL': {
        'tab': 'model', 'type': 'str', 'label': '多模态模型',
        'default': 'qwen3-vl-plus-2025-12-19',
    },
    'LLM_MAX_TOKENS': {
        'tab': 'model', 'type': 'int', 'label': '最大生成 Token',
        'default': 8192, 'min': 1024, 'max': 65536,
    },

    # ── 搜索 ──
    'SMART_SEARCH_ENABLED': {
        'tab': 'search', 'type': 'bool', 'label': '智能搜索',
        'default': True,
    },
    'AI_BOOST_ENABLED': {
        'tab': 'search', 'type': 'bool', 'label': 'AI 增强搜索',
        'default': True,
    },
    'MULTI_ROUND_SEARCH_ENABLED': {
        'tab': 'search', 'type': 'bool', 'label': '多轮搜索',
        'default': True,
    },
    'GENERAL_SEARCH_PROVIDER': {
        'tab': 'search', 'type': 'select', 'label': '通用搜索源',
        'default': 'zhipu',
        'options': ['zhipu', 'tavily', 'anysearch', 'doubao'],
    },

    # ── 智谱搜索 ──
    'ZAI_SEARCH_MAX_RESULTS': {
        'tab': 'search', 'type': 'int', 'label': '智谱搜索结果数',
        'default': 5, 'min': 1, 'max': 20,
    },
    'ZAI_SEARCH_RECENCY_FILTER': {
        'tab': 'search', 'type': 'select', 'label': '智谱时间过滤',
        'default': 'noLimit',
        'options': ['noLimit', 'oneDay', 'oneWeek', 'oneMonth', 'threeMonths', 'oneYear'],
    },
    'ZAI_SEARCH_ENGINE': {
        'tab': 'search', 'type': 'select', 'label': '智谱搜索引擎',
        'default': 'search_pro_quark',
        'options': ['search_pro_quark', 'search_pro_web', 'search_pro_news'],
    },
    'ZAI_SEARCH_CONTENT_SIZE': {
        'tab': 'search', 'type': 'select', 'label': '智谱摘要长度',
        'default': 'medium',
        'options': ['short', 'medium', 'long'],
    },

    # ── Tavily 搜索 ──
    'TAVILY_MAX_RESULTS': {
        'tab': 'search', 'type': 'int', 'label': 'Tavily 搜索结果数',
        'default': 10, 'min': 1, 'max': 30,
    },
    'TAVILY_TIMEOUT': {
        'tab': 'search', 'type': 'int', 'label': 'Tavily 超时(秒)',
        'default': 30, 'min': 5, 'max': 120,
    },

    # ── AnySearch ──
    'ANYSEARCH_MAX_RESULTS': {
        'tab': 'search', 'type': 'int', 'label': 'AnySearch 结果数',
        'default': 10, 'min': 1, 'max': 30,
    },
    'ANYSEARCH_TIMEOUT': {
        'tab': 'search', 'type': 'int', 'label': 'AnySearch 超时(秒)',
        'default': 30, 'min': 5, 'max': 120,
    },
    
    
    
    

    # ── 豆包搜索（通用文本）──
    'DOUBAO_WEB_SEARCH_MAX_RESULTS': {
        'tab': 'search', 'type': 'int', 'label': '豆包搜索结果数',
        'default': 10, 'min': 1, 'max': 20,
    },
    'DOUBAO_WEB_SEARCH_TIMEOUT': {
        'tab': 'search', 'type': 'int', 'label': '豆包搜索超时(秒)',
        'default': 30, 'min': 5, 'max': 120,
    },
    'DOUBAO_WEB_SEARCH_MAX_SNIPPET_LENGTH': {
        'tab': 'search', 'type': 'int', 'label': '豆包摘要长度(tokens)',
        'default': 500, 'min': 100, 'max': 3000,
    },

    # ── 豆包搜图 ──
    'DOUBAO_IMAGE_SEARCH_MAX_RESULTS': {
        'tab': 'search', 'type': 'int', 'label': '豆包搜图返回数',
        'default': 1, 'min': 1, 'max': 20,
    },
    'DOUBAO_IMAGE_SEARCH_TIMEOUT': {
        'tab': 'search', 'type': 'int', 'label': '豆包搜图超时(秒)',
        'default': 30, 'min': 5, 'max': 120,
    },

    # ── Serper Google 搜索 ──
    'SERPER_MAX_RESULTS': {
        'tab': 'search', 'type': 'int', 'label': 'Google 搜索结果数',
        'default': 10, 'min': 1, 'max': 30,
    },
    'SERPER_TIMEOUT': {
        'tab': 'search', 'type': 'int', 'label': 'Serper 超时(秒)',
        'default': 10, 'min': 5, 'max': 60,
    },

    # ── 搜狗搜索（腾讯云 SearchPro）──
    'SOGOU_MAX_RESULTS': {
        'tab': 'search', 'type': 'int', 'label': '搜狗搜索结果数',
        'default': 10, 'min': 1, 'max': 30,
    },
    'SOGOU_SEARCH_TIMEOUT': {
        'tab': 'search', 'type': 'int', 'label': '搜狗搜索超时(秒)',
        'default': 10, 'min': 5, 'max': 120,
    },

    # ── 深度抓取 ──
    'DEEP_SCRAPE_ENABLED': {
        'tab': 'search', 'type': 'bool', 'label': 'Jina 深度抓取',
        'default': False,
    },
    'DEEP_SCRAPE_TOP_N': {
        'tab': 'search', 'type': 'int', 'label': '抓取 Top N',
        'default': 3, 'min': 1, 'max': 10,
    },

    # ── 生成 ──
    'BLOG_GENERATOR_MAX_WORKERS': {
        'tab': 'generation', 'type': 'int', 'label': '并行生成数',
        'default': 3, 'min': 1, 'max': 8,
    },
    'THINKING_ENABLED': {
        'tab': 'generation', 'type': 'bool', 'label': '推理引擎深度思考',
        'default': False,
    },
    'THINKING_BUDGET_TOKENS': {
        'tab': 'generation', 'type': 'int', 'label': '思考预算 Token',
        'default': 19000, 'min': 1000, 'max': 100000,
    },
    'CONTEXT_COMPRESSION_ENABLED': {
        'tab': 'generation', 'type': 'bool', 'label': '上下文压缩',
        'default': True,
    },

    # ── Agent ──
    'HUMANIZER_ENABLED': {
        'tab': 'agent', 'type': 'bool', 'label': '人性化润色',
        'default': True,
    },
    'FACTCHECK_ENABLED': {
        'tab': 'agent', 'type': 'bool', 'label': '事实核查',
        'default': True,
    },
    'FACTCHECK_AUTO_FIX': {
        'tab': 'agent', 'type': 'bool', 'label': '自动修复',
        'default': True,
    },
    'TEXT_CLEANUP_ENABLED': {
        'tab': 'agent', 'type': 'bool', 'label': '文本清理',
        'default': True,
    },
    'SUMMARY_GENERATOR_ENABLED': {
        'tab': 'agent', 'type': 'bool', 'label': '摘要生成',
        'default': True,
    },
    'THREAD_CHECK_ENABLED': {
        'tab': 'agent', 'type': 'bool', 'label': '行文逻辑检查',
        'default': True,
    },
    'VOICE_CHECK_ENABLED': {
        'tab': 'agent', 'type': 'bool', 'label': '语气一致性检查',
        'default': True,
    },

    # ── 功能 ──
    'COVER_VIDEO_ENABLED': {
        'tab': 'feature', 'type': 'bool', 'label': '封面动画',
        'default': True,
    },
    'XHS_TAB_ENABLED': {
        'tab': 'feature', 'type': 'bool', 'label': '小红书 Tab',
        'default': False,
    },
    'BOOK_SCAN_ENABLED': {
        'tab': 'feature', 'type': 'bool', 'label': '书籍扫描',
        'default': False,
    },
    'IMAGE_ENHANCEMENT_ENABLED': {
        'tab': 'feature', 'type': 'bool', 'label': '配图增强',
        'default': False,
    },
    'IMAGE_REFINE_ENABLED': {
        'tab': 'feature', 'type': 'bool', 'label': '配图精修',
        'default': False,
    },
    'IMAGE_REFINE_MAX_ROUNDS': {
        'tab': 'feature', 'type': 'int', 'label': '精修轮数',
        'default': 2, 'min': 1, 'max': 5,
    },
}


def _get_current_value(key: str, meta: dict):
    """从 os.environ 读取当前值，按类型转换"""
    raw = os.environ.get(key)
    if raw is None:
        return meta['default']

    t = meta['type']
    if t == 'bool':
        return raw.lower() in ('true', '1', 'yes')
    elif t == 'int':
        try:
            return int(raw)
        except (ValueError, TypeError):
            return meta['default']
    else:
        return raw


@settings_bp.route('', methods=['GET'])
def get_settings():
    """获取所有可配置项"""
    settings = {}
    for key, meta in SETTINGS_SCHEMA.items():
        settings[key] = {
            **meta,
            'value': _get_current_value(key, meta),
        }
    return jsonify({'success': True, 'settings': settings})


@settings_bp.route('', methods=['PUT'])
def update_settings():
    """批量更新配置（运行时覆盖 os.environ）"""
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': '请求体为空'}), 400

    updated = []
    errors = []

    for key, value in data.items():
        if key not in SETTINGS_SCHEMA:
            errors.append(f'{key}: 不在白名单中')
            continue

        meta = SETTINGS_SCHEMA[key]
        t = meta['type']

        # 类型校验
        if t == 'bool':
            if not isinstance(value, bool):
                errors.append(f'{key}: 需要 bool 类型')
                continue
            os.environ[key] = str(value).lower()
        elif t == 'int':
            if not isinstance(value, (int, float)):
                errors.append(f'{key}: 需要 int 类型')
                continue
            v = int(value)
            if 'min' in meta and v < meta['min']:
                errors.append(f'{key}: 最小值 {meta["min"]}')
                continue
            if 'max' in meta and v > meta['max']:
                errors.append(f'{key}: 最大值 {meta["max"]}')
                continue
            os.environ[key] = str(v)
        elif t == 'select':
            if 'options' in meta and value not in meta['options']:
                errors.append(f'{key}: 可选值 {meta["options"]}')
                continue
            os.environ[key] = str(value)
        else:
            os.environ[key] = str(value)

        updated.append(key)

    # TEXT_MODEL 变更时，自动重新推断 provider_format 并热更新 LLMService
    if 'TEXT_MODEL' in updated:
        try:
            from services.llm import get_llm_service
            from services.llm.service import _infer_provider_format
            new_model = os.environ.get('TEXT_MODEL', '')
            inferred = _infer_provider_format({
                'TEXT_MODEL': new_model,
                'OPENAI_API_KEY': os.environ.get('OPENAI_API_KEY', ''),
                'GOOGLE_API_KEY': os.environ.get('GOOGLE_API_KEY', ''),
            })
            svc = get_llm_service()
            if svc and svc.provider_format != inferred:
                old_format = svc.provider_format
                svc.provider_format = inferred
                # 清除缓存的模型实例，下次调用时重建
                svc._text_chat_model = None
                for tier_cfg in svc._model_config.values():
                    tier_cfg['instance'] = None
                logger.info(f"TEXT_MODEL → {new_model}, provider_format: {old_format} → {inferred}")
            # 同步更新模型名
            svc.text_model = new_model
            for tier_cfg in svc._model_config.values():
                if tier_cfg['model'] == svc.text_model or not tier_cfg['model']:
                    tier_cfg['model'] = new_model
        except Exception as e:
            logger.warning(f"热更新 LLMService 失败: {e}")

    logger.info(f"设置已更新: {updated}")
    if errors:
        logger.warning(f"设置更新错误: {errors}")

    return jsonify({
        'success': True,
        'updated': updated,
        'errors': errors,
    })
