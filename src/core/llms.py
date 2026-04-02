"""
版权所有 (c) 2023-2026 北京慧测信息技术有限公司(但问智能) 保留所有权利。

本代码版权归北京慧测信息技术有限公司(但问智能)所有，仅用于学习交流目的，未经公司商业授权，
不得用于任何商业用途，包括但不限于商业环境部署、售卖或以任何形式进行商业获利。违者必究。

授权商业应用请联系微信：huice666
"""

import logging

from core.config import settings
# 配置日志
logger = logging.getLogger(__name__)


# 创建图片处理模型
def create_image_model():
    from langchain_openai import ChatOpenAI

    """创建图片处理模型"""
    try:
        return ChatOpenAI(
            base_url=settings.IMAGE_PARSER_API_BASE,
            api_key=settings.IMAGE_PARSER_API_KEY,
            model=settings.IMAGE_PARSER_MODEL,
        )
    except Exception as e:
        logger.error(f"Failed to create image model: {e}")
        return None

# 创建文本处理模型
def create_text_model():
    """创建文本处理模型"""
    from langchain_deepseek import ChatDeepSeek
    try:
        return ChatDeepSeek(
            api_key=settings.DEEPSEEK_API_KEY,
            model=settings.LLM_MODEL,
            temperature=0.3
        )
    except ImportError:
        logger.warning("langchain_deepseek not available")
        return None
    except Exception as e:
        logger.error(f"Failed to create text model: {e}")
        return None

image_llm_model = create_image_model()
deepseek_model = create_text_model()
