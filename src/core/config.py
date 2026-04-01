

import os
import logging
from typing import Any, Dict, List

from dotenv import load_dotenv
from pydantic_settings import BaseSettings
load_dotenv()
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    # LLM配置
    DEEPSEEK_API_KEY: str = ""
    LLM_API_BASE: str = os.getenv("LLM_API_BASE", "https://api.deepseek.com/v1")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "deepseek-chat")

    # 图片解析 LLM 配置
    IMAGE_PARSER_API_BASE: str = os.getenv("IMAGE_PARSER_API_BASE", "https://ark.cn-beijing.volces.com/api/v3")
    IMAGE_PARSER_API_KEY: str = os.getenv("IMAGE_PARSER_API_KEY", "e4ff4ed2-0721-4ac7-8abc-5487d4cbfff2")
    #e4ff4ed2-0721-4ac7-8abc-5487d4cbfff2#我自己的
    IMAGE_PARSER_MODEL: str = os.getenv("IMAGE_PARSER_MODEL", "doubao-seed-1-6-vision-250815")

    # PDF 多模态处理配置
    ENABLE_PDF_MULTIMODAL: bool = os.getenv("ENABLE_PDF_MULTIMODAL", "false").lower() == "true"
    # 图片解析提示词
    IMAGE_PARSER_PROMPT: str = os.getenv("IMAGE_PARSER_PROMPT", """你是一个专业的文档图像解析助手。请将图像内容完整转换为结构化文本：

## 任务要求
1. **文字提取**：完整提取所有可见文本，包括标题、正文、注释、页眉页脚
2. **结构还原**：保留原始层级关系（章节、列表、表格、代码块）
3. **视觉描述**：对图表、流程图、截图等补充必要的视觉说明
4. **语义标注**：标注关键信息（如字段名、按钮标签、错误提示等）

## 输出格式
- 使用标准 Markdown 格式
- 表格用 | 分隔，代码块标注语言类型
- 无需解释性文字，直接输出内容
- 开头和结尾不要添加 ```markdown 标记""").strip()


    # 日志配置
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "agentic_rag.log")
    ENABLE_DETAILED_LOGGING: bool = os.getenv("ENABLE_DETAILED_LOGGING", "true").lower() == "true"

    class Config:
        case_sensitive = True
        env_file = ".env"

    def validate_configuration(self) -> List[str]:
        """
        验证配置的有效性

        Returns:
            配置问题列表，空列表表示配置正常
        """
        issues = []

        # 检查必需的API密钥
        if not self.DEEPSEEK_API_KEY:
            issues.append("DEEPSEEK_API_KEY is required for DeepSeek models")

        if not self.IMAGE_PARSER_API_KEY:
            issues.append("IMAGE_PARSER_API_KEY is required for image processing")

        return issues

    def get_safe_config(self) -> Dict[str, Any]:
        """
        获取安全的配置信息（隐藏敏感信息）

        Returns:
            安全的配置字典
        """
        config = self.model_dump()

        # 隐藏敏感信息
        sensitive_keys = [
            "DEEPSEEK_API_KEY", "IMAGE_PARSER_API_KEY", "SECRET_KEY"
        ]

        for key in sensitive_keys:
            if key in config and config[key]:
                config[key] = "***" + config[key][-4:] if len(config[key]) > 4 else "***"

        return config


def create_settings() -> Settings:
    """创建并验证设置"""
    settings = Settings()

    # 验证配置
    issues = settings.validate_configuration()
    if issues:
        logger.warning("Configuration issues found:")
        for issue in issues:
            logger.warning(f"  - {issue}")

    return settings

# 该代码会自动执行，并且只会执行一次（单例设计模式）
settings = create_settings()

