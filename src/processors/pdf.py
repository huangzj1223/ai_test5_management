# PDF 处理器
"""
PDF 文档处理器，支持文本提取和缓存
"""
from core.config import settings

"""
版权所有 (c) 2023-2026 北京慧测信息技术有限公司(但问智能) 保留所有权利。

本代码版权归北京慧测信息技术有限公司(但问智能)所有，仅用于学习交流目的，未经公司商业授权，
不得用于任何商业用途，包括但不限于商业环境部署、售卖或以任何形式进行商业获利。违者必究。

授权商业应用请联系微信：huice666
"""

# uv pip install langchain-community langchain-pymupdf4llm

import tempfile
import os
import logging
import hashlib
import time
from typing import Optional

from langchain_community.document_loaders.parsers import LLMImageBlobParser
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from core.llms import image_llm_model

logger = logging.getLogger(__name__)

# PDF 内容缓存，避免重复解析同一个文件
_pdf_cache = {}


def _safe_delete_temp_file(file_path: str, max_retries: int = 3, delay: float = 0.1):
    """
    安全删除临时文件，处理Windows文件锁定问题

    Args:
        file_path: 要删除的文件路径
        max_retries: 最大重试次数
        delay: 重试间隔（秒）
    """
    if not os.path.exists(file_path):
        return

    for attempt in range(max_retries):
        try:
            os.unlink(file_path)
            logger.debug(f"临时文件已删除: {file_path}")
            return
        except PermissionError as e:
            if attempt < max_retries - 1:
                logger.debug(f"删除临时文件失败（尝试 {attempt + 1}/{max_retries}），等待后重试: {e}")
                time.sleep(delay)
            else:
                logger.warning(f"无法删除临时文件（已重试{max_retries}次），文件将由系统清理: {file_path}")
        except Exception as e:
            logger.warning(f"删除临时文件时发生异常: {e}")
            break


class PDFProcessor:
    """PDF 处理器类"""

    def __init__(self, enable_cache: bool = True):
        self.enable_cache = enable_cache
        self.cache = _pdf_cache if enable_cache else {}

    def extract_text(
            self,
            pdf_data: bytes,
            filename: str = "unknown.pdf",
            enable_multimodal: bool | None = None,
    ) -> str:
        """从PDF字节数据中提取文本

        Args:
            pdf_data: PDF 原始字节。
            filename: 文件名（用于日志/缓存key）。
            enable_multimodal: 是否启用多模态解析。
                None  → 由 settings.ENABLE_PDF_MULTIMODAL 决定（默认行为）。
                True  → 强制使用豆包多模态模型解析图片。
                False → 强制只做文本提取（不调用视觉模型）。
        """
        return extract_pdf_text(
            pdf_data,
            filename,
            self.cache if self.enable_cache else None,
            enable_multimodal=enable_multimodal,
        )

    def clear_cache(self):
        """清空缓存"""
        if self.enable_cache:
            self.cache.clear()

    def get_cache_stats(self) -> dict:
        """获取缓存统计信息"""
        return {
            "cache_enabled": self.enable_cache,
            "cached_files": len(self.cache) if self.enable_cache else 0,
            "cache_keys": list(self.cache.keys()) if self.enable_cache else []
        }


def extract_pdf_text(
        pdf_data: bytes,
        filename: str = "unknown.pdf",
        cache: Optional[dict] = None,
        enable_multimodal: Optional[bool] = None,
) -> str:
    """
    从PDF字节数据中提取文本，使用缓存避免重复解析

    Args:
        pdf_data: PDF 原始字节。
        filename: 文件名（用于日志/缓存key）。
        cache: 可选的缓存字典。
        enable_multimodal: 是否启用多模态解析。
            None  → 由 settings.ENABLE_PDF_MULTIMODAL 决定。
            True  → 强制多模态（豆包视觉模型）。
            False → 强制纯文本提取。
    提取的方法：
    1、langchain pdf加载器：https://docs.langchain.com/oss/python/integrations/document_loaders/index#pdfs
        推荐 pip install -qU langchain-community langchain-pymupdf4llm，支持基于多模态大模型进行图片解析
    2、DeepSeek ocr大模型
    3、PaddleOCR VL 0.9B（推荐）--部署需要GPU
        推荐 https://www.paddleocr.ai/latest/version3.x/pipeline_usage/PaddleOCR-VL.html

    """
    # 完全由调用方控制：True=豆包多模态，False/None=纯文本提取（不读 .env）
    use_multimodal = enable_multimodal is True
    # 生成PDF数据的哈希值作为缓存键
    pdf_hash = hashlib.md5(pdf_data).hexdigest()
    cache_key = f"{filename}_{pdf_hash}"

    # 检查缓存
    if cache is not None and cache_key in cache:
        logger.info(f"从缓存中获取内容: {filename}")
        return cache[cache_key]

    # 获取文件扩展名，默认为 '.pdf'
    _, ext = os.path.splitext(filename)
    if not ext:
        ext = '.pdf'

    # 创建临时文件（Windows需要先关闭文件句柄才能被其他程序访问）
    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    try:
        temp_file.write(pdf_data)
        temp_file.flush()  # 确保数据写入磁盘
        os.fsync(temp_file.fileno())  # 强制同步到磁盘
        temp_file_path = temp_file.name
    finally:
        temp_file.close()  # 显式关闭文件句柄，释放文件锁

    if ext.lower() in [".doc", ".docx"]:
        try:
            logger.info(f"使用 python-docx 解析Word文档: {filename}")
            import docx
            doc = docx.Document(temp_file_path)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            for table in doc.tables:
                for row in table.rows:
                    row_data = [cell.text for cell in row.cells]
                    full_text.append(" | ".join(row_data))
            text_content = "\n".join(full_text)

            if not text_content.strip():
                text_content = "Word文件解析后文本为空"
            else:
                logger.info(f"Word 解析成功，内容长度: {len(text_content)} 字符")

            # 缓存结果
            if cache is not None:
                cache[cache_key] = text_content
                logger.info(f"内容已缓存: {filename}")

            return text_content
        except Exception as e:
            logger.warning(f"Word 解析失败: {e}")
            if ext.lower() == ".doc":
                text_content = f"不支持老旧的 .doc 格式或解析失败，请在本地转存为 .docx 格式后再上传重试。（错误信息: {str(e)}）"
            else:
                text_content = f"Word文件处理出错: {str(e)}"

            # 缓存结果
            if cache is not None:
                cache[cache_key] = text_content
            return text_content
        finally:
            _safe_delete_temp_file(temp_file_path)

    try:
        logger.info(f"使用 多模态解析器 解析文件: {filename}，多模态={use_multimodal}")
        if use_multimodal:
            # 创建多模态图片解析器
            image_parser = LLMImageBlobParser(
                model=image_llm_model,
                prompt=settings.IMAGE_PARSER_PROMPT
            )

            # 使用PyMuPDF4LLM加载PDF，启用图片提取和多模态解析
            loader = PyMuPDF4LLMLoader(
                temp_file_path,
                mode="single",  # 作为单个文档处理 page
                extract_images=True,  # 提取图片
                images_parser=image_parser,  # 使用多模态模型解析图片
                table_strategy="lines"  # 提取表格
            )
        else:
            loader = PyMuPDF4LLMLoader(
                temp_file_path,
                mode="single",  # 作为单个文档处理
                table_strategy="lines"  # 提取表格
            )
        documents = loader.load()

        if documents:
            text_content = documents[0].page_content
            logger.info(f"PyMuPDF4LLM 解析成功，内容长度: {len(text_content)} 字符")
        else:
            text_content = "PDF文件解析后内容为空"

        # 缓存结果
        if cache is not None:
            cache[cache_key] = text_content
            logger.info(f"PDF内容已缓存: {filename}")

        return text_content
    except Exception as e:
        logger.warning(f"PyMuPDF4LLM 解析失败，尝试备用方法: {e}")
        logger.error(f"PDF文本提取失败: {e}")
        return f"PDF文件处理出错: {str(e)}"
    finally:
        # 清理临时文件（Windows上可能需要重试）
        _safe_delete_temp_file(temp_file_path)

