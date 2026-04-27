"""测试用例Agent的工具定义。

此模块包含所有可用的工具定义，包括：
- 基础工具：导出测试用例到 Excel / Word
- 文档工具：从 PDF 文件路径提取文本
- RAG工具：通过 MCP 客户端获取的检索增强工具
"""

import asyncio
from functools import lru_cache
from typing import Union, Optional

from langchain.tools import tool
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from agents.testcase.excel_exporter import export_test_cases_to_excel
from agents.testcase.docx_exporter import export_test_cases_to_docx
from processors.pdf import extract_pdf_text

# MCP服务器配置
MCP_SERVER_CONFIGS = {
    "rag-server": {
        "url": "http://localhost:8008/sse",
        "transport": "sse",
    }
}


@tool
def export_testcases_to_excel(test_cases: list, output_path: str, sheet_name: str = "测试用例") -> str:
    """
    将测试用例列表导出为 Excel 文件。

    当用户要求导出 Excel 格式、或需要将用例导入禅道/Tapd/TestRail 等工具时调用。
    """
    return export_test_cases_to_excel(test_cases, output_path, sheet_name)


@tool
def export_testcases_to_docx(test_cases: list, output_path: str) -> str:
    """
    将测试用例列表导出为 Word (.docx) 文件。

    当用户要求导出 Word / DOCX 格式的测试用例文档时调用。
    """
    return export_test_cases_to_docx(test_cases, output_path)


@tool
def extract_pdf_text_from_file(file_path: str, enable_multimodal: bool = False) -> str:
    """
    从 PDF 文件路径中提取文本。

    当用户上传 PDF 后，或系统提示需要解析本地 PDF 文件时调用。

    Args:
        file_path: PDF 文件的绝对路径或相对路径。
        enable_multimodal: 是否启用多模态图片解析。

    Returns:
        提取的文本内容。
    """
    import os

    if not os.path.isfile(file_path):
        return f"PDF文件不存在: {file_path}"

    try:
        with open(file_path, "rb") as f:
            pdf_data = f.read()
    except OSError as e:
        return f"读取PDF文件失败: {str(e)}"

    filename = os.path.basename(file_path)
    return extract_pdf_text(
        pdf_data,
        filename=filename,
        enable_multimodal=enable_multimodal,
    )


@lru_cache(maxsize=1)
def _cached_rag_tools() -> tuple[BaseTool, ...]:
    """缓存RAG工具列表，避免重复创建MCP客户端。"""
    client = MultiServerMCPClient(MCP_SERVER_CONFIGS)
    tools = asyncio.run(client.get_tools())
    return tuple(tools)


def rag_mcp_tools() -> list[BaseTool]:
    """获取RAG MCP工具列表。"""
    try:
        return list(_cached_rag_tools())
    except Exception:
        return []


def get_rag_tool_names() -> set[str]:
    """获取RAG工具的名称集合，用于识别和过滤。"""
    return {tool.name for tool in rag_mcp_tools()}


def get_tool_name(tool: Union[BaseTool, dict]) -> str:
    """获取工具名称，支持 BaseTool 对象和字典格式。"""
    if isinstance(tool, dict):
        return tool.get("name", "")
    return getattr(tool, "name", "")


RAG_SYSTEM_PROMPT_APPENDIX = """

---

## 附录：可用 RAG 工具列表

{rag_tools_description}

> 详细的 RAG 检索策略、mode 选择规范、结果引用规范等，请严格遵循 `rag-query` Skill 执行。
"""


def format_rag_tools_description() -> str:
    """格式化RAG工具描述，用于系统提示词。"""
    tools = rag_mcp_tools()
    if not tools:
        return "（暂无RAG工具配置或 RAG MCP 服务未启动）"

    descriptions = []
    for tool_item in tools:
        desc = getattr(tool_item, "description", "无描述")
        descriptions.append(f"- **{tool_item.name}**: {desc}")
    return "\n".join(descriptions)


def get_base_tools() -> list:
    """获取基础工具列表（不包含RAG工具）。"""
    return [
        export_testcases_to_excel,
        export_testcases_to_docx,
        extract_pdf_text_from_file,
    ]


def get_all_tools() -> list:
    """获取所有可用工具的完整列表。"""
    return get_base_tools() + rag_mcp_tools()
