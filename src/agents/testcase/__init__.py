"""
Testcase Agent Package

本包是一个基于 LangChain 和 LangGraph 构建的测试用例生成智能体，
用于自动分析需求文档并生成专业、全面的测试用例。

主要功能：
    - 通过 MCP (Model Context Protocol) 连接 Docling 文档解析服务
    - 解析各类需求文档（PDF、Word、Markdown 等）
    - 基于 LLM 生成结构化的测试用例

依赖说明：
    - langchain: LLM 应用开发框架
    - langchain-mcp-adapters: MCP 协议适配器
    - langchain-deepseek: DeepSeek 模型集成
    - docling-mcp: 文档解析 MCP 服务

使用示例：
    >>> from agents.testcase.agent import agent
    >>> result = agent.invoke({
    ...     "messages": [{"role": "user", "content": "请分析这份需求文档并生成测试用例"}]
    ... })

Author: AI Assistant
Date: 2026-03-14
"""

# 包版本号
__version__ = "0.1.0"

# 包级别的导出接口
# from .agent import agent
# from .tools import tools
