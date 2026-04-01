"""
LangChain 智能体示例模块

本模块提供了一个基础的 LangChain 智能体实现示例，
演示如何创建和使用带有自定义工具的简单智能体。

主要演示内容：
    1. 环境变量加载（.env 文件）
    2. 大语言模型初始化
    3. 自定义工具函数定义
    4. 智能体创建与配置

适用场景：
    - LangChain 初学者入门参考
    - 简单工具调用示例
    - 快速原型验证

依赖说明：
    - langchain: 核心框架
    - langchain-deepseek: DeepSeek 模型集成
    - python-dotenv: 环境变量管理
"""
from typing import Any

import mcp
from dotenv import load_dotenv
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langchain.chat_models import init_chat_model
from langgraph.runtime import Runtime


# 导入 LangChain DeepSeek 聊天模型（备用方案）
# from langchain_deepseek import ChatDeepSeek


# ============================================================================
# 环境初始化
# ============================================================================

# 加载 .env 文件中的环境变量
# 确保项目根目录存在 .env 文件，包含：
#   DEEPSEEK_API_KEY=your_api_key_here
load_dotenv()


# ============================================================================
# 大语言模型配置
# ============================================================================

# 方案1：使用 init_chat_model 统一接口（推荐）
# 优点：支持多模型切换，配置简洁
llm = init_chat_model("deepseek:deepseek-chat")

# 方案2：直接使用 ChatDeepSeek 类（备用）
# llm = ChatDeepSeek(model="deepseek-chat")


# ============================================================================
# LLM 基础调用示例（已注释）
# ============================================================================

# 基础对话示例代码（保留供参考）
# messages = [
#     (
#         "system",
#         "你可以回答关于编程的任何问题",
#     ),
#     ("human", "我喜欢编程"),
# ]
# ai_msg = llm.invoke(messages)
# print(ai_msg.content)


# ============================================================================
# 自定义工具定义
# ============================================================================

def get_weather(city: str) -> str:
    """
    获取指定城市的天气信息
    
    这是一个示例工具函数，演示如何为智能体定义可调用的工具。
    在实际应用中，该函数可以调用真实的天气 API。
    
    Args:
        city (str): 城市名称，如 "北京"、"Shanghai"
        
    Returns:
        str: 该城市的天气情况描述
        
    Example:
        >>> get_weather("北京")
        '北京，晴天'
    """
    return f"{city}，晴天"


# ============================================================================
# 智能体配置与创建
# ============================================================================

# 创建智能体实例
# create_agent 参数说明：
#   - model: 语言模型实例
#   - tools: 工具函数列表，智能体可根据需要调用
#   - system_prompt: 系统提示词，定义智能体的角色和行为


@before_model()
def check_message(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(state)
    return None

agent = create_agent(
    model=llm,                    # 使用 DeepSeek 模型
    tools=[get_weather],          # 注册天气查询工具
    middleware=[check_message],
    system_prompt="You are a helpful assistant",  # 基础系统提示词
)
