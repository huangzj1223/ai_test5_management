from typing import Any

from dotenv import load_dotenv
from langchain.agents import AgentState
from langchain.agents.middleware import before_model
from langchain.chat_models import init_chat_model
from langgraph.runtime import Runtime
from deepagents import create_deep_agent as create_agent
load_dotenv()
llm = init_chat_model("deepseek:deepseek-chat")

@before_model()
def check_message(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(state)
    return None


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
agent = create_agent(
    model=llm,                    # 使用 DeepSeek 模型
    tools=[get_weather],          # 注册天气查询工具
    middleware=[check_message],
    system_prompt="You are a helpful assistant",  # 基础系统提示词
)