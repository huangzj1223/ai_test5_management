from pathlib import Path

from deepagents import create_deep_agent as create_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, LocalShellBackend
from deepagents.middleware import SkillsMiddleware
from dotenv import load_dotenv
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.chat_models import init_chat_model

from core.llms import image_llm_model, deepseek_model
load_dotenv()


# ============================================================================
# 大语言模型配置
# ============================================================================
llm = init_chat_model("deepseek:deepseek-chat")

# ============================================================================
# 系统提示词（企业级重构版）
# 角色定位：资深测试架构师 + 智能体行为规范 + Skills激活协议
# ============================================================================
SYSTEM_PROMPT = """
你是一位基于 Microsoft Playwright 的 UI 自动化测试专家 Agent。

## 核心职责
- 分析用户的 UI 测试需求
- 调用 playwright-cli 技能完成浏览器自动化任务
- 生成可复用的 Playwright 测试代码

## 技能使用指南

你拥有 `playwright-cli` 技能：

| 参考文档 | 用途 |
|---------|------|
| `SKILL.md` | 核心命令参考和快速入门 |
| `references/test-generation.md` | 测试代码生成最佳实践 |
| `references/tracing.md` | 调试追踪和日志记录 |
| `references/session-management.md` | 浏览器会话管理和隔离 |
| `references/storage-state.md` | Cookie/LocalStorage 操作 |
| `references/request-mocking.md` | 网络请求拦截和 Mock |
| `references/video-recording.md` | 视频录制功能 |
| `references/running-code.md` | 自定义 Playwright 代码执行 |

## 工作原则

1. **先查阅技能文档** - 在执行任务前，根据场景阅读对应的参考文档
2. **优先语义化定位** - 使用 `getByRole`, `getByLabel` 等可访问性定位器
3. **保持测试隔离** - 使用命名会话 (`-s=`) 隔离不同场景
4. **及时验证状态** - 关键操作后使用 `snapshot` 确认页面状态

## 典型任务流程

1. **理解需求** → 确定测试目标和场景
2. **查阅技能** → 阅读相关参考文档获取命令细节
3. **执行操作** → 使用 `playwright-cli` 命令与页面交互
4. **收集代码** → 提取生成的 TypeScript 测试代码
5. **交付结果** → 输出完整的测试用例和说明
"""

skills_root = Path(r"C:\Users\65132\Desktop\workspace\testing\ai-test-agent-system\src\workspace\ui").resolve()

# 创建复合后端：
# - LocalShellBackend: 执行 shell 命令（如 playwright-cli）
# - FilesystemBackend: 访问技能文档
# 配置 PATH 环境变量，确保能找到 playwright-cli (位于 npm 全局安装目录)
# inherit_env=True 继承系统环境变量，否则 Windows 上 .cmd 文件无法执行
shell_backend = LocalShellBackend(
    inherit_env=True,
    env={"PATH": r"C:\Program Files\nodejs;C:\Users\65132\AppData\Roaming\npm;C:\Windows\System32;C:\Windows"}
)
skills_backend = FilesystemBackend(root_dir=skills_root, virtual_mode=True)

composite_backend = CompositeBackend(
    default=shell_backend,  # 默认使用 shell 执行命令
    routes={
        "/": skills_backend,  # /skills/ 路径访问文件系统
    }
)

# 创建技能中间件
skills_middleware = SkillsMiddleware(
    backend=skills_backend,
    sources=["/skills/"]
)

agent = create_agent(
    model=llm,
    tools=[],
    backend=composite_backend,
    middleware=[skills_middleware],
    system_prompt=SYSTEM_PROMPT
)