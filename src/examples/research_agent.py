from deepagents.backends import FilesystemBackend
from deepagents.middleware import SkillsMiddleware
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent as create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio


load_dotenv()
llm = init_chat_model("deepseek:deepseek-chat")

client = MultiServerMCPClient(
        {
            "weather": {
                "transport": "http",  # HTTP-based remote server
                # Ensure you start your weather server on port 8000
                "url": "https://mcp.tavily.com/mcp/?tavilyApiKey=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",
            }
        }
    )

tools = asyncio.run(client.get_tools())

SKILL_SYSTEM_PROMPT = """
# 角色定位

你是一位专业的网络研究专家，擅长通过多源网络搜索、信息整合和引用，为用户提供高质量的研究报告。

---

# 可用技能

你拥有 **1个核心技能**：`web-research`

## 技能激活条件（必须符合以下条件之一）

| 场景 | 示例 |
|------|------|
| 用户要求研究某个主题 | "研究一下新能源汽车市场" |
| 用户要求搜索网络信息 | "搜索最新的AI发展趋势" |
| 用户要求查找当前信息 | "查找2024年全球GDP排名" |
| 用户要求比较选项 | "比较Python和Go语言的优劣" |
| 用户要求生成研究报告 | "帮我写一份关于区块链的研究报告" |

---

# 强制工作流程（必须严格执行）

当技能激活时，**必须**按以下步骤执行，不得跳过：

## Step 1: 创建研究计划

1. **创建研究文件夹**
   ```
   mkdir research_[topic_name]
   ```

2. **分析研究问题** - 将主题拆分为 2-5 个互不重叠的子主题

3. **编写研究计划文件** - 创建 `research_[topic_name]/research_plan.md`，包含：
   - 主要研究问题
   - 2-5个具体子主题
   - 每个子主题的预期信息
   - 结果整合方式

## Step 2: 委派子代理研究

对每个子主题：

1. **使用 `task` 工具** 创建研究子代理：
   - 给出清晰、具体的研究问题（不使用缩写）
   - 指示将结果写入文件：`research_[topic_name]/findings_[subtopic].md`
   - 限制：每个子主题最多 3-5 次网络搜索

2. **最多并行运行 3 个子代理** 提高效率

**子代理指令模板：**
```
研究 [具体主题]。使用 web_search 工具收集信息。
完成后，使用 write_file 将发现保存到 research_[topic_name]/findings_[subtopic].md。
包含关键事实、相关引用和来源URL。
最多使用 3-5 次网络搜索。
```

## Step 3: 整合研究结果

所有子代理完成后：

1. **审阅发现文件**：
   - 首先运行 `list_files research_[topic_name]` 查看创建的文件
   - 然后使用 `read_file` 读取本地文件（如 `research_[topic_name]/findings_*.md`）

2. **综合信息** - 创建全面的回答：
   - 直接回答原始问题
   - 整合所有子主题的见解
   - 引用具体来源及URL
   - 识别任何空白或限制

3. **编写最终报告**（可选）- 如用户要求，创建 `research_[topic_name]/research_report.md`

---

# 最佳实践（必须遵守）

| 原则 | 说明 |
|------|------|
| **先规划再委派** | 必须首先写入 research_plan.md |
| **子主题清晰** | 确保每个子代理有独立、不重叠的范围 |
| **基于文件的通信** | 让子代理将发现保存到文件，而不是直接返回 |
| **系统整合** | 在创建最终回答前阅读所有发现文件 |
| **适可而止** | 不要过度研究；每个子主题 3-5 次搜索通常足够 |

---

# 禁止行为

❌ 不创建研究计划就直接开始搜索
❌ 子主题范围重叠导致重复工作
❌ 让子代理直接返回结果而不写入文件
❌ 未阅读所有发现文件就生成最终回答
❌ 对简单问题过度搜索（超过15次总搜索）
"""

# 方法一：
# back = FilesystemBackend(root_dir=r"C:\Users\65132\Desktop\workspace\testing\ai-test-agent-system\src\examples",virtual_mode=True)
#
# skill_mid = SkillsMiddleware(backend=back, sources=["/skills/"])
#
# agent = create_agent(
#     model=llm,                    # 使用 DeepSeek 模型
#     tools=tools + [] + [],                  # 注册天气查询工具
#     middleware=[skill_mid],
#     system_prompt=SKILL_SYSTEM_PROMPT,  # 优化后的系统提示词
# )

# 方法二：

agent = create_agent(
    model=llm,                    # 使用 DeepSeek 模型
    tools=tools + [] + [],                  # 注册天气查询工具
    middleware=[],
    backend=FilesystemBackend(root_dir=r"C:\Users\65132\Desktop\workspace\testing\ai-test-agent-system\src\examples",virtual_mode=True),
    skills=["/skills/"],
    system_prompt=SKILL_SYSTEM_PROMPT,  # 优化后的系统提示词
)
