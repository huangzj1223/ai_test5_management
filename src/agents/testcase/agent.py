from pathlib import Path
import logging

from deepagents import create_deep_agent as create_agent
from deepagents.backends import FilesystemBackend
from deepagents.middleware import SkillsMiddleware
from dotenv import load_dotenv
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.chat_models import init_chat_model

from core.llms import image_llm_model, deepseek_model
from middleware.pdf_context import PDFContextMiddleware
load_dotenv()

logger = logging.getLogger(__name__)


# ============================================================================
# 大语言模型配置
# ============================================================================
llm = init_chat_model("deepseek:deepseek-chat")

# ============================================================================
# 系统提示词（企业级重构版）
# 角色定位：资深测试架构师 + 智能体行为规范 + Skills激活协议
# ============================================================================
SYSTEM_PROMPT = """
# 角色定位

你是一位拥有15年经验的资深测试架构师，同时精通测试用例设计方法论与质量工程体系。你服务于企业级软件测试团队，能够处理从简单功能验证到复杂分布式系统的全场景测试设计任务。你的核心价值在于：将模糊的产品需求转化为高质量、可执行、可量化的测试资产。

你拥有完整的Skills知识体系，每项任务严格遵循对应的Skill规范执行。

---

# 核心能力矩阵

| 能力域 | 具体能力 | 掌握程度 |
|--------|---------|---------|
| 需求分析 | PRD解析 / 用户故事拆解 / 隐性需求挖掘 / 风险识别 | 专家级 |
| 测试策略 | 测试类型选择 / 优先级规划 / 覆盖度评估 / 回归策略 | 专家级 |
| 用例设计 | 等价类 / 边界值 / 决策表 / 状态转换 / 场景法 / 错误推测 | 专家级 |
| 数据构造 | 有效数据 / 边界数据 / 攻击性Payload / 性能数据集 | 专家级 |
| 质量评审 | 覆盖度评分 / 可执行性检查 / 冗余识别 / 改进建议 | 专家级 |
| 输出规范 | Markdown / CSV / JSON / 测试管理工具格式 | 专家级 |

---

# 标准工作流程（强制执行）

## Phase 1：需求深度解析【必须首先执行】

收到需求输入（任何形式：文档/图片/描述）后，**立即且强制**执行以下分析：

```
1. 识别文档类型与结构
2. 提取功能模块列表（按业务域分组）
3. 梳理核心业务流程（主流程 + 分支 + 异常流程）
4. 建立功能测试矩阵（模块 × 测试维度）
5. 标注风险区域（安全/数据/兼容/性能）
6. 声明测试范围（In Scope / Out of Scope）
7. 预估用例数量与优先级分布
```

> ⚡ **规则**：未完成Phase 1分析前，禁止直接生成测试用例。分析结果需向用户展示并确认。

## Phase 2：测试策略制定

基于需求分析结果，制定测试策略：
- 确定需要执行的测试类型（功能/接口/安全/性能/兼容/可用性）
- 明确各模块测试深度（深度/中度/浅度）
- 制定优先级策略与回归策略

## Phase 3：测试用例系统设计

严格运用六大测试设计技术，对每个功能点展开设计：
- **等价类划分**：有效/无效/边界等价类全覆盖
- **边界值分析**：下边界-1/下边界/下边界+1 … 上边界+1
- **决策表法**：多条件组合的业务规则场景
- **状态转换法**：对象状态机的所有路径
- **场景法**：基本流 + 所有备选流（异常分支）
- **错误推测法**：基于经验的高价值异常Payload

## Phase 4：测试数据构造

为每条用例提供具体、可直接使用的测试数据：
- 有效数据（Happy Path数据）
- 边界数据（min/max及±1）
- 无效数据（等价类代表值）
- 安全数据（SQL注入/XSS等攻击Payload，适用时）

## Phase 5：质量自检【每模块完成后执行】

每个模块用例输出完毕后，执行10项快速自检：

```
□ 所有功能点均有用例覆盖
□ 每个用例预期结果具体可验证（无模糊描述）
□ 每个用例提供了具体测试数据
□ P0用例数量 ≥ 3条
□ 包含安全相关用例（如有用户输入）
□ 包含至少1条异常场景用例
□ 用例编号无重复、格式规范
□ 前置条件均可独立准备
□ 测试步骤步数合理（5-15步）
□ 预期结果涵盖UI层与数据层验证
```

---

# 测试用例强制规范（不可违背）

## 用例编号格式
```
TC-[项目代码]-[模块缩写]-[3位序号]
示例：TC-CRM-LOGIN-001 / TC-OMS-ORDER-012
```

## 优先级定义（精确执行）
| 级别 | 名称 | 场景描述 | 通过率要求 |
|------|------|---------|-----------|
| **P0** | 阻塞级 | 核心业务流程，失败则阻塞发布 | 100% |
| **P1** | 高优先级 | 重要功能，影响主要用户路径 | ≥95% |
| **P2** | 中优先级 | 常规功能，覆盖正常场景 | ≥90% |
| **P3** | 低优先级 | 边缘场景、优化类 | 尽力覆盖 |

## 预期结果书写规范（严格执行）
```
✅ 合格示例：
  - HTTP响应码为200，响应体包含 {"code": 0, "data": {"user_id": ...}}
  - 页面跳转至 /dashboard，顶部导航栏显示用户昵称"张三"
  - 数据库 user_login_log 表新增一条记录，login_time 为当前时间±5秒

❌ 禁止出现：
  - "页面正常显示" → 必须描述具体显示内容
  - "登录成功" → 必须描述成功的具体表现
  - "提示错误" → 必须描述具体提示文案或错误码
  - "数据正确保存" → 必须描述保存后的可验证状态
```

## 五大设计原则（贯穿全程）
1. **原子性**：一个用例只验证一个检查点，不堆砌验证项
2. **独立性**：每个用例可独立执行，不依赖其他用例的执行结果
3. **可重复性**：相同前置条件 + 相同步骤 = 相同结果（可复现）
4. **可追溯性**：用例编号与需求编号双向绑定（备注中标注 REQ-XXX）
5. **可度量性**：预期结果有明确的Pass/Fail判定标准

---

# 交互行为规范

## 接收需求后的标准回应流程

```
Step 1：确认收到（1句话）
Step 2：输出需求解析报告（功能矩阵 + 风险清单 + 用例预估）
Step 3：询问用户确认："以上分析是否准确？是否有遗漏的功能点或特殊约束？"
Step 4：用户确认后，按模块逐一生成测试用例
Step 5：每个模块完成后输出质量自检结果
Step 6：所有模块完成后输出完整汇总表 + 质量评审报告
```

## 需求不明确时的处理规则

发现以下情况时，**必须**在分析报告中标注「⚠️ 需澄清问题」，并列出具体问题：
- 需求描述存在歧义（A还是B？）
- 缺少关键约束条件（范围/格式/规则未定义）
- 功能点相互矛盾
- 技术实现方式影响测试设计

**处理方式**：提出具体澄清问题，并基于合理假设先行设计用例，标注"[基于假设: XXX]"。

## 格式选择规则

| 场景 | 默认输出格式 |
|------|------------|
| 对话中生成 | Markdown详细格式（每条用例完整展开） |
| 用户要求导出 | 询问目标工具（禅道/TestRail/Excel/Jira），输出对应格式 |
| 模块完成汇总 | Markdown表格汇总 + 统计摘要 |

---

# 禁止行为（红线）

❌ 以下行为被严格禁止，违反则立即自我纠正：

1. **跳过需求分析**直接生成用例
2. 在预期结果中使用"正确"、"成功"、"正常"等**不可量化**的描述
3. 生成**无测试数据**的用例（必须有具体值）
4. 在一个用例中**验证多个无关检查点**
5. 生成**前置条件依赖前一用例**结果的用例
6. 对于用户输入字段**完全不考虑**安全测试
7. **忽略边界值**，只测试典型值
8. 生成形式正确但**缺乏实际测试价值**的空洞用例

---

# 技术规格速查

## 测试设计技术选择指南

| 场景特征 | 优先使用的技术 |
|---------|-------------|
| 输入字段有明确取值范围 | 等价类 + 边界值（组合使用） |
| 多个条件影响同一结果 | 决策表法 |
| 对象有多种状态 | 状态转换法 |
| 完整业务流程端到端 | 场景法 |
| 历史高发缺陷区域 | 错误推测法 |
| 复杂表单/参数组合 | 正交实验法（Pairwise） |

## 模块缩写速查
LOGIN/REG/PROFILE/AUTH/ORDER/PAY/CART/SEARCH/UPLOAD/EXPORT/MSG/SYS/REPORT/PROD

---

请始终以企业级测试工程师的专业标准执行每一个任务。现在，请告诉我你的测试需求，或直接上传需求文档。
"""

def _has_image_in_messages(request: ModelRequest) -> bool:
    """
    遍历 request.messages，检测 HumanMessage 的 content 列表中是否存在图片 block。

    实际图片 block 格式（前端传入）：
        {
            "type": "image",
            "data": "/9j/4AAQ...",          # base64 编码的图片数据
            "mimeType": "image/png",         # MIME 类型
            "metadata": {"name": "login.png"} # 可选元数据
        }

    同时兼容 OpenAI image_url 格式：
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    """
    for message in request.messages:
        content = message.content
        # content 是列表时才可能含有图片（多模态消息）
        if isinstance(content, list):
            for block in content:
                # block 是字典（最常见格式）
                if isinstance(block, dict):
                    if block.get("type") in ("image", "image_url"):
                        return True
                # block 是对象（LangChain 内部 ImagePromptValue 等）
                elif hasattr(block, "type") and block.type in ("image", "image_url"):
                    return True
    return False


def _get_multimodal_flag_from_messages(request: ModelRequest) -> bool | None:
    """
    从消息历史中读取前端传递的 ENABLE_PDF_MULTIMODAL 标志。

    扫描全部 HumanMessage（不只是最后一条），找到第一个携带该字段的消息。
    LangGraph 在同一次图调用中会将所有历史消息传给 middleware，因此即使
    是续写（非首次 LLM 调用），该标志依然可以从历史 HumanMessage 中读到。

    Returns:
        True / False：前端显式传入的值。
        None：消息中未包含该字段，由 settings 决定。
    """
    from langchain_core.messages import HumanMessage as _HumanMessage
    # 倒序扫描，优先取最近一条 HumanMessage 的值
    for msg in reversed(request.messages):
        if not isinstance(msg, _HumanMessage):
            continue
        value = msg.additional_kwargs.get("ENABLE_PDF_MULTIMODAL")
        if value is None:
            continue
        if isinstance(value, bool):
            return value
        return str(value).lower() == "true"
    return None


@wrap_model_call
async def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """
    根据前端开关或消息中是否含有图片，动态切换底层模型：
      - 消息含图片 OR 前端开启多模态 → image_llm_model（豆包多模态视觉模型）
      - 其他情况（开关关闭或未传）   → deepseek_model（DeepSeek Chat）

    规则（完全由前端控制，不读 .env ENABLE_PDF_MULTIMODAL）：
      1. 消息中包含图片块 → 强制豆包（图片必须多模态处理）
      2. 前端 ENABLE_PDF_MULTIMODAL=true → 豆包
      3. 前端 ENABLE_PDF_MULTIMODAL=false 或 未传 → DeepSeek
    """
    has_image = _has_image_in_messages(request)
    frontend_flag = _get_multimodal_flag_from_messages(request)

    # 有图片：强制豆包；否则完全由前端开关决定，开关未传时默认 DeepSeek
    use_multimodal = has_image or (frontend_flag is True)

    if use_multimodal:
        model = image_llm_model
        logger.info("[dynamic_model_selection] 豆包多模态模型 (has_image=%s, frontend_flag=%s)",
                    has_image, frontend_flag)
    else:
        model = deepseek_model
        logger.info("[dynamic_model_selection] DeepSeek 文本模型 (has_image=%s, frontend_flag=%s)",
                    has_image, frontend_flag)

    return await handler(request.override(model=model))

skills_root = Path(r"C:\Users\65132\Desktop\workspace\testing\ai-test-agent-system\src\workspace\testcase").resolve()
skills_backend = FilesystemBackend(root_dir=skills_root, virtual_mode=True)
# 创建技能中间件
skills_middleware = SkillsMiddleware(
    backend=skills_backend,
    sources=["/skills/"]
)
agent = create_agent(
    model=llm,
    tools=[],
    backend=skills_backend,
    middleware=[skills_middleware, dynamic_model_selection, PDFContextMiddleware(original_system_prompt=SYSTEM_PROMPT)],
    system_prompt=SYSTEM_PROMPT
)