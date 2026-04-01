"""
PDF 上下文注入中间件 (PDFContextMiddleware)

从 messages 中最后一个用户消息的 additional_kwargs.attachments 中提取 PDF。

消息格式示例：
    HumanMessage(
        content="解读一下",
        additional_kwargs={
            "attachments": [{
                "type": "file",
                "mimeType": "application/pdf",
                "data": "JVBERi0xLjc...",  # base64 编码
                "metadata": {"filename": "doc.pdf"}
            }]
        }
    )

设计说明（v4）：
    - thread_id 通过 langgraph.config.get_config() 从当前异步上下文获取。
    - 原始 SYSTEM_PROMPT 由构造函数直接传入（original_system_prompt 参数）。
    - 中间件内部维护 thread_id → doc_text 的 per-session 状态字典。
    - 每次请求扫描「最后一条」用户消息，通过 MD5 hash 去重避免重复解析。

    ⚡ v4 核心改动（兼容 SkillsMiddleware）：
    - _build_system_message() 接受 current_system_message 参数。
    - 注入 PDF 时以「当前 request.system_message」为 base（已含 Skills 等内容），
      而非硬编码的 _original_system_content，避免覆盖 SkillsMiddleware 注入的 Skills 列表。
    - awrap_model_call() 将 request.system_message 传给 _build_system_message()。

    middleware 执行顺序（洋葱模型，PDFContextMiddleware 注册在最后 = 最内层执行）：
      ┌─ SkillsMiddleware     → append skills 到 system_message
      │  ┌─ dynamic_model    → 选择模型
      │  │  ┌─ PDFContextMiddleware → 以当前 system_message（含Skills）为base，追加PDF
      │  │  └─ LLM
"""

from __future__ import annotations

import base64
import hashlib
import logging
from typing import Any, Callable, Awaitable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ResponseT
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.typing import ContextT

from processors.pdf import PDFProcessor

logger = logging.getLogger(__name__)

_DOCUMENT_TEMPLATE = """\
以下是用户上传的参考文档，请在回答时充分参考其内容：

<document>
{content}
</document>
"""


def _decode_base64(data: str) -> bytes:
    """将 base64 字符串解码为 bytes。"""
    if "," in data:
        data = data.split(",", 1)[1]
    return base64.b64decode(data)


class PDFContextMiddleware(AgentMiddleware):
    """PDF 文档上下文注入中间件（v3：构造函数传入原始提示词 + get_config 获取 thread_id）。

    核心改进：
    1. original_system_prompt 由构造函数直接传入，完全不依赖运行时 request.system_message，
       彻底规避因服务器未重启导致快照读到污染内容的问题。
    2. thread_id 通过官方 langgraph.config.get_config() 从当前异步上下文获取，
       Runtime 对象本身不含 thread_id。
    3. _session_docs 以 thread_id 为键，维护 per-session 文档状态，新会话天然隔离。
    4. 扫描全部历史消息找最新 PDF，同会话换文件时直接替换（不追加）。
    5. 使用 request.override() 官方不可变替换，不涉及 pickle/deepcopy，
       彻底规避 _thread.RLock 不可序列化问题。
    """

    def __init__(
            self,
            original_system_prompt: str | list | None = None,
            enable_cache: bool = True,
            max_content_length: int = 80_000,
    ):
        """
        Args:
            original_system_prompt: 智能体的原始系统提示词，直接传入 SYSTEM_PROMPT 常量。
                注入时始终以此为基底，确保内容干净，不受运行时污染影响。
                若为 None，则首次请求时从 request.system_message 读取（兼容旧用法）。
            enable_cache: 是否启用 PDF 解析缓存（相同文件不重复解析）。
            max_content_length: PDF 文档内容的最大字符数，超出时截断。
        """
        self._processor = PDFProcessor(enable_cache=enable_cache)
        self._max_content_length = max_content_length
        # 原始系统提示词（只读，永不被修改）
        self._original_system_content: str | list | None = original_system_prompt
        # per-session 文档状态：thread_id -> doc_text（替换语义）
        self._session_docs: dict[str, str] = {}
        # per-session 已解析 PDF 的 hash：thread_id -> pdf_md5
        # 用于判断"当前消息是否携带了和上次不同的新 PDF"
        # 相同 hash → 直接复用 _session_docs，跳过解析
        # 不同 hash → 新文件，重新解析并覆盖
        # 不存在   → 从未上传过，不触发解析
        self._session_pdf_hash: dict[str, str] = {}

    async def awrap_model_call(
            self,
            request: ModelRequest[ContextT],
            handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> Any:
        """拦截 LLM 调用，按会话注入 PDF 文档上下文后再转发。

        设计要点（v4：兼容 SkillsMiddleware 共存）：
            SkillsMiddleware 在 before_agent() 阶段把 skills 内容追加到 system_message，
            PDFContextMiddleware 在最内层（洋葱模型最后执行）收到的 request.system_message
            已经包含了 Skills 内容。
            因此注入 PDF 时必须以「当前 request.system_message」为 base，
            而非 _original_system_content，否则会把 Skills 内容覆盖掉。
        """

        # ── 第一步：兜底快照（original_system_prompt 未传入时的兼容逻辑）──
        if self._original_system_content is None and request.system_message is not None:
            self._original_system_content = request.system_message.content
            logger.warning(
                "[PDFContextMiddleware] original_system_prompt 未传入，"
                "已从首次 request.system_message 快照，建议通过构造函数显式传入以确保安全。"
            )

        # ── 第二步：从 LangGraph 异步上下文获取 thread_id ──
        thread_id = self._get_thread_id()

        # ── 第三步：只处理「最后一条」用户消息中的 PDF 或图片附件 ──
        file_infos = self._extract_files_from_last_message(request)
        # 读取前端传递的多模态开关（True=开启豆包解析，False/None=纯文本提取）
        # 完全由前端决定，不再读取 .env ENABLE_PDF_MULTIMODAL
        frontend_flag = self._get_enable_multimodal_flag(request)
        enable_multimodal = frontend_flag is True  # 未传或 False 统一视为关闭
        if file_infos:
            # 基于所有文件的 md5 计算 hash 以去重
            hash_builder = hashlib.md5()
            for file_data, _, _ in file_infos:
                hash_builder.update(file_data)
            combined_hash = hash_builder.hexdigest()

            if self._session_pdf_hash.get(thread_id) == combined_hash:
                logger.debug(
                    "[PDFContextMiddleware] 会话 %s 文件未变化（hash=%s），跳过重复解析",
                    thread_id, combined_hash,
                )
            else:
                logger.info(
                    "[PDFContextMiddleware] 检测到新文件: %d 个，多模态=%s（前端标志=%s），开始提取文本…",
                    len(file_infos), enable_multimodal, frontend_flag,
                )

                all_text = []
                for file_data, file_name, mime_type in file_infos:
                    logger.info("[PDFContextMiddleware] 解析文件: %s", file_name)
                    text = self._processor.extract_text(
                        file_data, file_name, enable_multimodal=enable_multimodal
                    )
                    if text:
                        all_text.append(f"--- {file_name} ---\n{text}")

                if all_text:
                    merged_text = "\n\n".join(all_text)
                    self._session_docs[thread_id] = merged_text
                    self._session_pdf_hash[thread_id] = combined_hash
                    logger.info(
                        "[PDFContextMiddleware] 会话 %s 文档已更新，共 %d 个文件，长度: %d 字符",
                        thread_id, len(file_infos), len(merged_text),
                    )

        # ── 第四步：根据会话文档状态决定是否注入 ──
        # ⚡ 关键修复：使用 request.system_message（已含 Skills 内容）作为 base，
        #             而非 _original_system_content，避免覆盖 SkillsMiddleware 注入的内容。
        current_doc = self._session_docs.get(thread_id)
        if current_doc:
            # 以当前请求的 system_message 为 base（包含 Skills 追加内容），再拼接 PDF 文档块
            current_system_msg = request.system_message
            request = request.override(
                system_message=self._build_system_message(current_doc, current_system_msg)
            )
            logger.info("[PDFContextMiddleware] 会话 %s system_message 已注入 PDF 文档（保留 Skills）", thread_id)
        else:
            logger.debug("[PDFContextMiddleware] 会话 %s 无文档记录，透传原始 request", thread_id)

        return await handler(request)

    # ──────────────────────────────────────────────
    # 内部辅助方法
    # ──────────────────────────────────────────────

    def _get_thread_id(self) -> str:
        """从 LangGraph 当前异步上下文获取 thread_id，用于区分不同会话。

        官方说明：Runtime 对象不含 config，thread_id 须通过
        langgraph.config.get_config() 从 contextvars 中读取。
        路径: get_config()["configurable"]["thread_id"]
        若取不到则回退到 "__default__"（单用户本地调试场景）。
        """
        try:
            from langgraph.config import get_config
            config = get_config()
            tid = (
                    config.get("metadata").get("thread_id")
                    or config.get("configurable", {}).get("thread_id")
            )
            if tid:
                return str(tid)
        except Exception:
            pass
        return "__default__"

    def _extract_files_from_last_message(self, request: ModelRequest) -> list[tuple[bytes, str, str]]:
        """只从「最后一条用户消息」中提取 PDF 和图片附件。
        返回列表：[(bytes, filename, mime_type), ...]
        """
        extracted_files = []
        if not request.messages:
            return extracted_files

        # 只取最后一条消息
        last_msg = request.messages[-1]
        if not isinstance(last_msg, HumanMessage):
            return extracted_files

        attachments = last_msg.additional_kwargs.get("attachments", [])
        if not isinstance(attachments, list):
            return extracted_files

        for att in attachments:
            if not isinstance(att, dict):
                continue

            filename = att.get("metadata", {}).get("filename", f"file_{len(extracted_files)}")
            mime_type = att.get("mimeType", "").lower()

            is_pdf = mime_type == "application/pdf" or filename.lower().endswith(".pdf")
            is_image = mime_type.startswith("image/") or filename.lower().endswith(
                (".jpg", ".jpeg", ".png", ".gif", ".webp"))
            is_word = (
                    mime_type in [
                "application/msword",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ]
                    or filename.lower().endswith((".doc", ".docx"))
            )

            if not (is_pdf or is_image or is_word):
                continue

            data = att.get("data")
            if not data or not isinstance(data, str):
                continue

            try:
                file_bytes = _decode_base64(data)
                filename = att.get("metadata", {}).get("filename", f"file_{len(extracted_files)}")
                extracted_files.append((file_bytes, filename, mime_type))
            except Exception as e:
                logger.warning("[PDFContextMiddleware] 附件解码失败: %s", e)
                continue

        return extracted_files

    def _get_enable_multimodal_flag(self, request: ModelRequest) -> bool | None:
        """从消息历史中读取前端传递的 ENABLE_PDF_MULTIMODAL 标志。

        倒序扫描全部 HumanMessage（不只看最后一条），取最近一条携带该字段的值。
        在 LangGraph 多步执行中，最后一条消息可能是 AI 消息，需要往前找。

        Returns:
            True / False：前端显式传入的值。
            None：消息中未包含该字段，此时由 settings.ENABLE_PDF_MULTIMODAL 决定。
        """
        for msg in reversed(request.messages):
            if not isinstance(msg, HumanMessage):
                continue
            value = msg.additional_kwargs.get("ENABLE_PDF_MULTIMODAL")
            if value is None:
                continue
            if isinstance(value, bool):
                return value
            return str(value).lower() == "true"
        return None

    def _build_system_message(
            self,
            doc_text: str,
            current_system_message: SystemMessage | None = None,
    ) -> SystemMessage:
        """以当前 system_message 为基底，拼接文档块，返回全新的 SystemMessage。

        v4 关键改动：
            优先使用 current_system_message.content 作为 base（已含 Skills 注入内容），
            仅在 current_system_message 为 None 时回退到 _original_system_content。
            这样无论 SkillsMiddleware 追加了多少内容，PDF 注入都不会覆盖它们。

        Args:
            doc_text: 已提取的 PDF 文档文本。
            current_system_message: 当前请求中的 SystemMessage（来自 request.system_message），
                应已包含 SkillsMiddleware 等上游中间件追加的内容。
                若为 None，则回退到 _original_system_content。
        """
        if len(doc_text) > self._max_content_length:
            doc_text = doc_text[: self._max_content_length] + "\n\n[文档内容已截断...]"

        doc_block_text = _DOCUMENT_TEMPLATE.format(content=doc_text)

        # ⚡ 优先以「当前请求的 system_message」为 base，保留 Skills 等中间件追加的内容
        if current_system_message is not None:
            base_content = current_system_message.content
        else:
            base_content = self._original_system_content

        if isinstance(base_content, str):
            new_content: str | list = base_content + "\n\n" + doc_block_text
        elif isinstance(base_content, list):
            new_content = list(base_content) + [{"type": "text", "text": doc_block_text}]
        else:
            new_content = doc_block_text

        return SystemMessage(content=new_content)

    def clear_session(self, thread_id: str) -> None:
        """主动清除指定会话的文档状态（可供外部调用，例如用户点击「清除上下文」）。"""
        removed = self._session_docs.pop(thread_id, None)
        self._session_pdf_hash.pop(thread_id, None)
        if removed is not None:
            logger.info("[PDFContextMiddleware] 会话 %s 的文档状态已清除", thread_id)

    def get_session_stats(self) -> dict:
        """获取当前所有活跃会话的文档状态统计（调试用）。"""
        return {
            "active_sessions": len(self._session_docs),
            "session_ids": list(self._session_docs.keys()),
            "doc_lengths": {tid: len(text) for tid, text in self._session_docs.items()},
        }
