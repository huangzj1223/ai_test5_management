"""
测试用例 JSON 导出工具

本模块提供将测试用例导出为 JSON 文件的能力，
用于给自动化脚本生成器等下游程序稳定消费。
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _extract_field(case: dict[str, Any], *keys: str, default: Any = "") -> Any:
    """从字典中按多个候选键提取值。"""
    for key in keys:
        if key in case:
            return case[key]
    return default


def _ensure_list(value: Any) -> list[Any]:
    """将值标准化为列表。"""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _normalize_steps(steps: Any) -> list[dict[str, Any]]:
    """标准化测试步骤。"""
    normalized_steps: list[dict[str, Any]] = []
    for index, step in enumerate(_ensure_list(steps), start=1):
        if isinstance(step, dict):
            normalized_steps.append(
                {
                    "seq": step.get("seq", step.get("step", index)),
                    "action": step.get("action", step.get("操作描述", "")),
                    "target": step.get("target", step.get("操作对象", "")),
                    "data": step.get("data", ""),
                }
            )
        else:
            normalized_steps.append(
                {
                    "seq": index,
                    "action": str(step),
                    "target": "",
                    "data": "",
                }
            )
    return normalized_steps


def _normalize_expected_results(expected_results: Any) -> list[str]:
    """标准化预期结果。"""
    return [str(item) for item in _ensure_list(expected_results) if str(item).strip()]


def _normalize_preconditions(preconditions: Any) -> list[str]:
    """标准化前置条件。"""
    return [str(item) for item in _ensure_list(preconditions) if str(item).strip()]


def _normalize_test_data(test_data: Any) -> dict[str, Any] | str:
    """标准化测试数据。"""
    if test_data is None:
        return {}
    if isinstance(test_data, dict):
        return test_data
    return str(test_data)


def _infer_automation_type(case_type: str, explicit_value: Any) -> str:
    """推断自动化类型。"""
    if explicit_value:
        return str(explicit_value)

    lowered = case_type.lower()
    if "接口" in case_type or "api" in lowered:
        return "api"
    if "手工" in case_type or "manual" in lowered:
        return "manual"
    return "ui"


def _infer_automation_candidate(case_type: str, automation_type: str, explicit_value: Any) -> bool:
    """推断是否适合自动化。"""
    if isinstance(explicit_value, bool):
        return explicit_value
    if explicit_value is not None and str(explicit_value).strip():
        return str(explicit_value).strip().lower() in {"true", "1", "yes", "y"}

    lowered = case_type.lower()
    if automation_type == "manual":
        return False
    if "可用性" in case_type or "体验" in case_type:
        return False
    if "兼容" in case_type or "性能" in case_type:
        return False
    if "manual" in lowered:
        return False
    return True


def _infer_login_required(module: str, explicit_value: Any) -> bool:
    """推断是否需要登录。"""
    if isinstance(explicit_value, bool):
        return explicit_value
    if explicit_value is not None and str(explicit_value).strip():
        return str(explicit_value).strip().lower() in {"true", "1", "yes", "y"}

    return module not in {"登录", "注册", "认证", "Login", "Register", "Auth"}


def _slugify(value: str) -> str:
    """生成适合作为文件名的一段标识。"""
    text = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "-", value).strip("-")
    return text or "generated-testcase"


def _build_script_path(case_id: str, module: str, explicit_value: Any, automation_candidate: bool, automation_type: str) -> str:
    """生成建议脚本路径。"""
    if explicit_value:
        return str(explicit_value)
    if not automation_candidate or automation_type != "ui":
        return ""

    module_slug = _slugify(module or "generated")
    case_slug = _slugify(case_id or "testcase")
    return f"tests/{module_slug}/{case_slug}.ui.spec.ts"


def _build_tags(module: str, priority: str, case_type: str, automation_type: str, explicit_tags: Any) -> list[str]:
    """生成标签列表。"""
    if isinstance(explicit_tags, list):
        return [str(tag) for tag in explicit_tags if str(tag).strip()]
    if explicit_tags is not None and str(explicit_tags).strip():
        return [str(explicit_tags)]

    tags = [module, priority, case_type, automation_type]
    return [str(tag) for tag in tags if str(tag).strip()]


def normalize_test_case(case: dict[str, Any]) -> dict[str, Any]:
    """将测试用例标准化为统一 JSON 结构，并补充自动化友好字段。"""
    case_id = str(_extract_field(case, "id", "用例编号", default="")).strip()
    title = str(_extract_field(case, "title", "用例标题", default="")).strip()
    module = str(_extract_field(case, "module", "所属模块", default="未分类")).strip() or "未分类"
    case_type = str(_extract_field(case, "type", "用例类型", default="功能测试")).strip() or "功能测试"
    priority = str(_extract_field(case, "priority", "优先级", default="P2")).strip() or "P2"
    remarks = str(_extract_field(case, "remarks", "备注", default="")).strip()

    automation_type = _infer_automation_type(
        case_type,
        _extract_field(case, "automation_type", "自动化类型", default=None),
    )
    automation_candidate = _infer_automation_candidate(
        case_type,
        automation_type,
        _extract_field(case, "automation_candidate", "是否适合自动化", default=None),
    )
    login_required = _infer_login_required(
        module,
        _extract_field(case, "login_required", "是否需要登录", default=None),
    )

    return {
        "id": case_id,
        "title": title,
        "module": module,
        "feature": str(_extract_field(case, "feature", "一级分类", default=module)).strip() or module,
        "story": str(_extract_field(case, "story", "二级分类", default=title)).strip() or title,
        "type": case_type,
        "priority": priority,
        "preconditions": _normalize_preconditions(_extract_field(case, "preconditions", "前置条件", default=[])),
        "steps": _normalize_steps(_extract_field(case, "steps", "测试步骤", default=[])),
        "test_data": _normalize_test_data(_extract_field(case, "test_data", "测试数据", default={})),
        "expected_results": _normalize_expected_results(_extract_field(case, "expected_results", "预期结果", default=[])),
        "remarks": remarks,
        "automation_candidate": automation_candidate,
        "automation_type": automation_type,
        "login_required": login_required,
        "script_path": _build_script_path(
            case_id,
            module,
            _extract_field(case, "script_path", "脚本路径", default=None),
            automation_candidate,
            automation_type,
        ),
        "tags": _build_tags(
            module,
            priority,
            case_type,
            automation_type,
            _extract_field(case, "tags", "标签", default=None),
        ),
        "locator_hints": _extract_field(case, "locator_hints", "定位提示", default=[]),
        "assertion_level": str(
            _extract_field(case, "assertion_level", "断言层级", default="ui" if automation_type == "ui" else automation_type)
        ).strip(),
    }


def export_test_cases_to_json(
    test_cases: list[dict[str, Any]],
    output_path: str | Path,
    project_name: str = "XINZHI-TEST",
) -> str:
    """
    将测试用例列表导出为 JSON 文件，并补充自动化友好字段。

    Args:
        test_cases: 测试用例字典列表。
        output_path: 导出的 JSON 文件路径。
        project_name: 项目标识，默认 XINZHI-TEST。

    Returns:
        导出文件的绝对路径字符串。
    """
    if not test_cases:
        raise ValueError("测试用例列表为空，无法导出 JSON。")

    normalized_cases = [normalize_test_case(case) for case in test_cases]

    payload = {
        "schema_version": "1.0",
        "project": project_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_count": len(normalized_cases),
        "test_cases": normalized_cases,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output_path.resolve())
