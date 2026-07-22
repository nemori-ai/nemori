"""Safe, educational SWE-bench payloads for the dashboard API."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from evaluation.swebench.contracts import FORBIDDEN_API_FIELDS, assert_no_forbidden_fields


_DEMO_ISSUE = """3.0: DateTime fields cannot be used as inner field for List or Tuple fields
Between releases 3.0.0rc8 and 3.0.0rc9, `DateTime` fields started throwing an
error when used as inner fields of container fields such as `List` or `Tuple`.

```python
from marshmallow import fields, Schema

class MySchema(Schema):
    times = fields.List(fields.DateTime())

s = MySchema()
```

The failing code raises `AttributeError: 'List' object has no attribute 'opts'`.
The container field is being treated as though it were the root Schema.
"""

_DEMO_AGENT_OUTPUT_DIFF = """diff --git a/src/marshmallow/fields.py b/src/marshmallow/fields.py
--- a/src/marshmallow/fields.py
+++ b/src/marshmallow/fields.py
@@ -1114,7 +1114,7 @@ def _bind_to_schema(self, field_name, schema):
         super()._bind_to_schema(field_name, schema)
         self.format = (
             self.format
-            or getattr(schema.opts, self.SCHEMA_OPTS_VAR_NAME)
+            or getattr(self.root.opts, self.SCHEMA_OPTS_VAR_NAME)
             or self.DEFAULT_FORMAT
         )
"""

_GUIDE_PAYLOAD: dict[str, Any] = {
    "title": "SWE-bench Experiment",
    "summary": (
        "SWE-bench 不是问答测试。它检查 AI 能否理解真实 GitHub Issue，"
        "阅读并修改真实代码仓库，最后通过确定性的回归测试。"
    ),
    "lifecycle": [
        "理解 GitHub Issue",
        "checkout 固定的 base_commit",
        "搜索并分析代码",
        "生成 git diff",
        "在隔离 Docker 环境中应用候选差异",
        "执行 FAIL_TO_PASS 与 PASS_TO_PASS",
        "两组测试都满足要求才记为 resolved",
    ],
    "data_structure": {
        "agent_visible_fields": [
            "instance_id",
            "repo",
            "base_commit",
            "problem_statement",
        ],
        "evaluation_only": [
            "开发者参考修复差异",
            "评测专用测试差异",
            "确定性测试判定元数据",
        ],
        "safety_note": (
            "评测专用答案和测试实现不会进入 Agent 输入、Agent sandbox 或公共 API。"
        ),
    },
    "demo": {
        "instance_id": "marshmallow-code__marshmallow-1359",
        "repo": "marshmallow-code/marshmallow",
        "base_commit": "b40a0f4e33823e6d0f341f7e8684e359a99060d1",
        "source_dataset": "princeton-nlp/SWE-bench_Lite",
        "source_split": "dev",
        "excluded_from_scoring": True,
        "problem_statement": _DEMO_ISSUE,
        "agent_input": {
            "issue": _DEMO_ISSUE,
            "repo": "marshmallow-code/marshmallow",
            "base_commit": "b40a0f4e33823e6d0f341f7e8684e359a99060d1",
            "visible_fields": [
                "instance_id",
                "repo",
                "base_commit",
                "problem_statement",
            ],
        },
        "agent_output_diff": _DEMO_AGENT_OUTPUT_DIFF,
        "harness": {
            "resolved": True,
            "patch_applied": True,
            "fail_to_pass": {
                "before": "failed",
                "after": "passed",
                "tests": [
                    "tests/test_fields.py::TestParentAndName::test_datetime_list_inner_format"
                ],
            },
            "pass_to_pass": {
                "before": "passed",
                "after": "passed",
                "count": 76,
                "examples": [
                    "tests/test_fields.py::TestParentAndName::test_simple_field_parent_and_name",
                    "tests/test_fields.py::TestField::test_repr",
                    "tests/test_fields.py::TestField::test_custom_field_receives_attr_and_obj",
                ],
            },
            "before": "DateTime inner fields fail while binding to a List or Tuple.",
            "after": "The field reads options from the root Schema and regression tests pass.",
            "success_reason": (
                "候选差异可以应用；目标失败测试转为通过；原有 76 个回归测试仍通过。"
            ),
        },
    },
}


def load_dataset_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load a local SWE-bench JSON array and validate stable instance IDs."""

    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("SWE-bench dataset must be a JSON array")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(data):
        if not isinstance(item, Mapping):
            raise ValueError(f"SWE-bench row {index} must be an object")
        row = dict(item)
        instance_id = row.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id.strip():
            raise ValueError(f"SWE-bench row {index} has no non-empty instance_id")
        if instance_id in seen:
            raise ValueError(f"SWE-bench dataset contains duplicate {instance_id}")
        seen.add(instance_id)
        rows.append(row)
    if not rows:
        raise ValueError("SWE-bench dataset is empty")
    return rows


def swebench_guide_payload() -> dict[str, Any]:
    """Return an isolated, recursively leakage-scanned guide payload."""

    payload = copy.deepcopy(_GUIDE_PAYLOAD)
    assert_no_forbidden_fields(payload, forbidden_fields=FORBIDDEN_API_FIELDS)
    return payload
