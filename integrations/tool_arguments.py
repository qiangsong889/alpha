"""工具参数的比较视图；不修改实际调用参数，也不推断未声明的默认值。"""

from copy import deepcopy
import json
from typing import Any


def comparison_arguments(arguments: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """省略 schema 明确声明的可选默认值；列表顺序、重复项、字符串保持不变。

    只处理明确的 properties/items；不猜测 oneOf 等分支的默认值。
    required 字段即使声明 default 也不能省略，避免与缺少必填参数混为一谈。
    """
    def encoded(value):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))

    def visit(value, spec):
        if not isinstance(spec, dict):
            return deepcopy(value)
        if isinstance(value, dict):
            properties = spec.get('properties', {})
            required = spec.get('required', [])
            result = {}
            for key, child in value.items():
                field = properties.get(key, {}) if isinstance(properties, dict) else {}
                if (isinstance(field, dict) and key not in required and 'default' in field
                        and encoded(child) == encoded(field['default'])):
                    continue
                result[key] = visit(child, field)
            return result
        if isinstance(value, list):
            return [visit(child, spec.get('items', {})) for child in value]
        return deepcopy(value)

    return visit(arguments, schema)
