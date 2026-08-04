#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any


TARGET_MAP_KEYS = {"correction_map", "regex_correction_map"}


def parse_pairs(text: str) -> Any:
    return json.loads(text, object_pairs_hook=lambda pairs: list(pairs))


def is_pairs_object(obj: Any) -> bool:
    return (
        isinstance(obj, list)
        and all(
            isinstance(x, tuple)
            and len(x) == 2
            and isinstance(x[0], str)
            for x in obj
        )
    )


def stable_key(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def dedupe_keep_order(items: list[Any]) -> list[Any]:
    seen = set()
    out = []
    for item in items:
        k = stable_key(item)
        if k not in seen:
            seen.add(k)
            out.append(item)
    return out


def clean_alias_list(v: Any) -> Any:
    if not isinstance(v, list):
        return v

    out = []
    for x in v:
        if isinstance(x, str):
            s = x.strip()
            if s != "":
                out.append(s)
        else:
            out.append(x)

    return dedupe_keep_order(out)


def merge_map_pairs(pairs: list[tuple[str, Any]]) -> OrderedDict[str, Any]:
    out: OrderedDict[str, Any] = OrderedDict()

    for raw_key, raw_val in pairs:
        key = raw_key.strip() if isinstance(raw_key, str) else raw_key

        val = walk(raw_val)
        val = clean_alias_list(val)

        if key != "" and isinstance(val, list) and len(val) == 0:
            continue

        if key not in out:
            out[key] = val
            continue

        old = out[key]
        if isinstance(old, list) and isinstance(val, list):
            out[key] = dedupe_keep_order(old + val)
        elif isinstance(old, list):
            out[key] = dedupe_keep_order(old + [val])
        elif isinstance(val, list):
            out[key] = dedupe_keep_order([old] + val)
        else:
            if old != val:
                out[key] = dedupe_keep_order([old, val])

    return out


def walk(node: Any) -> Any:
    if is_pairs_object(node):
        result: OrderedDict[str, Any] = OrderedDict()
        for k, v in node:
            if k in TARGET_MAP_KEYS and is_pairs_object(v):
                result[k] = merge_map_pairs(v)
            else:
                result[k] = walk(v)
        return result

    if isinstance(node, list):
        return [walk(x) for x in node]

    return node


def dumps_compact_inline_array(arr: list[Any]) -> str:
    return "[" + ", ".join(json.dumps(x, ensure_ascii=False) for x in arr) + "]"


def render_json(obj: Any, indent: int = 0, parent_key: str | None = None) -> str:
    sp = "  " * indent
    sp2 = "  " * (indent + 1)

    if isinstance(obj, OrderedDict) or isinstance(obj, dict):
        if not obj:
            return "{}"

        parts = []
        for k, v in obj.items():
            key_s = json.dumps(k, ensure_ascii=False)

            if parent_key in TARGET_MAP_KEYS and isinstance(v, list):
                val_s = dumps_compact_inline_array(v)
            else:
                val_s = render_json(v, indent + 1, parent_key=k)

            parts.append(f"{sp2}{key_s}: {val_s}")

        return "{\n" + ",\n".join(parts) + f"\n{sp}" + "}"

    if isinstance(obj, list):
        if not obj:
            return "[]"

        parts = [f'{"  " * (indent + 1)}{render_json(x, indent + 1)}' for x in obj]
        return "[\n" + ",\n".join(parts) + f"\n{sp}" + "]"

    return json.dumps(obj, ensure_ascii=False)


def main() -> int:
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "config/domain_defaults.json")
    if not target.exists():
        print(f"找不到檔案: {target}", file=sys.stderr)
        return 1

    raw = target.read_text(encoding="utf-8")
    parsed = parse_pairs(raw)
    cleaned = walk(parsed)

    text = render_json(cleaned) + "\n"
    target.write_text(text, encoding="utf-8")

    print(f"done: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
