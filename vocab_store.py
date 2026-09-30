"""网页可改的词表和打标设置。缺文件时回落到 billfish_vocab 默认树。"""

from __future__ import annotations

import json
from pathlib import Path

from billfish_vocab import (
    default_groups,
    normalize_groups,
    parent_of_from_groups,
    parent_parent_from_groups,
    system_prompt_from_groups,
)

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "config"
VOCAB_PATH = CONFIG_DIR / "vocab.json"
SETTINGS_PATH = CONFIG_DIR / "settings.json"

DEFAULT_SETTINGS = {
    "folder": "",
    "out": str(ROOT / "outputs" / "web_batch"),
    "resume": True,
    "limit": 0,
    "billfish_db": "",
}


def _read_json(path: Path, fallback):
    if not path.is_file():
        return fallback
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback
    return data if data is not None else fallback


def load_groups() -> list[dict]:
    data = _read_json(VOCAB_PATH, None)
    groups = data.get("groups") if isinstance(data, dict) else data
    try:
        return normalize_groups(groups)
    except (TypeError, ValueError):
        return default_groups()


def save_groups(groups: list[dict]) -> list[dict]:
    cleaned = normalize_groups(groups)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"groups": cleaned}
    VOCAB_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return cleaned


def reset_groups() -> list[dict]:
    if VOCAB_PATH.exists():
        VOCAB_PATH.unlink()
    return default_groups()


def load_settings() -> dict:
    data = _read_json(SETTINGS_PATH, {})
    if not isinstance(data, dict):
        data = {}
    out = dict(DEFAULT_SETTINGS)
    if isinstance(data.get("folder"), str):
        out["folder"] = data["folder"].strip()
    if isinstance(data.get("out"), str) and data["out"].strip():
        out["out"] = data["out"].strip()
    if isinstance(data.get("resume"), bool):
        out["resume"] = data["resume"]
    try:
        limit = int(data.get("limit", 0) or 0)
        out["limit"] = max(0, limit)
    except (TypeError, ValueError):
        pass
    if isinstance(data.get("billfish_db"), str):
        out["billfish_db"] = data["billfish_db"].strip()
    return out


def save_settings(settings: dict) -> dict:
    current = load_settings()
    folder = settings.get("folder")
    if isinstance(folder, str):
        current["folder"] = folder.strip()
    out = settings.get("out")
    if isinstance(out, str) and out.strip():
        current["out"] = out.strip()
    if isinstance(settings.get("resume"), bool):
        current["resume"] = settings["resume"]
    if "limit" in settings:
        try:
            current["limit"] = max(0, int(settings.get("limit") or 0))
        except (TypeError, ValueError):
            pass
    if isinstance(settings.get("billfish_db"), str):
        current["billfish_db"] = settings["billfish_db"].strip()
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return current


def load_active() -> dict:
    groups = load_groups()
    parent_of = parent_of_from_groups(groups)
    return {
        "groups": groups,
        "parent_of": parent_of,
        "parent_parent": parent_parent_from_groups(groups),
        "allowed": set(parent_of) | {"跳过"},
        "prompt": system_prompt_from_groups(groups),
        "vocab_path": str(VOCAB_PATH) if VOCAB_PATH.is_file() else None,
        "leaf_count": len(parent_of),
    }
