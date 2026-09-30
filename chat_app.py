#!/usr/bin/env python3
"""本地视觉聊天页：浏览器拖图对话，模型可在页面里启动/停止。"""

from __future__ import annotations

import argparse
import base64
import gc
import io
import json
import os
import subprocess
import threading
import time
import webbrowser
from collections import deque
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from tag_images import (
    PRIMARY_MODEL,
    collect_images,
    pending_images,
    resize_max_edge,
    run_loaded_batch,
)
from vocab_store import (
    ROOT as WORKSPACE,
    load_active,
    load_settings,
    reset_groups,
    save_groups,
    save_settings,
)

CHAT_SYSTEM = """你是运行在用户电脑上的本地视觉助手。根据用户附带的图片回答。
规则：
1. 只根据可见内容回答，不要编造看不见的信息。
2. 用中性、简洁的中文说明，不要输出除回答以外的系统套话。
3. 若主体明显是未成年人：只回答「跳过」，不要描述。"""

UI_DIR = Path(__file__).resolve().parent / "ui"
MODEL_ID = PRIMARY_MODEL
MODEL_CACHE = (
    Path.home()
    / ".cache/huggingface/hub/models--mlx-community--Qwen3-VL-8B-NSFW-Caption-V4.5-mxfp4"
)

_lock = threading.Lock()
_ctrl = threading.Lock()
_state = {
    "status": "stopped",  # stopped | loading | ready | stopping | error
    "error": None,
    "model": None,
    "processor": None,
    "config": None,
    "load_s": None,
    "loaded_at": None,
}
_batch_lock = threading.Lock()
_batch = {
    "status": "idle",  # idle | running | stopping | done | error
    "error": None,
    "folder": "",
    "out": "",
    "total": 0,
    "done": 0,
    "skipped": 0,
    "current": None,
    "last": None,
    "stats": {},
    "log": deque(maxlen=400),
    "started_at": None,
    "report": None,
    "cancel": False,
    "jsonl": "",
}


class ChatIn(BaseModel):
    mode: str = Field(default="chat")
    message: str = Field(default="")
    image_b64: str | None = None
    history: list[dict] = Field(default_factory=list)
    max_tokens: int | None = None


class SettingsIn(BaseModel):
    folder: str = ""
    out: str = ""
    resume: bool = True
    limit: int = 0
    billfish_db: str = ""


class VocabIn(BaseModel):
    groups: list[dict]


class BillfishIn(BaseModel):
    jsonl: str = ""
    db: str = ""
    out: str = ""
    apply: bool = False
    strip_old: bool = False
    create_missing: bool = True
    replace: bool = True


class ScanIn(BaseModel):
    folder: str
    out: str = ""
    resume: bool = True


class BatchStartIn(BaseModel):
    folder: str
    out: str = ""
    resume: bool = True
    limit: int = 0


def _dir_gb(path: Path) -> float | None:
    if not path.exists():
        return None
    blobs = path / "blobs"
    root = blobs if blobs.exists() else path
    seen = set()
    total = 0
    for p in root.rglob("*"):
        if not p.is_file() or p.name.endswith(".incomplete"):
            continue
        try:
            key = p.stat().st_ino
        except OSError:
            key = str(p)
        if key in seen:
            continue
        seen.add(key)
        total += p.stat().st_size
    return round(total / (1024**3), 2)


def _mlx_peak_gb() -> float | None:
    try:
        import mlx.core as mx

        return round(mx.get_peak_memory() / (1024**3), 2)
    except Exception:
        return None


def _swap_usage() -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True).strip()
    except Exception:
        return ""


def _snapshot() -> dict:
    with _batch_lock:
        batch = {
            "status": _batch["status"],
            "error": _batch["error"],
            "folder": _batch["folder"],
            "out": _batch["out"],
            "total": _batch["total"],
            "done": _batch["done"],
            "skipped": _batch["skipped"],
            "current": _batch["current"],
            "last": _batch["last"],
            "stats": dict(_batch["stats"]),
            "log": list(_batch["log"]),
            "started_at": _batch["started_at"],
            "report": _batch["report"],
            "jsonl": _batch["jsonl"],
            "running": _batch["status"] in ("running", "stopping"),
        }
    vocab = load_active()
    return {
        "status": _state["status"],
        "ready": _state["status"] == "ready",
        "loading": _state["status"] == "loading",
        "error": _state["error"],
        "model_id": MODEL_ID,
        "model_name": MODEL_ID.split("/")[-1],
        "quant": "mxfp4 4-bit",
        "engine": "MLX",
        "disk_gb": _dir_gb(MODEL_CACHE),
        "load_s": _state["load_s"],
        "loaded_at": _state["loaded_at"],
        "mlx_peak_gb": _mlx_peak_gb() if _state["status"] == "ready" else None,
        "swap": _swap_usage(),
        "cache_path": str(MODEL_CACHE),
        "batch": batch,
        "leaf_count": vocab["leaf_count"],
    }


def _decode_image(image_b64: str):
    from PIL import Image

    raw = image_b64.split(",", 1)[-1]
    data = base64.b64decode(raw)
    img = Image.open(io.BytesIO(data))
    return resize_max_edge(img, 768)[0]


def _load_model() -> None:
    from mlx_vlm import load
    from mlx_vlm.utils import load_config

    t0 = time.perf_counter()
    try:
        model, processor = load(MODEL_ID)
        config = load_config(MODEL_ID)
        with _lock:
            _state.update(
                {
                    "model": model,
                    "processor": processor,
                    "config": config,
                    "status": "ready",
                    "error": None,
                    "load_s": round(time.perf_counter() - t0, 2),
                    "loaded_at": time.strftime("%H:%M:%S"),
                }
            )
    except Exception as e:
        with _lock:
            _state.update(
                {
                    "model": None,
                    "processor": None,
                    "config": None,
                    "status": "error",
                    "error": repr(e),
                    "load_s": None,
                    "loaded_at": None,
                }
            )


def _unload_model() -> None:
    with _lock:
        _state.update(
            {
                "status": "stopping",
                "model": None,
                "processor": None,
                "config": None,
                "ready": False,
            }
        )
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
        mx.reset_peak_memory()
    except Exception:
        pass
    gc.collect()
    with _lock:
        _state.update(
            {
                "status": "stopped",
                "error": None,
                "load_s": None,
                "loaded_at": None,
            }
        )


def _messages(req: ChatIn) -> list[dict]:
    if req.mode == "tag":
        system = load_active()["prompt"]
        user = (req.message or "").strip() or "给这张图打标签。只输出 JSON。tags 只能 1 到 4 个。"
    else:
        system = CHAT_SYSTEM
        user = (req.message or "").strip() or "描述这张图。"
    msgs: list[dict] = [{"role": "system", "content": system}]
    for turn in req.history[-8:]:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": user})
    return msgs


def _resolve_dir(raw: str, *, must_exist: bool) -> Path:
    text = (raw or "").strip()
    if not text:
        raise ValueError("请填写目录")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (WORKSPACE / path).resolve()
    else:
        path = path.resolve()
    if must_exist and not path.is_dir():
        raise ValueError(f"找不到文件夹：{path}")
    return path


def _scan_folder(folder: str, out: str, resume: bool) -> dict:
    src = _resolve_dir(folder, must_exist=True)
    images = collect_images(src)
    out_dir = _resolve_dir(out, must_exist=False) if (out or "").strip() else (WORKSPACE / "outputs" / "web_batch")
    skipped = 0
    pending = list(images)
    if resume and images:
        pending, skipped = pending_images(images, out_dir, src)
    subdirs = []
    try:
        for p in sorted(src.iterdir()):
            if p.is_dir() and not p.name.startswith(".") and p.name not in {"__pycache__", "node_modules"}:
                subdirs.append({"name": p.name, "path": str(p)})
    except OSError:
        pass
    samples = [p.name for p in images[:8]]
    jsonl = out_dir / "results.jsonl"
    return {
        "folder": str(src),
        "out": str(out_dir),
        "exists": True,
        "n_images": len(images),
        "n_done": skipped,
        "n_todo": len(pending),
        "samples": samples,
        "subdirs": subdirs[:40],
        "has_jsonl": jsonl.is_file(),
        "jsonl": str(jsonl) if jsonl.is_file() else "",
    }


def _suggest_folders() -> list[dict]:
    items = []
    seen = set()

    def add(path: Path, label: str):
        try:
            if not path.is_dir():
                return
            key = str(path.resolve())
        except OSError:
            return
        if key in seen:
            return
        seen.add(key)
        items.append({"path": key, "label": label})

    settings = load_settings()
    if settings.get("folder"):
        add(Path(settings["folder"]).expanduser(), "上次素材目录")
    add(WORKSPACE, "工作区")
    volumes = Path("/Volumes")
    if volumes.is_dir():
        for vol in sorted(volumes.iterdir()):
            if not vol.is_dir() or vol.name.startswith("."):
                continue
            add(vol, f"磁盘 {vol.name}")
            lib = vol / "AI自动打标素材库"
            if lib.is_dir():
                add(lib, "AI自动打标素材库")
                try:
                    for child in sorted(lib.iterdir()):
                        if child.is_dir() and not child.name.startswith("."):
                            add(child, child.name)
                except OSError:
                    pass
    return items[:30]


def _suggest_billfish_dbs(folder: str = "") -> list[dict]:
    seen: set[str] = set()
    items: list[dict] = []

    def add(path: Path, label: str):
        try:
            if not path.is_file():
                return
            key = str(path.resolve())
        except OSError:
            return
        if key in seen:
            return
        seen.add(key)
        items.append({"path": key, "label": label})

    settings = load_settings()
    if settings.get("billfish_db"):
        add(Path(settings["billfish_db"]).expanduser(), "上次数据库")
    roots: list[Path] = []
    if folder.strip():
        try:
            src = _resolve_dir(folder, must_exist=False)
            roots.extend([src, src.parent])
        except ValueError:
            pass
    roots.append(WORKSPACE)
    volumes = Path("/Volumes")
    if volumes.is_dir():
        try:
            roots.extend(sorted(p for p in volumes.iterdir() if p.is_dir() and not p.name.startswith("."))[:8])
        except OSError:
            pass
    for root in roots:
        add(root / ".bf" / "billfish.db", str(root.name or root))
        try:
            for child in root.iterdir():
                if child.is_dir() and not child.name.startswith("."):
                    add(child / ".bf" / "billfish.db", child.name)
        except OSError:
            pass
    return items[:20]


def _append_log(msg: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    _batch["log"].append(f"{stamp}  {msg}")


def _wait_model_ready(timeout: float = 180.0) -> None:
    deadline = time.time() + timeout
    last_note = 0.0
    while time.time() < deadline:
        with _batch_lock:
            if _batch["cancel"]:
                raise RuntimeError("已手动停止")
        status = _state["status"]
        if status == "ready":
            with _batch_lock:
                _append_log("模型已就绪")
            return
        if status == "error":
            raise RuntimeError(_state.get("error") or "模型加载失败")
        if status == "stopped":
            with _ctrl:
                if _state["status"] == "stopped":
                    _state["status"] = "loading"
                    _state["error"] = None
                    threading.Thread(target=_load_model, daemon=True).start()
            with _batch_lock:
                _append_log("正在启动模型…")
            last_note = time.time()
        elif status == "loading":
            now = time.time()
            if last_note == 0:
                with _batch_lock:
                    _append_log("等待模型加载完成…")
                last_note = now
            elif now - last_note >= 5:
                with _batch_lock:
                    _append_log("模型仍在加载，请稍等…")
                last_note = now
        time.sleep(0.4)
    raise RuntimeError("等待模型启动超时")


def _run_batch_job(folder: str, out: str, resume: bool, limit: int) -> None:
    try:
        src = _resolve_dir(folder, must_exist=True)
        out_dir = _resolve_dir(out, must_exist=False)
        images = collect_images(src)
        if limit and limit > 0:
            images = images[:limit]
        if not images:
            raise RuntimeError(f"文件夹里没有图片：{src}")
        with _batch_lock:
            _batch.update(
                {
                    "folder": str(src),
                    "out": str(out_dir),
                    "total": len(images),
                    "done": 0,
                    "skipped": 0,
                    "jsonl": str(out_dir / "results.jsonl"),
                }
            )
            _append_log(f"扫描到 {len(images)} 张，输出 {out_dir}")
        _wait_model_ready()
        with _batch_lock:
            if _batch["cancel"]:
                raise RuntimeError("已手动停止")
        vocab = load_active()
        with _batch_lock:
            _append_log(f"使用词表 {vocab['leaf_count']} 个叶子标签，开始推理")

        def on_progress(rec, i, total, skipped):
            line = f"[{i}/{total}] {rec.get('seconds')}s {rec.get('status')} {Path(rec.get('file') or '').name} -> {rec.get('tags')}"
            with _batch_lock:
                _batch["done"] = i
                _batch["total"] = total
                _batch["skipped"] = skipped
                _batch["current"] = rec.get("rel") or rec.get("file")
                _batch["last"] = {
                    "file": Path(rec.get("file") or "").name,
                    "status": rec.get("status"),
                    "tags": rec.get("tags") or [],
                    "seconds": rec.get("seconds"),
                }
                stats = dict(_batch["stats"])
                stats[rec.get("status") or "error"] = stats.get(rec.get("status") or "error", 0) + 1
                _batch["stats"] = stats
                _append_log(line)

        def should_stop():
            with _batch_lock:
                return bool(_batch["cancel"])

        with _lock:
            if _state["status"] != "ready" or _state["model"] is None:
                raise RuntimeError("模型未就绪")
            model = _state["model"]
            processor = _state["processor"]
            config = _state["config"]
        report = run_loaded_batch(
            model,
            processor,
            config,
            src_root=src,
            out_dir=out_dir,
            images=images,
            resume=resume,
            system_prompt=vocab["prompt"],
            allowed_tags=vocab["allowed"],
            on_progress=on_progress,
            should_stop=should_stop,
            infer_lock=_lock,
        )
        with _batch_lock:
            _batch["report"] = report
            _batch["stats"] = report.get("stats") or _batch["stats"]
            _batch["skipped"] = report.get("skipped_resume") or _batch["skipped"]
            _batch["jsonl"] = report.get("jsonl") or _batch["jsonl"]
            _batch["status"] = "stopping" if report.get("stopped") else "done"
            _batch["error"] = "已手动停止" if report.get("stopped") else None
            _append_log("批次结束" if not report.get("stopped") else "批次已停止")
            if report.get("stopped"):
                _batch["status"] = "idle"
    except Exception as e:
        with _batch_lock:
            _batch["status"] = "error"
            _batch["error"] = str(e)
            _append_log(f"出错：{e}")


def _batch_snapshot() -> dict:
    return _snapshot()["batch"]


app = FastAPI(title="本地视觉助手")
app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(UI_DIR / "index.html")


@app.get("/api/status")
def status():
    return _snapshot()


@app.post("/api/model/start")
def model_start():
    with _ctrl:
        if _state["status"] == "ready":
            return _snapshot()
        if _state["status"] in ("loading", "stopping"):
            raise HTTPException(409, "模型正在切换，请稍等")
        _state["status"] = "loading"
        _state["error"] = None
        threading.Thread(target=_load_model, daemon=True).start()
    return _snapshot()


@app.post("/api/model/stop")
def model_stop():
    with _batch_lock:
        if _batch["status"] in ("running", "stopping"):
            raise HTTPException(409, "批量打标进行中，请先停止批次")
    with _ctrl:
        if _state["status"] == "stopped":
            return _snapshot()
        if _state["status"] in ("loading", "stopping"):
            raise HTTPException(409, "模型正在切换，请稍等")
        if _state["status"] == "error":
            _state["status"] = "stopped"
            _state["error"] = None
            return _snapshot()
        _unload_model()
    return _snapshot()


@app.post("/api/chat")
def chat(req: ChatIn):
    with _batch_lock:
        if _batch["status"] in ("running", "stopping"):
            raise HTTPException(409, "批量打标进行中，请稍后再聊")
    if _state["status"] != "ready":
        raise HTTPException(503, "请先在页面上启动模型")
    if not req.image_b64:
        raise HTTPException(400, "请先放一张图片")

    try:
        image = _decode_image(req.image_b64)
    except Exception as e:
        raise HTTPException(400, f"图片读不出来：{e}")

    from mlx_vlm.generate import stream_generate
    from mlx_vlm.prompt_utils import apply_chat_template

    max_tokens = req.max_tokens or (128 if req.mode == "tag" else 512)
    temperature = 0.0 if req.mode == "tag" else 0.2
    prompt = apply_chat_template(
        _state["processor"], _state["config"], _messages(req), num_images=1
    )

    def event_stream():
        with _lock:
            if _state["status"] != "ready":
                yield f"data: {json.dumps({'t': '模型已停止'}, ensure_ascii=False)}\n\n"
                yield "data: {\"done\": true}\n\n"
                return
            for result in stream_generate(
                _state["model"],
                _state["processor"],
                prompt,
                image,
                max_tokens=max_tokens,
                temperature=temperature,
                max_kv_size=4096,
                verbose=False,
            ):
                text = getattr(result, "text", "") or ""
                if text:
                    yield f"data: {json.dumps({'t': text}, ensure_ascii=False)}\n\n"
            yield "data: {\"done\": true}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/settings")
def get_settings():
    settings = load_settings()
    vocab = load_active()
    return {**settings, "leaf_count": vocab["leaf_count"], "vocab_path": vocab["vocab_path"]}


@app.post("/api/settings")
def post_settings(req: SettingsIn):
    return save_settings(req.model_dump())


@app.get("/api/vocab")
def get_vocab():
    vocab = load_active()
    return {
        "groups": vocab["groups"],
        "leaf_count": vocab["leaf_count"],
        "vocab_path": vocab["vocab_path"],
        "custom": bool(vocab["vocab_path"]),
    }


@app.post("/api/vocab")
def post_vocab(req: VocabIn):
    try:
        groups = save_groups(req.groups)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    vocab = load_active()
    return {"groups": groups, "leaf_count": vocab["leaf_count"], "custom": True}


@app.post("/api/vocab/reset")
def vocab_reset():
    groups = reset_groups()
    vocab = load_active()
    return {"groups": groups, "leaf_count": vocab["leaf_count"], "custom": False}


@app.get("/api/folders/suggest")
def folders_suggest():
    return {"items": _suggest_folders()}


@app.post("/api/folder/scan")
def folder_scan(req: ScanIn):
    try:
        save_settings({"folder": req.folder, "out": req.out, "resume": req.resume})
        return _scan_folder(req.folder, req.out, req.resume)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.post("/api/batch/start")
def batch_start(req: BatchStartIn):
    with _batch_lock:
        if _batch["status"] in ("running", "stopping"):
            raise HTTPException(409, "已有批次在跑")
        try:
            scan = _scan_folder(req.folder, req.out, req.resume)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        if scan["n_images"] == 0:
            raise HTTPException(400, "这个文件夹里没有图片")
        if req.resume and scan["n_todo"] == 0:
            raise HTTPException(400, "按断点续跑已全部完成，没有待打图片")
        save_settings(req.model_dump())
        _batch.update(
            {
                "status": "running",
                "error": None,
                "folder": scan["folder"],
                "out": scan["out"],
                "total": scan["n_todo"] if req.resume else scan["n_images"],
                "done": 0,
                "skipped": scan["n_done"] if req.resume else 0,
                "current": None,
                "last": None,
                "stats": {},
                "started_at": time.strftime("%H:%M:%S"),
                "report": None,
                "cancel": False,
                "jsonl": scan["jsonl"] or str(Path(scan["out"]) / "results.jsonl"),
            }
        )
        _batch["log"].clear()
        todo = scan["n_todo"] if req.resume else scan["n_images"]
        _append_log(f"开始批量：{scan['folder']}")
        _append_log(f"图片 {scan['n_images']} 张，待打 {todo}，续跑跳过 {scan['n_done'] if req.resume else 0}")
        _append_log(f"结果写入 {scan['out']}")
    threading.Thread(
        target=_run_batch_job,
        args=(req.folder, req.out or scan["out"], req.resume, req.limit),
        daemon=True,
    ).start()
    return _batch_snapshot()


@app.post("/api/batch/stop")
def batch_stop():
    with _batch_lock:
        if _batch["status"] != "running":
            return _batch_snapshot()
        _batch["cancel"] = True
        _batch["status"] = "stopping"
        _append_log("正在停止，当前这张打完就会停")
    return _batch_snapshot()


@app.get("/api/batch/status")
def batch_status():
    return _batch_snapshot()


@app.post("/api/export/csv")
def export_csv(req: ScanIn):
    try:
        out_dir = _resolve_dir(req.out, must_exist=False) if (req.out or "").strip() else None
        if out_dir is None:
            settings = load_settings()
            out_dir = _resolve_dir(settings["out"], must_exist=False)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    jsonl = out_dir / "results.jsonl"
    if not jsonl.is_file():
        raise HTTPException(400, f"还没有结果文件：{jsonl}")
    from import_billfish import BillfishError, run_import

    csv_path = out_dir / "billfish_tags.csv"
    try:
        result = run_import(jsonl, None, csv_path, apply=False)
    except BillfishError as e:
        raise HTTPException(400, str(e)) from e
    return {"csv": result.get("csv") or str(csv_path), "n": result.get("n_rows") or 0}


@app.get("/api/billfish/suggest")
def billfish_suggest(folder: str = ""):
    return {"items": _suggest_billfish_dbs(folder)}


@app.post("/api/billfish/preview")
def billfish_preview(req: BillfishIn):
    return _billfish_run(req, apply=False)


@app.post("/api/billfish/apply")
def billfish_apply(req: BillfishIn):
    return _billfish_run(req, apply=True)


def _billfish_run(req: BillfishIn, apply: bool) -> dict:
    import sqlite3

    from import_billfish import BillfishError, run_import

    settings = load_settings()
    try:
        if (req.out or "").strip():
            out_dir = _resolve_dir(req.out, must_exist=False)
        else:
            out_dir = _resolve_dir(settings["out"], must_exist=False)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    jsonl = Path(req.jsonl).expanduser() if (req.jsonl or "").strip() else (out_dir / "results.jsonl")
    if not jsonl.is_absolute():
        jsonl = (WORKSPACE / jsonl).resolve()
    db_raw = (req.db or settings.get("billfish_db") or "").strip()
    if not db_raw:
        raise HTTPException(400, "请填写 Billfish 数据库路径（素材库\\.bf\\billfish.db）")
    db = Path(db_raw).expanduser()
    if not db.is_absolute():
        db = (WORKSPACE / db).resolve()
    save_settings({"out": str(out_dir), "billfish_db": str(db)})
    try:
        result = run_import(
            jsonl,
            db,
            out_dir / "billfish_tags.csv",
            apply=apply,
            strip_old=req.strip_old,
            create_missing=req.create_missing,
            replace=req.replace,
        )
    except BillfishError as e:
        raise HTTPException(400, str(e)) from e
    except sqlite3.Error as e:
        raise HTTPException(400, f"数据库打不开或正在被占用（请先退出 Billfish）：{e}") from e
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--preload", action="store_true", help="启动网页时同时加载模型")
    args = parser.parse_args()

    if args.preload:
        with _ctrl:
            _state["status"] = "loading"
        threading.Thread(target=_load_model, daemon=True).start()

    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    import uvicorn

    print(f"本地聊天页: {url}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
