#!/usr/bin/env python3
"""本地图片中文打标：最长边 768 仅用于推理，不改原图。"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PRIMARY_MODEL = "mlx-community/Qwen3-VL-8B-NSFW-Caption-V4.5-mxfp4"
FALLBACK_MODEL = "mlx-community/huihui-Qwen3-VL-2B-Instruct-ab-4bit"

from billfish_vocab import ALLOWED_TAGS, SYSTEM_PROMPT

try:
    from vocab_store import load_active as _load_active_vocab
except Exception:  # pragma: no cover
    _load_active_vocab = None

USER_PROMPT = "给这张图打标签。只输出 JSON。tags 只能 1 到 4 个。"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
JSON_RE = re.compile(r"\{.*\}", re.S)
FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)
REFUSAL_RE = re.compile(
    r"(抱歉|无法|不能|拒绝|对不起|sorry|cannot|can't|unable|refuse|nsfw|policy)",
    re.I,
)


def resize_max_edge(img, max_edge: int = 768):
    from PIL import Image

    img = img.convert("RGB")
    w, h = img.size
    longest = max(w, h)
    if longest <= max_edge:
        return img, (w, h)
    scale = max_edge / float(longest)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return img.resize(new_size, Image.Resampling.LANCZOS), (w, h)


def active_vocab() -> dict:
    if _load_active_vocab is not None:
        try:
            return _load_active_vocab()
        except Exception:
            pass
    return {"allowed": ALLOWED_TAGS, "prompt": SYSTEM_PROMPT}


def parse_json_output(
    raw: str, allowed_tags: set[str] | None = None
) -> tuple[dict | None, str, list[str]]:
    """Return (obj, status, oov_tags). status: ok / skip / non_json / refusal / bad_schema."""
    allowed = allowed_tags if allowed_tags is not None else active_vocab()["allowed"]
    text = (raw or "").strip()
    if not text:
        return None, "non_json", []
    cleaned = FENCE_RE.sub("", text).strip()
    obj = None
    match = JSON_RE.search(cleaned)
    if match:
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            obj = None
    if obj is None:
        obj = _salvage_truncated_tags(cleaned)
    if obj is None:
        if REFUSAL_RE.search(text):
            return None, "refusal", []
        return None, "non_json", []
    if not isinstance(obj, dict) or "tags" not in obj or not isinstance(obj["tags"], list):
        return None, "bad_schema", []
    tags = []
    for t in obj["tags"]:
        if isinstance(t, str) and t not in tags:
            tags.append(t)
    obj["tags"] = tags
    if tags == ["跳过"]:
        return obj, "skip", []
    tags = _clamp_tags(tags, allowed)
    obj["tags"] = tags
    if not tags:
        return obj, "bad_schema", []
    oov = [t for t in tags if t not in allowed]
    extra = cleaned[match.end() :].strip() if match else ""
    if extra and REFUSAL_RE.search(text):
        return obj, "refusal", oov
    return obj, "ok", oov


def _salvage_truncated_tags(text: str) -> dict | None:
    m = re.search(r'\{\s*"tags"\s*:\s*\[(.*)$', text, re.S)
    if not m:
        return None
    tags = []
    for t in re.findall(r'"([^"]+)"', m.group(1)):
        if t != "tags" and t not in tags:
            tags.append(t)
    if not tags:
        return None
    return {"tags": tags}


def _clamp_tags(tags: list[str], allowed_tags: set[str] | None = None) -> list[str]:
    allowed = allowed_tags if allowed_tags is not None else ALLOWED_TAGS
    out = []
    for t in tags:
        if t == "跳过":
            continue
        if t in allowed and t not in out:
            out.append(t)
        if len(out) >= 4:
            break
    return out


def collect_images(folder: Path) -> list[Path]:
    files = [
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith(".")
    ]
    return sorted(files)


def swap_usage() -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True).strip()
    except Exception:
        return "unknown"


def memory_free_pct() -> str:
    try:
        out = subprocess.check_output(["memory_pressure"], text=True)
        for line in out.splitlines():
            if "free percentage" in line.lower():
                return line.strip()
        return out.strip().splitlines()[-1]
    except Exception:
        return "unknown"


def mlx_peak_gb() -> float:
    try:
        import mlx.core as mx

        return mx.get_peak_memory() / (1024**3)
    except Exception:
        return 0.0


def rss_gb() -> float:
    try:
        import resource

        # ru_maxrss is bytes on macOS
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)
    except Exception:
        return 0.0


def download_model(model_id: str) -> str:
    from huggingface_hub import snapshot_download

    print(f"[download] {model_id} via {os.environ.get('HF_ENDPOINT')}", flush=True)
    path = snapshot_download(repo_id=model_id)
    print(f"[download] cached at {path}", flush=True)
    return path


def load_vlm(model_id: str):
    from mlx_vlm import load
    from mlx_vlm.utils import load_config

    print(f"[load] {model_id}", flush=True)
    t0 = time.perf_counter()
    model, processor = load(model_id)
    config = load_config(model_id)
    dt = time.perf_counter() - t0
    print(
        f"[load] done in {dt:.1f}s | mlx_peak={mlx_peak_gb():.2f}GB | "
        f"rss_max={rss_gb():.2f}GB | swap={swap_usage()} | {memory_free_pct()}",
        flush=True,
    )
    return model, processor, config, dt


def should_fallback(load_seconds: float) -> bool:
    """仅在 8B 加载后明显狂换页时才降到 2B。轻换页（Cursor 占内存）不算。"""
    free = memory_free_pct()
    swap = swap_usage()
    print(f"[mem-check] {free} | {swap} | load={load_seconds:.1f}s", flush=True)
    used_mb = 0.0
    if "used =" in swap:
        try:
            used_s = swap.split("used =")[1].split()[0]
            used_mb = float(used_s.replace("M", "").replace("G", ""))
            if "G" in used_s:
                used_mb *= 1024
        except Exception:
            used_mb = 0.0
    free_n = 100.0
    try:
        if "%" in free:
            free_n = float(free.split(":")[-1].replace("%", "").strip())
    except Exception:
        pass
    # 16GB 机器加载 5GB 权重后有 1GB 级 swap 很常见；狂换页才回退。
    return used_mb >= 4096 or free_n <= 5 or load_seconds >= 90


def generate_tags(
    model,
    processor,
    config,
    image,
    max_tokens: int,
    max_kv_size: int,
    system_prompt: str | None = None,
):
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    messages = [
        {"role": "system", "content": system_prompt or active_vocab()["prompt"]},
        {"role": "user", "content": USER_PROMPT},
    ]
    prompt = apply_chat_template(processor, config, messages, num_images=1)
    result = generate(
        model,
        processor,
        prompt,
        image,
        max_tokens=max_tokens,
        temperature=0.0,
        max_kv_size=max_kv_size,
        repetition_penalty=1.2,
        verbose=False,
    )
    text = result.text if hasattr(result, "text") else str(result)
    return text, result


def sidecar_path(out_dir: Path, src_root: Path, image_path: Path) -> Path:
    rel = image_path.relative_to(src_root)
    dest = out_dir / rel.with_suffix(".json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest


def sidecar_complete(side: Path) -> bool:
    if not side.is_file():
        return False
    try:
        prev = json.loads(side.read_text(encoding="utf-8"))
    except Exception:
        return False
    tags = prev.get("tags") if isinstance(prev, dict) else None
    return (
        isinstance(tags, list)
        and not prev.get("error")
        and (tags == ["跳过"] or 1 <= len(tags) <= 4)
    )


def pending_images(images: list[Path], out_dir: Path, src_root: Path) -> tuple[list[Path], int]:
    pending = []
    skipped = 0
    for p in images:
        if sidecar_complete(sidecar_path(out_dir, src_root, p)):
            skipped += 1
        else:
            pending.append(p)
    return pending, skipped


def process_image(
    model,
    processor,
    config,
    path: Path,
    src_root: Path,
    index: int,
    max_edge: int,
    max_tokens: int,
    max_kv_size: int,
    system_prompt: str,
    allowed_tags: set[str],
) -> dict:
    from PIL import Image

    rec = {
        "file": str(path),
        "rel": str(path.relative_to(src_root)),
        "index": index,
    }
    t0 = time.perf_counter()
    try:
        with Image.open(path) as im:
            resized, orig_size = resize_max_edge(im, max_edge)
        rec["orig_size"] = list(orig_size)
        rec["infer_size"] = list(resized.size)
        raw, result = generate_tags(
            model,
            processor,
            config,
            resized,
            max_tokens,
            max_kv_size,
            system_prompt=system_prompt,
        )
        rec["raw"] = raw
        rec["prompt_tokens"] = getattr(result, "prompt_tokens", None)
        rec["generation_tokens"] = getattr(result, "generation_tokens", None)
        rec["mlx_peak_gb"] = round(getattr(result, "peak_memory", 0.0) or mlx_peak_gb(), 3)
        obj, status, oov = parse_json_output(raw, allowed_tags)
        rec["status"] = status
        rec["parsed"] = obj
        rec["oov_tags"] = oov
        rec["tags"] = obj.get("tags", []) if obj else []
        rec["tags_clean"] = [t for t in rec["tags"] if t in allowed_tags and t != "跳过"][:4]
    except Exception as e:
        rec["status"] = "error"
        rec["error"] = repr(e)
        rec["raw"] = ""
        rec["tags"] = []
        rec["tags_clean"] = []
        rec["oov_tags"] = []
    rec["seconds"] = round(time.perf_counter() - t0, 2)
    return rec


def write_outputs(rec: dict, sidecar: Path, jsonl_path: Path) -> None:
    if rec.get("status") == "skip":
        payload = {"tags": ["跳过"]}
    else:
        payload = {"tags": rec.get("tags_clean") or rec.get("tags") or []}
        if rec.get("status") not in ("ok", "skip"):
            payload["error"] = rec.get("status")
    sidecar.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_loaded_batch(
    model,
    processor,
    config,
    *,
    src_root: Path,
    out_dir: Path,
    images: list[Path],
    resume: bool = True,
    max_edge: int = 768,
    max_tokens: int = 128,
    max_kv_size: int = 4096,
    system_prompt: str | None = None,
    allowed_tags: set[str] | None = None,
    on_progress=None,
    should_stop=None,
    infer_lock=None,
) -> dict:
    vocab = active_vocab()
    prompt = system_prompt or vocab["prompt"]
    allowed = allowed_tags if allowed_tags is not None else vocab["allowed"]
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "results.jsonl"
    report_path = out_dir / "run_report.json"
    skipped = 0
    if resume:
        images, skipped = pending_images(images, out_dir, src_root)
        if not jsonl_path.exists():
            jsonl_path.write_text("", encoding="utf-8")
    else:
        jsonl_path.write_text("", encoding="utf-8")

    stats = Counter()
    times = []
    peak_after_load = mlx_peak_gb()
    peak_seen = peak_after_load
    stopped = False
    total = len(images)

    for i, path in enumerate(images, 1):
        if should_stop and should_stop():
            stopped = True
            break
        if infer_lock:
            infer_lock.acquire()
        try:
            rec = process_image(
                model,
                processor,
                config,
                path,
                src_root,
                i,
                max_edge,
                max_tokens,
                max_kv_size,
                prompt,
                allowed,
            )
        finally:
            if infer_lock:
                infer_lock.release()
        stats[rec.get("status") or "error"] += 1
        if rec.get("oov_tags"):
            stats["oov"] += 1
        times.append(rec["seconds"])
        peak_seen = max(peak_seen, mlx_peak_gb(), rec.get("mlx_peak_gb") or 0)
        write_outputs(rec, sidecar_path(out_dir, src_root, path), jsonl_path)
        print(
            f"[{i}/{total}] {rec['seconds']:.1f}s {rec.get('status')} "
            f"{path.name} -> {rec.get('tags')}",
            flush=True,
        )
        if on_progress:
            on_progress(rec, i, total, skipped)

    avg = sum(times) / len(times) if times else 0
    report = {
        "started": datetime.now(timezone.utc).isoformat(),
        "n_images": total,
        "skipped_resume": skipped,
        "processed": len(times),
        "stopped": stopped,
        "avg_seconds": round(avg, 2),
        "min_seconds": round(min(times), 2) if times else None,
        "max_seconds": round(max(times), 2) if times else None,
        "mlx_peak_gb_after_load": round(peak_after_load, 3),
        "mlx_peak_gb_seen": round(peak_seen, 3),
        "rss_max_gb": round(rss_gb(), 3),
        "swap": swap_usage(),
        "memory": memory_free_pct(),
        "stats": dict(stats),
        "jsonl": str(jsonl_path),
        "out": str(out_dir),
        "folder": str(src_root),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="本地视觉模型批量中文打标")
    parser.add_argument("--folder", required=True, help="样本图文件夹")
    parser.add_argument("--out", default="outputs", help="结果目录（不写回原图）")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 张，0 表示全部")
    parser.add_argument("--max-edge", type=int, default=768)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-kv-size", type=int, default=4096)
    parser.add_argument("--model", default=PRIMARY_MODEL)
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--resume", action="store_true", help="跳过已有 sidecar 的图片，并追加 jsonl")
    args = parser.parse_args()

    src_root = Path(args.folder).expanduser().resolve()
    if not src_root.is_dir():
        print(f"找不到文件夹: {src_root}", file=sys.stderr)
        return 2
    images = collect_images(src_root)
    if args.limit and args.limit > 0:
        images = images[: args.limit]
    if not images:
        print(f"文件夹里没有图片: {src_root}", file=sys.stderr)
        return 2

    out_dir = Path(args.out).expanduser().resolve()
    print(f"[info] images={len(images)} from {src_root}", flush=True)
    print(f"[info] out={out_dir}", flush=True)
    print(f"[info] before: {memory_free_pct()} | swap={swap_usage()}", flush=True)

    model_id = args.model
    download_model(model_id)
    model, processor, config, load_s = load_vlm(model_id)
    used_fallback = False
    if args.allow_fallback and model_id == PRIMARY_MODEL and should_fallback(load_s):
        print("[warn] 8B 加载后内存过紧，改用 2B abliterated 4-bit", flush=True)
        del model, processor, config
        try:
            import mlx.core as mx
            import gc

            gc.collect()
            mx.clear_cache()
        except Exception:
            pass
        model_id = FALLBACK_MODEL
        download_model(model_id)
        model, processor, config, load_s = load_vlm(model_id)
        used_fallback = True

    vocab = active_vocab()
    report = run_loaded_batch(
        model,
        processor,
        config,
        src_root=src_root,
        out_dir=out_dir,
        images=images,
        resume=args.resume,
        max_edge=args.max_edge,
        max_tokens=args.max_tokens,
        max_kv_size=args.max_kv_size,
        system_prompt=vocab["prompt"],
        allowed_tags=vocab["allowed"],
    )
    report["model"] = model_id
    report["used_fallback"] = used_fallback
    report["load_seconds"] = round(load_s, 2)
    (out_dir / "run_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
