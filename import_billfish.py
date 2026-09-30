#!/usr/bin/env python3
"""把打标 JSONL 写入 Billfish SQLite。

只依赖标准库 + billfish_vocab / vocab_store，可在 Windows 上单独跑。
默认先预览，加 --apply 才写入；写入前会备份 billfish.db。
务必先关闭 Billfish。
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

from billfish_vocab import OLD_ROOT_TAGS
from vocab_store import load_active


class BillfishError(Exception):
    pass


def current_vocab() -> dict:
    return load_active()


def load_jsonl(path: Path, parent_of: dict[str, str] | None = None) -> list[dict]:
    allowed = parent_of if parent_of is not None else current_vocab()["parent_of"]
    latest: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tags = [t for t in rec.get("tags") or [] if t and t != "跳过" and t in allowed]
            name = Path(rec.get("file") or rec.get("rel") or "").name
            if not name:
                continue
            latest[name] = {"filename": name, "tags": tags, "rel": rec.get("rel")}
    return list(latest.values())


def write_csv(rows: list[dict], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "tags"])
        for r in rows:
            w.writerow([r["filename"], ",".join(r["tags"])])


def detect(conn: sqlite3.Connection) -> dict:
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "bf_file" not in tables or "bf_tag_join_file" not in tables:
        raise BillfishError(f"不像 Billfish 库，现有表：{sorted(tables)[:20]}")
    tag_table = "bf_tag_v2" if "bf_tag_v2" in tables else ("bf_tag" if "bf_tag" in tables else None)
    if not tag_table:
        raise BillfishError("找不到 bf_tag / bf_tag_v2")
    tag_cols = {r[1] for r in conn.execute(f"PRAGMA table_info({tag_table})")}
    file_cols = {r[1] for r in conn.execute("PRAGMA table_info(bf_file)")}
    join_cols = {r[1] for r in conn.execute("PRAGMA table_info(bf_tag_join_file)")}
    return {
        "tag_table": tag_table,
        "has_pid": "pid" in tag_cols,
        "tag_cols": tag_cols,
        "file_cols": file_cols,
        "join_cols": join_cols,
    }


def norm_name(name: str) -> str:
    n = (name or "").replace("\\", "/").split("/")[-1].lower()
    if n.endswith(".lnk"):
        n = n[:-4]
    return n


def file_index(conn: sqlite3.Connection) -> dict[str, list[int]]:
    idx: dict[str, list[int]] = {}
    for fid, name in conn.execute("SELECT id, name FROM bf_file"):
        key = norm_name(name or "")
        if not key:
            continue
        idx.setdefault(key, []).append(int(fid))
    return idx


def find_tag(conn: sqlite3.Connection, tag_table: str, name: str, pid: int | None) -> int | None:
    if pid is not None:
        row = conn.execute(
            f"SELECT id FROM {tag_table} WHERE name=? AND pid=?", (name, pid)
        ).fetchone()
        if row:
            return int(row[0])
    rows = conn.execute(f"SELECT id, pid FROM {tag_table} WHERE name=?", (name,)).fetchall()
    if not rows:
        return None
    if pid is not None:
        for r in rows:
            if int(r[1] or 0) == int(pid):
                return int(r[0])
    if len(rows) == 1:
        return int(rows[0][0])
    return int(rows[0][0])


def insert_tag(
    conn: sqlite3.Connection, tag_table: str, cols: set[str], name: str, pid: int
) -> int:
    fields: dict[str, object] = {"name": name}
    if "pid" in cols:
        fields["pid"] = int(pid or 0)
    if "seq" in cols:
        fields["seq"] = 0
    if "icon" in cols:
        fields["icon"] = 0
    if "color" in cols:
        fields["color"] = 0
    if "hide" in cols:
        fields["hide"] = 0
    if "born" in cols:
        fields["born"] = int(time.time())
    keys = [k for k in fields if k in cols or k == "name"]
    placeholders = ",".join("?" * len(keys))
    conn.execute(
        f"INSERT INTO {tag_table} ({','.join(keys)}) VALUES ({placeholders})",
        tuple(fields[k] for k in keys),
    )
    return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])


def ensure_tag(
    conn: sqlite3.Connection,
    tag_table: str,
    cols: set[str],
    name: str,
    pid: int,
    create: bool,
) -> int | None:
    found = find_tag(conn, tag_table, name, pid if "pid" in cols else None)
    if found is not None:
        return found
    if not create:
        return None
    return insert_tag(conn, tag_table, cols, name, pid)


def descendants(conn: sqlite3.Connection, tag_table: str, root_ids: set[int]) -> set[int]:
    kids = {int(r[0]): int(r[1] or 0) for r in conn.execute(f"SELECT id, pid FROM {tag_table}")}
    out = set(root_ids)
    changed = True
    while changed:
        changed = False
        for tid, pid in kids.items():
            if pid in out and tid not in out:
                out.add(tid)
                changed = True
    return out


def strip_old_tree(conn: sqlite3.Connection, tag_table: str, file_ids: set[int] | None) -> tuple[int, int]:
    roots = []
    for name in OLD_ROOT_TAGS:
        rows = conn.execute(f"SELECT id FROM {tag_table} WHERE name=?", (name,)).fetchall()
        roots.extend(int(r[0]) for r in rows)
    if not roots:
        return 0, 0
    old_ids = descendants(conn, tag_table, set(roots))
    if not old_ids:
        return 0, 0
    if file_ids:
        q = ",".join("?" * len(file_ids))
        cur = conn.execute(
            f"DELETE FROM bf_tag_join_file WHERE tag_id IN ({','.join('?'*len(old_ids))}) "
            f"AND file_id IN ({q})",
            (*old_ids, *file_ids),
        )
    else:
        cur = conn.execute(
            f"DELETE FROM bf_tag_join_file WHERE tag_id IN ({','.join('?'*len(old_ids))})",
            tuple(old_ids),
        )
    unlinked = cur.rowcount if cur.rowcount is not None else 0
    still = {
        int(r[0])
        for r in conn.execute(
            f"SELECT DISTINCT tag_id FROM bf_tag_join_file WHERE tag_id IN ({','.join('?'*len(old_ids))})",
            tuple(old_ids),
        )
    }
    deletable = [i for i in old_ids if i not in still]
    deleted = 0
    if deletable:
        conn.execute(
            f"DELETE FROM {tag_table} WHERE id IN ({','.join('?'*len(deletable))})",
            tuple(deletable),
        )
        deleted = len(deletable)
    return unlinked, deleted


def already_linked(conn: sqlite3.Connection, file_id: int, tag_id: int) -> bool:
    row = conn.execute(
        "SELECT 1 FROM bf_tag_join_file WHERE file_id=? AND tag_id=? LIMIT 1",
        (file_id, tag_id),
    ).fetchone()
    return row is not None


def insert_join(conn: sqlite3.Connection, join_cols: set[str], file_id: int, tag_id: int) -> None:
    fields = {"file_id": file_id, "tag_id": tag_id}
    if "born" in join_cols:
        fields["born"] = int(time.time())
    keys = [k for k in fields if k in join_cols]
    if "file_id" not in keys:
        keys = ["file_id", "tag_id"]
    placeholders = ",".join("?" * len(keys))
    conn.execute(
        f"INSERT INTO bf_tag_join_file ({','.join(keys)}) VALUES ({placeholders})",
        tuple(fields[k] for k in keys),
    )


def ensure_tree(
    conn: sqlite3.Connection,
    tag_table: str,
    cols: set[str],
    groups: list[dict],
    create: bool,
) -> dict[str, int | None]:
    """返回 分类名/叶子名 -> tag id。"""
    ids: dict[str, int | None] = {}
    for g in groups:
        gid = ensure_tag(conn, tag_table, cols, g["name"], 0, create)
        ids[g["name"]] = gid
        parent_id = gid if gid is not None else 0
        for t in g.get("tags") or []:
            ids[t] = ensure_tag(conn, tag_table, cols, t, parent_id, create)
        for child in g.get("children") or []:
            cid = ensure_tag(conn, tag_table, cols, child["name"], parent_id, create)
            ids[child["name"]] = cid
            cpid = cid if cid is not None else parent_id
            for t in child.get("tags") or []:
                ids[t] = ensure_tag(conn, tag_table, cols, t, cpid, create)
    return ids


def resolve_parent_id(
    conn,
    tag_table: str,
    child: str,
    cache: dict[str, int | None],
    parent_of: dict[str, str],
    parent_parent: dict[str, str],
) -> int | None:
    parent = parent_of[child]
    if parent in cache:
        return cache[parent]
    grand = parent_parent.get(parent)
    grand_id = None
    if grand:
        grand_id = find_tag(conn, tag_table, grand, 0)
        if grand_id is None:
            grand_id = find_tag(conn, tag_table, grand, None)
    pid = find_tag(conn, tag_table, parent, grand_id)
    if pid is None:
        pid = find_tag(conn, tag_table, parent, None)
    cache[parent] = pid
    return pid


def unlink_vocab_tags(
    conn: sqlite3.Connection,
    tag_table: str,
    file_ids: set[int],
    leaf_names: set[str],
) -> int:
    if not file_ids or not leaf_names:
        return 0
    names = sorted(leaf_names)
    tag_ids = [
        int(r[0])
        for r in conn.execute(
            f"SELECT id FROM {tag_table} WHERE name IN ({','.join('?'*len(names))})",
            names,
        )
    ]
    if not tag_ids:
        return 0
    cur = conn.execute(
        f"DELETE FROM bf_tag_join_file WHERE file_id IN ({','.join('?'*len(file_ids))}) "
        f"AND tag_id IN ({','.join('?'*len(tag_ids))})",
        (*file_ids, *tag_ids),
    )
    return cur.rowcount if cur.rowcount is not None else 0


def match_rows(rows: list[dict], files: dict[str, list[int]]) -> dict:
    matched = missed = multi = 0
    preview = []
    file_ids: set[int] = set()
    matched_rows = []
    for rec in rows:
        ids = files.get(norm_name(rec["filename"]), [])
        if not ids:
            missed += 1
            preview.append(f"未匹配  {rec['filename']}")
            continue
        if len(ids) > 1:
            multi += 1
        matched += 1
        file_ids.update(ids)
        matched_rows.append({**rec, "file_ids": ids})
        preview.append(f"匹配×{len(ids)}  {rec['filename']}  -> {','.join(rec['tags'])}")
    return {
        "matched": matched,
        "missed": missed,
        "multi": multi,
        "preview": preview,
        "file_ids": file_ids,
        "matched_rows": matched_rows,
    }


def run_import(
    jsonl: Path,
    db: Path | None = None,
    csv_path: Path | None = None,
    *,
    apply: bool = False,
    strip_old: bool = False,
    strip_old_all: bool = False,
    create_missing: bool = False,
    replace: bool = False,
) -> dict:
    jsonl = Path(jsonl).expanduser()
    if not jsonl.is_file():
        raise BillfishError(f"找不到结果文件：{jsonl}")
    vocab = current_vocab()
    rows = load_jsonl(jsonl, vocab["parent_of"])
    tagged = sum(1 for r in rows if r["tags"])
    result: dict = {
        "jsonl": str(jsonl),
        "n_rows": len(rows),
        "n_tagged": tagged,
        "n_empty": len(rows) - tagged,
        "roots": list(g["name"] for g in vocab["groups"]),
        "apply": apply,
        "csv": None,
        "backup": None,
        "log": [],
    }
    if csv_path:
        csv_path = Path(csv_path).expanduser()
        write_csv(rows, csv_path)
        result["csv"] = str(csv_path)
        result["log"].append(f"清单 {len(rows)} 张 -> {csv_path}")

    if db is None:
        result["log"].append("未指定 Billfish 数据库，只导出了 CSV。")
        return result

    db_path = Path(db).expanduser()
    if not db_path.is_file():
        raise BillfishError(f"找不到数据库：{db_path}")
    result["db"] = str(db_path)

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA foreign_keys=OFF")
        info = detect(conn)
        files = file_index(conn)
        tag_table = info["tag_table"]
        match = match_rows(rows, files)
        result.update(
            {
                "tag_table": tag_table,
                "library_files": sum(len(v) for v in files.values()),
                "matched": match["matched"],
                "missed": match["missed"],
                "multi": match["multi"],
                "preview": match["preview"][:80],
                "preview_more": max(0, len(match["preview"]) - 80),
            }
        )
        result["log"].append(
            f"库内文件 {result['library_files']}，匹配 {match['matched']}，未匹配 {match['missed']}，重名 {match['multi']}"
        )

        if not apply:
            result["log"].append("这是预览。确认后加 --apply / 网页点「写入库」才会改数据库。")
            result["log"].append("写入前请先彻底退出 Billfish。")
            return result

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = db_path.with_name(db_path.name + f".bak-{stamp}")
        shutil.copy2(db_path, backup)
        result["backup"] = str(backup)
        result["log"].append(f"已备份 {backup}")

        file_ids = match["file_ids"]
        if strip_old or strip_old_all:
            unlinked, deleted = strip_old_tree(
                conn, tag_table, None if strip_old_all else file_ids
            )
            result["strip_unlinked"] = unlinked
            result["strip_deleted"] = deleted
            result["log"].append(f"剥离旧树：解除关联 {unlinked}，删除空标签 {deleted}")

        if replace:
            removed = unlink_vocab_tags(conn, tag_table, file_ids, set(vocab["parent_of"]))
            result["replaced"] = removed
            result["log"].append(f"已清掉这批文件上的旧词表标签 {removed} 条")

        tree_ids = ensure_tree(
            conn, tag_table, info["tag_cols"], vocab["groups"], create_missing
        )
        parent_cache: dict[str, int | None] = {}
        new_links = missing_tags = 0
        missing_names: set[str] = set()
        for rec in match["matched_rows"]:
            for file_id in rec["file_ids"]:
                for tag in rec["tags"]:
                    tag_id = tree_ids.get(tag)
                    if tag_id is None:
                        pid = resolve_parent_id(
                            conn,
                            tag_table,
                            tag,
                            parent_cache,
                            vocab["parent_of"],
                            vocab["parent_parent"],
                        )
                        tag_id = find_tag(conn, tag_table, tag, pid)
                        if tag_id is None and create_missing:
                            tag_id = insert_tag(
                                conn, tag_table, info["tag_cols"], tag, pid or 0
                            )
                            tree_ids[tag] = tag_id
                    if tag_id is None:
                        missing_tags += 1
                        missing_names.add(tag)
                        continue
                    if already_linked(conn, file_id, tag_id):
                        continue
                    insert_join(conn, info["join_cols"], file_id, tag_id)
                    new_links += 1
        conn.commit()
        result["new_links"] = new_links
        result["missing_tags"] = missing_tags
        result["missing_names"] = sorted(missing_names)
        result["log"].append(
            f"新写入关联 {new_links} 条。库中缺失未创建的标签 {missing_tags}"
            + (f"：{sorted(missing_names)}" if missing_names else "")
        )
        result["log"].append("写完后重新打开 Billfish 即可看到标签。")
        return result
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="把本地打标 JSON 写入 Billfish 标签树")
    parser.add_argument("--jsonl", default="outputs/web_batch/results.jsonl")
    parser.add_argument("--db", help=r"Billfish 库路径，例如 D:\图库\.bf\billfish.db")
    parser.add_argument("--csv", default="outputs/billfish_tags.csv")
    parser.add_argument("--apply", action="store_true", help="真正写入；默认只预览")
    parser.add_argument("--strip-old", action="store_true", help="剥掉旧的媒介/分级树")
    parser.add_argument("--strip-old-all", action="store_true", help="剥旧树时不限这批文件")
    parser.add_argument("--create-missing", action="store_true", help="库里没有的分类/叶子也新建")
    parser.add_argument("--replace", action="store_true", help="先去掉这批文件上已有的词表标签再写入")
    args = parser.parse_args()

    try:
        result = run_import(
            Path(args.jsonl),
            Path(args.db) if args.db else None,
            Path(args.csv),
            apply=args.apply,
            strip_old=args.strip_old,
            strip_old_all=args.strip_old_all,
            create_missing=args.create_missing,
            replace=args.replace,
        )
    except BillfishError as e:
        print(str(e), file=sys.stderr)
        return 2

    print(f"清单 {result['n_rows']} 张（有标签 {result['n_tagged']}）")
    print(f"根分类（只挂靠，不当作图标签）：{' / '.join(result['roots'])}")
    for line in result.get("preview") or []:
        print(line)
    more = result.get("preview_more") or 0
    if more:
        print(f"... 另有 {more} 行")
    for line in result.get("log") or []:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
