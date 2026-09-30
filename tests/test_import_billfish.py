#!/usr/bin/env python3
"""用内存 SQLite 冒充 Billfish 库，检查导入逻辑。"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from import_billfish import BillfishError, run_import


SCHEMA = """
CREATE TABLE bf_file (id INTEGER PRIMARY KEY, name TEXT);
CREATE TABLE bf_tag_v2 (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT,
  pid INTEGER DEFAULT 0,
  seq INTEGER DEFAULT 0,
  born INTEGER,
  hide INTEGER DEFAULT 0
);
CREATE TABLE bf_tag_join_file (file_id INTEGER, tag_id INTEGER);
"""


class ImportBillfishTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db = self.root / "billfish.db"
        conn = sqlite3.connect(self.db)
        conn.executescript(SCHEMA)
        conn.execute("INSERT INTO bf_file (id, name) VALUES (1, 'a.jpg')")
        conn.execute("INSERT INTO bf_file (id, name) VALUES (2, 'b.jpg.lnk')")
        conn.execute("INSERT INTO bf_file (id, name) VALUES (3, 'orphan.jpg')")
        conn.commit()
        conn.close()
        self.jsonl = self.root / "results.jsonl"
        from vocab_store import load_active

        leaves = [t for t in load_active()["parent_of"] if t]
        self.assertGreaterEqual(len(leaves), 3)
        a, b, c = leaves[0], leaves[1], leaves[0]
        self.jsonl.write_text(
            json.dumps({"file": "/lib/a.jpg", "tags": [a, b]}, ensure_ascii=False)
            + "\n"
            + json.dumps({"file": "/lib/b.jpg", "tags": [leaves[2]]}, ensure_ascii=False)
            + "\n"
            + json.dumps({"file": "/lib/missing.jpg", "tags": [a]}, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        self.leaf_a, self.leaf_b = a, b
        self.root_a = load_active()["parent_of"][a]

    def tearDown(self):
        self.tmp.cleanup()

    def test_preview_does_not_write(self):
        result = run_import(self.jsonl, self.db, self.root / "out.csv", apply=False)
        self.assertEqual(result["matched"], 2)
        self.assertEqual(result["missed"], 1)
        conn = sqlite3.connect(self.db)
        n = conn.execute("SELECT COUNT(*) FROM bf_tag_join_file").fetchone()[0]
        conn.close()
        self.assertEqual(n, 0)
        self.assertTrue((self.root / "out.csv").is_file())

    def test_apply_creates_tree_and_links(self):
        result = run_import(
            self.jsonl,
            self.db,
            self.root / "out.csv",
            apply=True,
            create_missing=True,
            replace=True,
        )
        self.assertGreaterEqual(result["new_links"], 3)
        self.assertTrue(Path(result["backup"]).is_file())
        conn = sqlite3.connect(self.db)
        names = {r[0] for r in conn.execute("SELECT name FROM bf_tag_v2")}
        self.assertIn(self.root_a, names)
        self.assertIn(self.leaf_a, names)
        joins = conn.execute("SELECT COUNT(*) FROM bf_tag_join_file").fetchone()[0]
        conn.close()
        self.assertGreaterEqual(joins, 3)

    def test_missing_jsonl(self):
        with self.assertRaises(BillfishError):
            run_import(self.root / "nope.jsonl", self.db, apply=False)


if __name__ == "__main__":
    unittest.main()
