"""
hyde_cache.py — disk cache for HyDE's hypothetical answers.

HyDE asks the LLM for a hypothetical answer at temperature 0.3, so two
identical searches retrieve with different text and an A/B of anything
downstream is noise. Caching the FIRST answer per question makes repeated
searches identical (the eval depends on that) and takes an LLM round-trip
off the warm path. The key covers everything that changes the answer:
the question (whitespace-normalised, case kept), the prompt, the model
and the temperature — change any of them and it is a different entry.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path


class HydeCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # One connection shared across FastAPI's threadpool, serialised by a
        # lock: sqlite3 objects are not safe to use from two threads at once.
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        with self._lock:
            # WAL + synchronous=NORMAL: a put stops paying a full fsync (measured
            # ~140 ms per commit on A:, ~0.05 ms in WAL). The price is that a
            # power cut can lose the last few entries — for a CACHE that costs
            # one LLM call each to regenerate, never a wrong answer.
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS hyde ("
                "key TEXT PRIMARY KEY, text TEXT NOT NULL, created_at TEXT NOT NULL)")
            self._db.commit()

    @staticmethod
    def make_key(question: str, prompt_fp: str, model: str, temperature: float) -> str:
        norm = " ".join(question.split())
        blob = json.dumps([norm, prompt_fp, model, float(temperature)], ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT text FROM hyde WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, text: str) -> str:
        """Store `text` unless the key already has an entry, and return the text
        that is stored. The cache keeps the FIRST draft per question: two searches
        that both missed and both asked the LLM must not each retrieve with their
        own draft, and a later write must not replace the one earlier searches
        already used. The caller retrieves with what this returns."""
        with self._lock:
            self._db.execute(
                "INSERT OR IGNORE INTO hyde (key, text, created_at) VALUES (?, ?, ?)",
                (key, text, time.strftime("%Y-%m-%dT%H:%M:%S")))
            self._db.commit()
            return self._db.execute(
                "SELECT text FROM hyde WHERE key = ?", (key,)).fetchone()[0]

    def __len__(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM hyde").fetchone()[0]
