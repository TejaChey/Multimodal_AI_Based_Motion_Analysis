"""SQLite feature store: every request's features/quality/prediction, for monitoring and retraining."""
import json
import sqlite3
import time

import pandas as pd

from .config import art_dir

SCHEMA = """CREATE TABLE IF NOT EXISTS requests(
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, recording_id TEXT, athlete_id TEXT,
    quality_pass INTEGER, model_version TEXT, predicted_tier TEXT, v_statures_per_s REAL,
    features TEXT, quality TEXT, timings TEXT)"""


class FeatureStore:
    def __init__(self, path=None):
        self.path = str(path or art_dir() / "feature_store.db")
        with self._c() as c:
            c.execute(SCHEMA)

    def _c(self):
        con = sqlite3.connect(self.path, timeout=30)
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def add(self, recording_id, athlete_id, features, quality, v_statures_per_s,
            model_version=None, predicted_tier=None, timings=None, ts=None):
        with self._c() as c:
            c.execute("INSERT INTO requests(ts,recording_id,athlete_id,quality_pass,model_version,predicted_tier,"
                      "v_statures_per_s,features,quality,timings) VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (ts or time.time(), recording_id, athlete_id, int(quality["passed"]), model_version,
                       predicted_tier, v_statures_per_s, json.dumps(features), json.dumps(quality),
                       json.dumps(timings or {})))

    def count(self, passed_only=False) -> int:
        with self._c() as c:
            return c.execute("SELECT COUNT(*) FROM requests" + (" WHERE quality_pass=1" if passed_only else "")).fetchone()[0]

    def frame(self, passed_only=True, last=None) -> pd.DataFrame:
        """Rows expanded to one column per feature (oldest first)."""
        q = "SELECT * FROM requests" + (" WHERE quality_pass=1" if passed_only else "") + " ORDER BY id"
        with self._c() as c:
            df = pd.read_sql_query(q, c)
        if df.empty:
            return df
        feats = pd.DataFrame([json.loads(x) for x in df.features])
        out = pd.concat([df.drop(columns=["features", "quality", "timings"]), feats], axis=1)
        return out.tail(last) if last else out
