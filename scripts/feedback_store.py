"""SQLite storage + EMA update logic for the like/dislike feedback loop.

Same two-table shape originally designed for DynamoDB (raw append-only events:
restaurant_id/song_id/direction/timestamp, and a derived restaurant_id ->
adjustment vector table) -- just backed by a local file instead of a managed
cloud table, since DynamoDB's free tier expires and this doesn't need to
survive at cloud scale. If it ever does, only the bodies of these functions
change, not what callers see.

No DynamoDB Streams equivalent exists locally, so there's no separate consumer
process: maybe_update_adjustment() is called inline, right after each write.
"""
import json
import os
import sqlite3
import time
import uuid

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get(
    "FEEDBACK_DB_PATH",
    os.path.join(HERE, "..", "data", "processed", "feedback.db"),
)

EVENT_THRESHOLD = 4
ALPHA = 0.3
MAX_ADJUSTMENT_NORM = 2.0


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA busy_timeout = 5000")  # wait up to 5s on a lock instead of failing immediately
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback_events (
            restaurant_id TEXT NOT NULL,
            event_id TEXT NOT NULL,
            song_id TEXT NOT NULL,
            direction TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            PRIMARY KEY (restaurant_id, event_id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS restaurant_adjustments (
            restaurant_id TEXT PRIMARY KEY,
            adjustment TEXT NOT NULL,
            last_processed_event_id TEXT NOT NULL,
            updated_at INTEGER NOT NULL
        )
    """)
    return conn


def write_feedback_event(restaurant_id: str, song_id: str, direction: str) -> str:
    if direction not in ("like", "dislike"):
        raise ValueError(f"direction must be 'like' or 'dislike', got {direction!r}")

    now_ms = int(time.time() * 1000)
    event_id = f"{now_ms}#{uuid.uuid4().hex[:8]}"

    with _connect() as conn:
        conn.execute(
            "INSERT INTO feedback_events (restaurant_id, event_id, song_id, direction, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (restaurant_id, event_id, song_id, direction, now_ms),
        )
    return event_id


def get_events_since(restaurant_id: str, after_event_id: str = "") -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT event_id, song_id, direction, timestamp FROM feedback_events "
            "WHERE restaurant_id = ? AND event_id > ? ORDER BY event_id",
            (restaurant_id, after_event_id),
        ).fetchall()
    return [
        {"event_id": r[0], "song_id": r[1], "direction": r[2], "timestamp": r[3]}
        for r in rows
    ]


def get_current_adjustment(restaurant_id: str):
    """Returns (adjustment: list[float] | None, last_processed_event_id: str)."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT adjustment, last_processed_event_id FROM restaurant_adjustments "
            "WHERE restaurant_id = ?",
            (restaurant_id,),
        ).fetchone()
    if row is None:
        return None, ""
    return json.loads(row[0]), row[1]


def save_current_adjustment(restaurant_id: str, adjustment: list[float], last_processed_event_id: str):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO restaurant_adjustments "
            "(restaurant_id, adjustment, last_processed_event_id, updated_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(restaurant_id) DO UPDATE SET "
            "adjustment = excluded.adjustment, "
            "last_processed_event_id = excluded.last_processed_event_id, "
            "updated_at = excluded.updated_at",
            (restaurant_id, json.dumps(adjustment), last_processed_event_id, int(time.time() * 1000)),
        )


def compute_ema_update(old_adjustment, liked_vecs, disliked_vecs, alpha=ALPHA, max_norm=MAX_ADJUSTMENT_NORM):
    """batch_signal = mean(liked) - mean(disliked); EMA-blend with the previous
    adjustment; clip magnitude so repeated updates in one direction can't grow
    unbounded. Simplest version on purpose -- see notes/progress_log.md.
    """
    dim = len(liked_vecs[0]) if liked_vecs else len(disliked_vecs[0])
    batch_signal = np.zeros(dim)
    if liked_vecs:
        batch_signal += np.mean(liked_vecs, axis=0)
    if disliked_vecs:
        batch_signal -= np.mean(disliked_vecs, axis=0)

    base = np.asarray(old_adjustment) if old_adjustment is not None else np.zeros(dim)
    new_adjustment = alpha * batch_signal + (1 - alpha) * base

    norm = np.linalg.norm(new_adjustment)
    if norm > max_norm:
        new_adjustment = new_adjustment * (max_norm / norm)

    return new_adjustment


def maybe_update_adjustment(restaurant_id: str, song_vectors: dict, threshold: int = EVENT_THRESHOLD) -> bool:
    """Call right after writing a new event. If this restaurant has crossed the
    event threshold since its last update, folds the new events in and saves.
    Returns True if an update happened."""
    old_adjustment, last_event_id = get_current_adjustment(restaurant_id)
    new_events = get_events_since(restaurant_id, last_event_id)

    if len(new_events) < threshold:
        return False

    liked_vecs = [song_vectors[e["song_id"]] for e in new_events
                  if e["direction"] == "like" and e["song_id"] in song_vectors]
    disliked_vecs = [song_vectors[e["song_id"]] for e in new_events
                      if e["direction"] == "dislike" and e["song_id"] in song_vectors]

    if not liked_vecs and not disliked_vecs:
        return False  # none of the songs in this batch were recognized

    new_adjustment = compute_ema_update(old_adjustment, liked_vecs, disliked_vecs)
    newest_event_id = max(e["event_id"] for e in new_events)
    save_current_adjustment(restaurant_id, new_adjustment.tolist(), newest_event_id)
    return True
