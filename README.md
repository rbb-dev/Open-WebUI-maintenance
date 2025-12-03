# Open WebUI Maintenance Pipe

This pipe/function turns Open WebUI into a chat-first maintenance console. From a single conversation you can:

- Scan the `file` table, chats, knowledge bases, and the upload directory for orphaned data.
- Repair chats that contain malformed Unicode.
- Detect and detach inline `data:image/...` blobs so they become managed uploads.
- Produce per-user storage summaries for capacity planning.

All destructive actions require the literal keyword `confirm`, and every run emits Markdown you can paste into change tickets.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Command Matrix](#command-matrix)
3. [Database Hygiene](#database-hygiene)
4. [Storage Hygiene](#storage-hygiene)
5. [Chat Integrity](#chat-integrity)
6. [Inline Image Cleanup](#inline-image-cleanup)
7. [User Footprints](#user-footprints)
8. [Options Cheat Sheet](#options-cheat-sheet)
9. [Input Safety](#input-safety)
10. [Valve Reference](#valve-reference)
11. [Development & Testing](#development--testing)

---

## Quick Start

1. **Prerequisites**
   - Open WebUI **0.6.28+**
   - Python 3.10+ (only if you plan to run the tests locally)

2. **Install / enable the pipe**
   - Copy `open-webui-maintenance.py` into Open WebUI’s functions directory or reference this repo’s Git URL.
   - In **Admin → Functions**, add a new function that points to the file. It will appear in the chat composer as **“Open WebUI: Maintenance.”**

3. **First run**
   - Open a chat with the pipe and type `help` to see the in-app runbook.
   - Start with narrow scopes such as `db-scan user=me limit=5`.

4. **Testing (optional)**
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements-dev.txt
   pytest
   ```

---

## Command Matrix

| Category | Scan Command | Clean Command | Default Limit |
| --- | --- | --- | --- |
| Database hygiene | `db-scan` | `db-clean-missing-files`, `db-clean-orphan-files` | 25 rows |
| Storage hygiene | `storage-scan` | `storage-clean` | 25 rows |
| Chat integrity | `chat-scan` | `chat-repair` | Scan unlimited (0). Repair 10 chats. |
| Inline image cleanup | `image-scan` | `image-detach` | 25 users / 10 chats |
| User footprints | `user-report` | — | Unlimited (0) |

All destructive commands require **`confirm`** and respect `limit`. `limit=0` removes the cap (bounded by safety valves).

---

## Database Hygiene

### `db-scan`
Cross-checks the `file` table with chats, knowledge bases (when present), and the upload directory. Highlights orphaned uploads and database rows whose files are already gone.

```text
db-scan limit=5
```

### `db-clean-missing-files confirm`
Deletes database rows whose binaries were already deleted (think “make the DB match reality”).

```text
db-clean-missing-files confirm limit=5
```

### `db-clean-orphan-files confirm`
Removes uploads that no longer have references anywhere (database row **and** storage object). Run immediately after reviewing `db-scan`.

```text
db-clean-orphan-files confirm limit=10 user="John Citizen"
```

---

## Storage Hygiene

### `storage-scan`
Walks `UPLOAD_DIR` and lists binaries that have no database record—common after manual filesystem changes or failed uploads.

```text
storage-scan limit=5
```

### `storage-clean confirm`
Deletes the on-disk files flagged by `storage-scan`. Database rows stay untouched.

```text
storage-clean confirm limit=5
```

---

## Chat Integrity

### `chat-scan`
Streams chats that contain null bytes or lone surrogate halves. Results show up live so you can stop whenever you’ve seen enough.

```text
chat-scan user_query="John Citizen" limit=10
```

### `chat-repair confirm`
Applies the sanitation in-place. Default limit is 10 chats; set `limit=0` to work through the entire scope.

```text
chat-repair confirm limit=5 user_query="John Citizen"
```

---

## Inline Image Cleanup

### `image-scan`
Searches chats for Markdown snippets like `![alt](data:image/png;base64,...)` or raw data URIs and estimates how many bytes are trapped inside the database per user.

```text
image-scan limit=3
```

### `image-detach confirm`
Uploads each inline blob through the configured storage provider, inserts a `file` row, and rewrites the chat to use `/api/v1/files/{id}/content`.

```text
image-detach confirm limit=5 user="John Citizen"
```

---

## User Footprints

### `user-report`
Aggregates per-user chat counts/sizes and file counts/sizes so you can identify big consumers before a cleanup or chargeback review.

```text
user-report limit=10
```

---

## Options Cheat Sheet

| Option | Applies To | Notes |
| --- | --- | --- |
| `limit=<n>` | Every command | Caps result rows or repairs. `0` = no cap (bounded by valve ceilings). |
| `user=<uuid>` / `user=me` | Every command | Restrict to a single owner. `me`/`self` resolves to the current user. |
| `user_query="text"` | Chat & image commands | Fuzzy match users by name/username/email when you don’t know the UUID. |
| `id=<id1,id2>` | File/chat commands | Comma/semicolon-separated UUIDs. Applies to file IDs for db/storage commands and chat IDs for chat/image maintenance. |
| `confirm` | Clean/repair/detach commands | Required safeguard for anything that mutates data. |

---

## Input Safety

- **Command prefix.** Instructions are CLI-style text—kick things off with `help` to see the menu or `db-scan` to run a scan. This keeps free-form chat text from being mistaken for maintenance actions when histories are replayed.
- **Token limits.** Individual command tokens longer than 256 characters are rejected. This keeps hostile payloads from bloating logs or exhausting parsers.
- **User/ID validation.** `user=` filters and `id=` lists must contain UUIDs (or `me`). That way the pipe never injects untrusted strings directly into SQL filters.
- **Path sandboxing.** Any future `path=`/`prefix=` option is normalized to stay inside `UPLOAD_DIR`, eliminating path-traversal surprises.

If a request violates one of these rules, the pipe returns a friendly error instead of executing the command.

---

## Valve Reference

Valves are environment-style settings exposed through `Pipe.Valves`. Highlights:

| Valve | Default | Purpose |
| --- | --- | --- |
| `SCAN_DEFAULT_LIMIT` / `SCAN_MAX_LIMIT` | 25 / 500 | Default/cap for `db-scan` and `db-clean-orphan-files`. |
| `STORAGE_SCAN_DEFAULT_LIMIT` / `STORAGE_SCAN_MAX_LIMIT` | 25 / 500 | Defaults/caps for `storage-scan` and `storage-clean`. |
| `CHAT_SCAN_DEFAULT_LIMIT` / `CHAT_SCAN_MAX_LIMIT` | 0 / 200 | Unlimited `chat-scan` by default; capped by `CHAT_SCAN_MAX_LIMIT`. |
| `CHAT_REPAIR_DEFAULT_LIMIT` / `CHAT_REPAIR_MAX_LIMIT` | 10 / 200 | Batch sizes for `chat-repair`. |
| `IMAGE_SCAN_DEFAULT_LIMIT` / `IMAGE_SCAN_MAX_LIMIT` | 25 / 200 | User rows shown per `image-scan`. |
| `IMAGE_DETACH_DEFAULT_LIMIT` / `IMAGE_DETACH_MAX_LIMIT` | 10 / 200 | Chat batches per `image-detach`. |
| `DB_CHUNK_SIZE` / `CHAT_DB_CHUNK_SIZE` | 400 / 200 | SQLAlchemy batch sizes for db/storage scans vs. chat/image scans. |
| `ENABLE_LOGGING` | False | Set to `True` to log INFO-level status messages for audit trails. |

See `docs/VALVES.md` for the full matrix.

---

## Development & Testing

```
git clone https://github.com/rbb-dev/Open-WebUI-maintenance.git
cd Open-WebUI-maintenance
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

The pytest suite uses lightweight stubs (see `tests/conftest.py`) so it runs without a full Open WebUI stack. When contributing, please keep sample names anonymized as “John Citizen” and run `pytest` before opening a PR.
