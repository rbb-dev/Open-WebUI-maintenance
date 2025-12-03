# Open WebUI Maintenance Pipe

Open WebUI Maintenance is a chat-first function/pipe for [Open WebUI](https://github.com/open-webui/open-webui) administrators. It scans uploaded files, cross-checks chats and knowledge bases, audits on-disk storage, and produces usage summaries so you can keep deployments tidy without leaving the UI. The result is a concise Markdown report (plus optional cleanup helpers) you can drop straight into tickets or runbooks.

Unlike destructive clean-up scripts, this pipe never deletes anything—it simply answers the question “which files are safe to purge?” and leaves the remediation step in your hands.

---

## Table of Contents
1. [Features](#features)
2. [How It Works](#how-it-works)
3. [Installation](#installation)
4. [Usage](#usage)
   - [Commands](#commands)
   - [Options](#options)
   - [Examples](#examples)
5. [Output Format](#output-format)
6. [Valve Reference](#valve-reference)
7. [Testing](#testing)
8. [Repository Layout](#repository-layout)

---

## Features
- **Workspace-wide scan:** Loads the entire `file` table (optionally scoped by user or file IDs) and checks each upload against chats and knowledge bases.
- **Filesystem audit:** `storage-scan` walks the upload directory and lists files on disk that have no matching database entry.
- **Guided cleanup commands:** `db-clean` and `storage-clean` delete only the rows/files you scoped, and they require `confirm` for safety.
- **CLI-style workflow:** Interact with the pipe directly from a chat window; each command streams Markdown that you can copy into tickets or runbooks.
- **Configurable limits:** Valves let you change default/maximum row counts and batch sizes without editing code.

## How It Works
1. **Command parsing:** Every user prompt is interpreted as a CLI command (`db-scan`, `storage-scan`, etc.) or `help`.
2. **File loading:** The pipe fetches the `file` table in batches, honoring `user=` and `id=` filters when provided.
3. **Reference discovery:** All chats are traversed for `history.messages.*.files[].id` entries, and the `knowledge_file` table (when available) is checked for additional references.
4. **Orphan detection:** File IDs that never appear in either data source are tagged as orphans and surfaced in a Markdown table.
5. **Reporting:** The summary lists totals (files, referenced uploads, suspected orphans) plus a table of detailed orphan metadata.

## Installation
1. Ensure you are running Open WebUI **0.6.28 or newer** (matches the `required_open_webui_version`).
2. Clone this repository or copy `open-webui-maintenance.py` into your Open WebUI functions directory.
3. From Open WebUI, open **Admin → Functions** and point a new function to this file (or use the provided Git URL).
4. After saving, the function appears as **“Open WebUI: Maintenance”** in the chat composer’s function picker.

## Usage
Interact with the pipe as if it were a CLI exposed through chat.

### Database maintenance
- `db-scan [options]` — load the file table, compare it against chats/knowledge bases, and list uploads that no longer have references. Also surfaces database records whose files are missing on disk.
- `db-clean confirm [options]` — delete the database records whose files were flagged as missing. Requires the literal word `confirm`.

### Storage maintenance
- `storage-scan [options]` — walk the upload directory and list files sitting on disk without database records.
- `storage-clean confirm [options]` — delete the files on disk that lack database records. Also requires `confirm`.

### User reporting
- `user-report [options]` — aggregate per-user chat counts/sizes and file counts/sizes, then show the combined storage footprint. Helpful for chargeback, quota reviews, or targeting cleanup work.

### Options
| Option | Description |
| --- | --- |
| `limit=<n>` | Cap the number of orphaned files in the output. `limit=0` lifts the cap (default `25`, max `500`). |
| `user=<uuid>` / `user=me` | Restrict the scan to files uploaded by a specific user (current user when `me`). |
| `id=<file-id>` | One or more comma/semicolon-separated file IDs to inspect directly. |
| `confirm` | Required for `db-clean` and `storage-clean`. Prevents accidental deletions. |

### Examples
```
db-scan                      # Database vs. chat/knowledge scan (default limit=25 entries)
storage-scan                 # Filesystem audit (default limit=25 entries per table)
db-clean confirm limit=10    # Delete up to 10 database records whose files are missing
storage-clean confirm        # Delete every file on disk that lacks a database record (respecting the limit)
user-report limit=20         # List the top 20 users by combined chat+file size
```

## Output Format
Each `db-scan` command returns a Markdown block similar to:

```
### Database scan results

- Files in database: 312
- Referenced in chats: 287
- Referenced in knowledge bases: 10
- Files with no remaining references: 15
- Database records with missing files on disk: 2
- Scope: entire workspace
- Output limit: 25

| File ID | Owner | Name | Size | Created (UTC) | Path |
| --- | --- | --- | --- | --- | --- |
| `f877...` | Alice | sample.pdf | 1.24 MB | 2024-05-01 12:04:55 UTC | local upload dir |
```

Copy the table into your docs or issue tracker, then delete/archive the listed files through your preferred channel (API, CLI, or storage console).

## Valve Reference
See [`docs/VALVES.md`](docs/VALVES.md) for the full valve matrix. Highlights:

| Valve | Default | Purpose |
| --- | --- | --- |
| `ENABLE_LOGGING` | `False` | Emit INFO logs for each scan stage when you need audit trails. |
| `SCAN_DEFAULT_LIMIT` / `SCAN_MAX_LIMIT` | `25` / `500` | Defaults/caps for the orphaned-file scan. |
| `STORAGE_SCAN_DEFAULT_LIMIT` / `STORAGE_SCAN_MAX_LIMIT` | `25` / `500` | Defaults/caps for storage-scan output and cleanup batch sizes. |
| `DB_CHUNK_SIZE` | `400` | Number of rows fetched per SQLAlchemy batch. Increase for high-latency databases. |

## Testing
The repository includes a lightweight pytest suite that exercises helper utilities without requiring a live Open WebUI instance.

```
python -m venv .venv && source .venv/bin/activate  # optional but recommended
pip install -r requirements-dev.txt
pytest
```

## Repository Layout
```
open-webui-maintenance/
├── open-webui-maintenance.py   # Pipe implementation
├── README.md                    # This document
├── docs/
│   └── VALVES.md                # Extended valve guidance
└── tests/
    ├── conftest.py             # Open WebUI stubs for pytest
    ├── test_file_cleanup_service.py
    └── test_pipe_utilities.py
```

Contributions and bug reports are welcome—please include repro steps and ensure `pytest` passes before submitting a PR. 🤝
