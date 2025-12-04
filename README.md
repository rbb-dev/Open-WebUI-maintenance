# Open WebUI Maintenance Pipe

Open WebUI deployments naturally accumulate data over time: uploaded files, chat histories, and embedded images. This maintenance pipe transforms an ordinary chat conversation into a full-featured operations console, letting administrators audit storage, repair corrupted data, and clean up orphaned files—all without leaving the chat interface or touching the command line.

Every scan produces detailed Markdown tables you can paste directly into change tickets. Every cleanup command shows exactly what it removed. And because everything runs inside a chat, your entire maintenance session becomes an auditable trail that's automatically preserved in your conversation history.

## Core Design Principles

- **Chat-first workflow.** Run maintenance commands the same way you chat with models—just type what you need. Every scan and cleanup is logged inline for compliance and change control.
- **Conservative defaults.** Destructive operations default to `limit=5`, so a mis-scoped command only touches a handful of records. You can widen the limit after previewing results.
- **`confirm` safety switch.** Nothing that mutates data (cleanups, repairs, detachments) runs without the literal word `confirm` in your command.
- **Streaming output.** Long-running scans stream results as they're calculated. Watch progress in real-time and cancel early if you spot an issue.
- **Flexible filtering.** Every command supports `user=<id>` (or `user=me` for yourself), `user_query="name"` for fuzzy matching, and specific ID filters to precisely control scope.

---

## Table of Contents

1. [Capability Matrix](#capability-matrix)
2. [Functional Runbooks](#functional-runbooks)
   - [Database Hygiene](#database-hygiene)
   - [Storage Hygiene](#storage-hygiene)
   - [Chat Integrity](#chat-integrity)
   - [Inline Image Cleanup](#inline-image-cleanup)
   - [User Footprints](#user-footprints)
3. [Operational Safeguards](#operational-safeguards)
4. [Configuration Valves](#configuration-valves)
5. [Deployment](#deployment)
6. [Development & Testing](#development--testing)

---

## Capability Matrix

| Functional Area | Scan Command | Remediation Command | Default Limit | Notes |
| --- | --- | --- | --- | --- |
| Database hygiene | `db-scan` | `db-clean-missing-files`, `db-clean-orphan-files` | 5 rows | Cross-checks uploaded files against your chats and knowledge bases to find orphans (no references) and ghosts (database entries with missing files). |
| Storage hygiene | `storage-scan` | `storage-clean` | 5 rows | Walks your upload directory to find files on disk that don't have database records—often leftovers from interrupted uploads or manual file operations. |
| Chat integrity | `chat-scan` | `chat-repair` | 5 rows | Identifies chats with broken Unicode (null bytes or corrupted characters) that cause database errors. Repair sanitizes them in place. |
| Inline image cleanup | `image-scan` | `image-detach` | 5 users / 5 chats | Finds images embedded directly in chats and converts them to proper file attachments, freeing database space and making them manageable through standard file operations. |
| User footprints | `user-report` | N/A | 0 (report for all users) | Shows each user's chat and file storage consumption in one table—useful for capacity planning or identifying high-usage accounts. |

> **Default limit rationale:** every workflow starts with `limit=5` to limit the blast radius if an operator forgets to tighten the scope. Raise the limit only after validating the preview rows.

---

## Functional Runbooks

### Database Hygiene

| Command | Purpose | Example |
| --- | --- | --- |
| `db-scan [limit=<n>] [user=<id>\|user=me] [id=<file-id>]` | Check which files are still referenced in chats or knowledge bases. Identifies orphaned files (safe to delete) and missing files (database entries without matching disk files). | `db-scan user=me limit=5` |
| `db-clean-missing-files confirm [...]` | Clean up database entries for files that have already been deleted from disk (fixes "ghost references"). | `db-clean-missing-files confirm limit=5 user=me` |
| `db-clean-orphan-files confirm [...]` | Delete uploaded files that have no references in chats or knowledge bases (both database record and file). | `db-clean-orphan-files confirm limit=5 user_query="John Citizen"` |

> **How it works:** `db-scan` reads your file table in batches, then checks each file against chat attachments (`history.messages.*.files[]`) and knowledge base references. It also verifies that every database record has a matching file on disk. The scan produces two lists: orphaned files (database entries with no references) and missing files (database entries where the actual file is gone). Run this after bulk deletions or migrations to catch inconsistencies. Use `limit=0` to see everything, or keep the default `limit=5` for a quick preview.

Example: db-scan limit=5
### Database scan results

- Files in database: 2360
- Referenced in chats: 414
- Referenced in knowledge bases: 230
- Orphaned files (no remaining references): 1725
- Database records with missing files on disk: 123
- Scope: entire workspace
- Output limit: 5
- Some orphaned files were not shown because of the output limit.

#### Orphaned files (no remaining references)
| File ID | Owner | Name | Size | Created (UTC) | Path |
| --- | --- | --- | --- | --- | --- |
| `e7f0036d-cce4-44e5-9702-bbc508fc9b38` | John Citizen | Configure CoS.pdf | 2.57 MB | 2025-07-04 11:49:46 UTC | /app/backend/data/uploads/e7f0036d-cce4-44e5-9702-bbc508fc9b38_Configure CoS.pdf |
| `adde8e0c-b75f-4222-b483-d039c4f2fdf4` | John Citizen | Untitled.txt | 26.57 KB | 2025-07-11 14:17:14 UTC | /app/backend/data/uploads/adde8e0c-b75f-4222-b483-d039c4f2fdf4_Untitled.txt |
| `2399f88c-657e-4fee-b284-42cb177fef4b` | John Citizen | 01_Initial_Software_Configuration_Overview.md | 5.87 KB | 2025-07-18 06:25:23 UTC | /app/backend/data/uploads/2399f88c-657e-4fee-b284-42cb177fef4b_01_Initial_Software_Configuration_Overview.md |
| `bc1d7f66-aa91-4e31-bd43-9a85b5a0980c` | John Citizen | Install_on_Azure.md | 27.31 KB | 2025-07-18 06:44:10 UTC | /app/backend/data/uploads/bc1d7f66-aa91-4e31-bd43-9a85b5a0980c_Install_on_Azure.md |
| `45b95aca-0d3c-483a-8afc-764f1d27b040` | John Citizen | Verify_Support_for_UEFI_Secure_Boot.md | 9.43 KB | 2025-07-18 06:49:29 UTC | /app/backend/data/uploads/45b95aca-0d3c-483a-8afc-764f1d27b040_Verify_Support_for_UEFI_Secure_Boot.md |

#### Database records with missing files
| File ID | Owner | Name | Last Known Path | Updated (UTC) |
| --- | --- | --- | --- | --- |
| `869e4c28-9a5a-451c-9741-e9705dd6941e` | John Citizen | Getting_Started.md | /app/backend/data/uploads/869e4c28-9a5a-451c-9741-e9705dd6941e_Getting_Started.md | 2025-07-18 08:07:04 UTC |
| `59834895-065f-4ee3-901c-d6c0a6dc1e1e` | John Citizen | Install_on_Nutanix.md | /app/backend/data/uploads/59834895-065f-4ee3-901c-d6c0a6dc1e1e_Install_on_Nutanix.md | 2025-07-18 07:03:38 UTC |
| `d943e243-d3e7-4375-bf84-e1bf4f268a5b` | John Citizen | Verification.md | /app/backend/data/uploads/d943e243-d3e7-4375-bf84-e1bf4f268a5b_Verification.md | 2025-07-18 07:06:27 UTC |
| `10067d3f-7139-4b14-828c-fc9386af1b22` | John Citizen | Installation.md | /app/backend/data/uploads/10067d3f-7139-4b14-828c-fc9386af1b22_Installation.md | 2025-07-18 07:06:19 UTC |
| `3de8d576-43f1-478b-a15c-c8d0c17b5ec4` | John Citizen | Deployment_Basics.md | /app/backend/data/uploads/3de8d576-43f1-478b-a15c-c8d0c17b5ec4_Deployment_Basics.md | 2025-07-18 07:14:25 UTC |

Use `db-scan limit=0` to list every result or add `user=<id>` to focus on a single owner.


---


### Storage Hygiene

| Command | Purpose | Example |
| --- | --- | --- |
| `storage-scan [limit=<n>] [...]` | Walk through your upload directory and identify files that exist on disk but don't have a matching database record (often left behind after manual file operations or interrupted uploads). Shows the file path, size, and when it was last modified. | `storage-scan limit=5` |
| `storage-clean confirm [limit=<n>] [...]` | Delete the orphaned files that `storage-scan` found, files sitting on disk with no database entry. Requires you to add `confirm` so nothing gets removed by accident. The database stays untouched, only the physical files are removed. | `storage-clean confirm limit=5 user=me` |

> **How it works:** `storage-scan` walks your entire upload directory (recursively), derives file IDs from filenames using Open WebUI's UUID naming pattern, and compares each discovered file against the database. Files without matching records are reported with their path, size, and modification time. This is particularly useful after manual filesystem operations, restores from backup, or interrupted uploads. The `storage-clean` command physically deletes these orphaned files while leaving the database untouched.

Example: storage-scan
### Storage scan results

- Files found on disk: 2245
- Files on disk with no database record: 3
- Scope: entire workspace
- Output limit: 25

#### Files on disk without database records
| Derived ID | Name | Size | Modified (UTC) | Path |
| --- | --- | --- | --- | --- |
| `e8c9bb0b-6340-4949-8f19-d992303e6daa` | e8c9bb0b-6340-4949-8f19-d992303e6daa_generated-image.png | 77.24 KB | 2025-07-03 06:17:55 UTC | /app/backend/data/uploads/e8c9bb0b-6340-4949-8f19-d992303e6daa_generated-image.png |
| `342933f7-0b93-44f7-9278-33cede82cdd2` | 342933f7-0b93-44f7-9278-33cede82cdd2_generated-image.png | 850.04 KB | 2025-11-26 06:48:37 UTC | /app/backend/data/uploads/342933f7-0b93-44f7-9278-33cede82cdd2_generated-image.png |
| `94c56889-1bbe-4087-b609-442fa94858ae` | 94c56889-1bbe-4087-b609-442fa94858ae_manifest.csv | 1.00 MB | 2025-09-30 01:08:11 UTC | /app/backend/data/uploads/94c56889-1bbe-4087-b609-442fa94858ae_manifest.csv |


---

### Chat Integrity

| Command | Purpose | Example |
| --- | --- | --- |
| `chat-scan [limit=<n>] [user=<id>\|user=me] [user_query="text"] [id=<chat-id>]` | Find chats that contain broken Unicode (null bytes or orphaned surrogate characters that can cause database errors). Shows exactly what's wrong with each chat so you can decide what to fix first. | `chat-scan user_query="John Citizen" limit=5` |
| `chat-repair confirm [limit=<n>] [...]` | Clean up the problematic chats that `chat-scan` found by replacing broken characters with safe placeholders. Writes the fixed version back to the database. Defaults to 5 chats per run so you can verify results before processing more. | `chat-repair confirm limit=5 user_query="John Citizen"` |

> **How it works:** `chat-scan` batches chats by owner and inspects the title, payload, and metadata fields for malformed Unicode: null bytes (`\x00`), lone high surrogates (0xD800-0xDBFF), and lone low surrogates (0xDC00-0xDFFF). These issues typically arise from copy-paste operations, data imports, or buggy client code. The scan reports exactly which fields are affected and what kinds of corruption exist. `chat-repair` then sanitizes these strings in place by removing null bytes and replacing broken surrogates with the Unicode replacement character (�). Run this before PostgreSQL reindexing or whenever you see "invalid byte sequence" errors.

### Inline Image Cleanup

| Command | Purpose | Example |
| --- | --- | --- |
| `image-scan [limit=<n>] [user=<id>\|user=me] [user_query="text"]` | Find chats that have images embedded directly into chat (instead of proper file attachments). Shows how much database space you could reclaim by converting them to normal files. | `image-scan limit=5` |
| `image-detach confirm [limit=<n>] [id=<chat-id>] [...]` | Convert embedded images into proper Open WebUI files by uploading them to storage and updating the chat to link to the new file instead. | `image-detach confirm limit=5 user=me` |

> **How it works:** `image-scan` traverses chat payloads looking for Markdown-style inline images (`![alt](data:image/...;base64,...)`) and bare data URIs. These blobs bloat the database, bypass storage quotas, and prevent proper access control. The scan aggregates total bytes per user and estimates how much space you'll reclaim. `image-detach` then decodes each base64 blob, uploads it via the storage provider (creating a proper `file` record), and rewrites the chat to reference `/api/v1/files/{id}/content` instead. This moves images into managed storage where they can be tracked, backed up, and governed like normal uploads. Run this after migrations or when database size grows unexpectedly.

### User Footprints

| Command | Purpose | Example |
| --- | --- | --- |
| `user-report [limit=<n>] [user_query="text"] [stream]` | Show a detailed breakdown for each user: how many chats they have (with total size), how many files they have stored (with total size), and their combined storage footprint. Useful for capacity planning or identifying who's using the most space. | `user-report stream limit=0` |

> When `stream` is set, the pipe emits each row as soon as it is calculated; the conversation shows live progress without buffering the entire dataset.

Example: user-report
### User usage report

| User | Chats | Files | Total storage |
| --- | --- | --- | --- |
| John Citizen | 136 (66.36 MB) | 34 (59.90 MB) | 126.25 MB |
| Jane Citizen | 1024 (455.43 MB) | 1031 (694.61 MB) | 1.12 GB |
| James Citizen | 5 (70.97 KB) | 0 (0 B) | 70.97 KB |
| Jennifer Citizen | 287 (46.15 MB) | 38 (23.67 MB) | 69.82 MB |
| Joe Citizen | 1384 (413.61 MB) | 348 (416.13 MB) | 829.74 MB |

Report complete – displayed 145 users.

---
Example: user-report user=me
### User usage report

| User | Chats | Files | Total storage |
| --- | --- | --- | --- |
| John Citizen | 1028 (455.92 MB) | 1031 (694.61 MB) | 1.12 GB |

Report complete – displayed 1 user.

#### Largest chats for John Citizen
| Title | Size |
| --- | --- |
| 🌇 Eiffel Tower Sunset Image | 28.15 MB |
| 🚁 Drone Resort Flight | 27.39 MB |
| 🛠️ API Tool Data Testing | 23.49 MB |
| 🛠️ Microsoft 365 Agent Functions | 8.63 MB |
| Clone of 🛠️ OpenRouter API Code Review | 8.42 MB |

#### Largest files for John Citizen
| File name | Size |
| --- | --- |
| generated-video-a4fced218a114be6b69a0e4abf22b128.mp4 | 8.11 MB |
| pasted-image-ba01c0e4540348638fced305c2ad9c31.png | 6.46 MB |
| pasted-image-3bedb585bcdf4e0e8680c1f5f8b3f8e3.png | 6.46 MB |
| generated-video-694c3f89ee1848098de655f9a8977736.mp4 | 5.13 MB |
| generated-video-79321120374f4e99a12673e88411be67.mp4 | 5.10 MB |


---

## Operational Safeguards

- **`confirm` handshake.** Any command that mutates state (`*-clean`, `*-repair`, `image-detach`) hard-stops until the request includes `confirm`.
- **Scoped limits.** `limit` is honored everywhere. `5` is the default; `0` means “no cap” but still respects the maximum configured valve.
- **Streaming scans.** Long-running scans push rows incrementally. You can stop after reviewing the first few entries without waiting for the entire run.
- **Input validation.** User IDs and file/chat IDs must be UUIDs (or `me`). Tokens longer than 256 characters are rejected. Optional future `path` values are normalized to stay inside `UPLOAD_DIR`.
- **Audit-friendly Markdown.** Every run produces Markdown tables that can be pasted directly into change records or incident tickets.

---

## Configuration Valves

Valves expose dial-tone settings for administrators. Adjust them via **Admin → Functions → Valves** without patching the source.

| Valve | Default | Description |
| --- | --- | --- |
| `SCAN_DEFAULT_LIMIT` / `SCAN_MAX_LIMIT` | 5 / 500 | Default and ceiling for `db-scan`/`db-clean-orphan-files`. |
| `STORAGE_SCAN_DEFAULT_LIMIT` / `STORAGE_SCAN_MAX_LIMIT` | 5 / 500 | Default and ceiling for storage scan/clean commands. |
| `CHAT_SCAN_DEFAULT_LIMIT` / `CHAT_SCAN_MAX_LIMIT` | 5 / 200 | Rows streamed per `chat-scan`. |
| `CHAT_REPAIR_DEFAULT_LIMIT` / `CHAT_REPAIR_MAX_LIMIT` | 5 / 200 | Chats repaired per `chat-repair`. |
| `IMAGE_SCAN_DEFAULT_LIMIT` / `IMAGE_SCAN_MAX_LIMIT` | 5 / 200 | Users included per `image-scan`. |
| `IMAGE_DETACH_DEFAULT_LIMIT` / `IMAGE_DETACH_MAX_LIMIT` | 5 / 200 | Chats processed per `image-detach`. |
| `DB_CHUNK_SIZE` / `CHAT_DB_CHUNK_SIZE` | 400 / 200 | SQLAlchemy batch sizes for file vs. chat workloads. |
| `ENABLE_LOGGING` | False | Switches the module logger to DEBUG/INFO for traceability. |

Full details live in [`docs/VALVES.md`](docs/VALVES.md), including tuning advice for large installations.

---

## Deployment

1. **Prerequisites**
   - Open WebUI **0.6.28 or newer**.
   - Python 3.10+ if you plan to run the local test suite.

2. **Install or reference the pipe**
   - Copy `open-webui-maintenance.py` into the Open WebUI functions directory *or* reference this repository’s Git URL from the Functions UI.
   - In **Admin → Functions**, create a new function pointing at the file. It will appear in the composer as **“Open WebUI: Maintenance.”**

3. **First run**
   - Open a conversation with the pipe and send `help`. The inline runbook mirrors the guidance in this README.
   - Start with a scoped command such as `db-scan user=me limit=5 confirm`.

---

## Development & Testing

```bash
git clone https://github.com/rbb-dev/Open-WebUI-maintenance.git
cd Open-WebUI-maintenance
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

The pytest suite uses lightweight stubs so it runs without a full Open WebUI backend. Please keep sample names anonymized (e.g., “John Citizen”) and run `pytest` before sending a pull request.
