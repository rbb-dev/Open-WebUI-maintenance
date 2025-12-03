# Valve Reference

The Open WebUI Maintenance pipe exposes a handful of environment-configurable knobs (“valves”) via the `Pipe.Valves` Pydantic model. You can override them from Open WebUI’s **Function → Valves** UI without touching the source file.

Each valve is validated (min/max) before the pipe boots. Invalid values fall back to defaults.

| Valve | Type | Default | Range | When to change |
| --- | --- | --- | --- | --- |
| `ENABLE_LOGGING` | bool | `False` | `True`/`False` | Turn on when you want INFO-level logs for each scan stage (helpful during incident response). |
| `SCAN_DEFAULT_LIMIT` | int | `25` | `0 – 5000` | Set to `0` if you always want the full orphaned-file list without specifying `limit=0`. |
| `SCAN_MAX_LIMIT` | int | `500` | `25 – 2000` | Absolute ceiling for `limit=`. Raise cautiously if you expect thousands of orphaned files. |
| `STORAGE_SCAN_DEFAULT_LIMIT` | int | `25` | `0 – 5000` | Default number of storage mismatches reported (and cleaned) per run. |
| `STORAGE_SCAN_MAX_LIMIT` | int | `500` | `25 – 2000` | Hard ceiling for storage scans/cleanups—raise carefully if you expect huge backlogs. |
| `CHAT_SCAN_DEFAULT_LIMIT` | int | `0` | `0 – 5000` | Default chat rows listed during `chat-scan` (0 = unlimited stream). |
| `CHAT_SCAN_MAX_LIMIT` | int | `200` | `25 – 1000` | Hard ceiling for chat-scan output. |
| `CHAT_REPAIR_DEFAULT_LIMIT` | int | `10` | `0 – 200` | Default number of chats repaired per `chat-repair` run (0 = unlimited). |
| `CHAT_REPAIR_MAX_LIMIT` | int | `200` | `10 – 1000` | Safety net for chat repair batches. |
| `DB_CHUNK_SIZE` | int | `400` | `50 – 2000` | Rows fetched per SQLAlchemy batch for file/storage operations. |
| `CHAT_DB_CHUNK_SIZE` | int | `200` | `50 – 1000` | Chat rows fetched per batch during `chat-scan`/`chat-repair`. |

## Usage Notes
- **Changes apply instantly.** Once you tweak a valve in the UI, the next invocation of the pipe will pick it up.
- **Limit safety.** The `SCAN_MAX_LIMIT` valve prevents someone from accidentally dumping hundreds of thousands of rows into a single chat response.
- **Chunk size trade-offs.** Larger values reduce round trips but keep database transactions open longer; adjust according to your infrastructure.
- **Logging valve.** When `ENABLE_LOGGING=True`, the module logger switches to `DEBUG` and propagates through the hosting app. Remember to turn it off when you’re done triaging.

## Updating Valves Programmatically
Valves are regular Pydantic fields, so you can also override them in code:

```python
from open_webui_maintenance import Pipe

pipe = Pipe()
pipe.valves.SCAN_DEFAULT_LIMIT = 0
pipe._apply_logging_valve()  # re-evaluate logging level after manual changes
```

In production, prefer using Open WebUI’s UI so settings stay consistent across replicas.
