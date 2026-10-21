# lib/llm/src/kv_router

## Module map

- `kv_router.rs` estimates how much of a request each worker already has cached, then asks the scheduler to choose a worker. It selects a worker but does not send the request.
- `embedded.rs` connects discovered workers and their current load to the scheduling code in `lib/kv-router`.
- `indexer/` tracks which KV-cache blocks each worker holds; lookups for reusable blocks run in `lib/kv-router` before selection.
- `publisher/` receives KV-cache events and worker metrics from inference engines.
- `session_stats.rs` turns the session prefix index into per-session telemetry: the working set against capacity, thrash (prior prefix a returning session had to recompute or reload), and one `session_summary` log per session. It is passive and active only with `enable_session_prefix_index`. Metric definitions live in `metrics/session.rs`; the catalog entry is "Session telemetry" in `docs/fern/pages/reference/observability/metrics-catalog.mdx`. Session IDs go in the log, never in a metric label. The tracker and its tests are in `lib/kv-router/src/session_stats.rs`.
- `routing_host.rs` owns selection, dispatch, and response cleanup. `routing_host/kv_selection.rs` chooses KV-routed workers, `routing_host/request_guard.rs` tracks progress and cleanup, and `routing_host/cancellation.rs` stops unfinished work when the client cancels.
- `prefill_router/` optionally runs a request on a prefill worker before sending it to decode.
- `encoder_router.rs` optionally runs multimodal inputs through an encoder worker before token generation.
