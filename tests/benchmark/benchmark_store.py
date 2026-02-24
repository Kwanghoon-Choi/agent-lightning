# pyright: reportUnknownMemberType=false
# pyright: reportUnknownCallType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnnecessaryIsInstance=false
# pyright: reportUnknownParameterType=false
# pyright: reportMissingTypeArgument=false
# pyright: reportMissingParameterType=false


# Copyright (c) Microsoft. All rights reserved.

"""Benchmarking store performance by writing and querying spans from the store."""

from __future__ import annotations
import argparse
import asyncio
from dataclasses import dataclass
from datetime import datetime
import os
import random
import sys
import threading
import time
import base64
import requests
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple, cast
from langfuse import get_client

from rich.console import Console

import agentlightning as agl
from agentlightning.utils.otel import get_tracer

from .utils import flatten_dict, random_dict

console = Console(width=200)

# Minus 10 to leave time for setting up env.
MAX_RUNTIME_SECONDS = (int(os.getenv("GITHUB_ACTIONS_TIMEOUT_MINUTES", "30")) - 10) * 60
MAX_STALE_SECONDS = 300
LANGFUSE_API_MAX_LIMIT = 100
LANGFUSE_QUERY_MAX_RETRIES = 8
LANGFUSE_QUERY_RETRY_INITIAL_DELAY_SECONDS = 0.2
LANGFUSE_QUERY_RETRY_MAX_DELAY_SECONDS = 2.0

# kh: v1 Langfuse SDK
_langfuse = None

def _get_langfuse_client():
    global _langfuse
    if _langfuse is None:
        _langfuse = get_client()
    return _langfuse


def _clamp_langfuse_limit(limit: int) -> int:
    if limit <= 0:
        return 1
    return min(limit, LANGFUSE_API_MAX_LIMIT)


def lf_fetch_obs_page(*, page: int, limit: int = 200, trace_id: Optional[str] = None):
    lf = _get_langfuse_client()
    return lf.api.observations.get_many(limit=_clamp_langfuse_limit(limit), page=page, trace_id=trace_id)


def lf_fetch_obs_page_by_session_id(*, rollout_id: str, limit: int = 200, page: int = 1):
    """Query traces by Langfuse v1 API using `session_id` (rollout_id) filter."""
    lf = _get_langfuse_client()
    return lf.api.trace.list(
        page=page,
        limit=_clamp_langfuse_limit(limit),
        session_id=rollout_id,
    )

def lf_next_page(meta: Any) -> Optional[int]:
    # kh: dict 일수도 있고 아닐수도 있다해서 일단 고쳤는데, 아마 보통 dict 아닐듯?.. TEST 필요!!!
    if isinstance(meta, dict):
        # kh: 이게 진짜 발생하는지 TEST 필요!!!
        total = meta.get("totalPages") or meta.get("total_pages")
        cur = meta.get("page") or meta.get("currentPage")
    else:
        total = getattr(meta, "totalPages", None) or getattr(meta, "total_pages", None)
        cur = getattr(meta, "page", None) or getattr(meta, "currentPage", None)

    if isinstance(total, int) and isinstance(cur, int) and cur < total:
        return cur + 1
    return None

def _find_in_dict(d: dict, key: str) -> Optional[str]:
    """중첩 dict에서 key를 DFS로 찾음. 값이 str/int면 str로 반환."""
    stack = [d]
    while stack:
        cur = stack.pop()
        if not isinstance(cur, dict):
            continue
        if key in cur:
            v = cur[key]
            if v is None:
                return None
            return str(v)
        for v in cur.values():
            if isinstance(v, dict):
                stack.append(v)
    return None


def _obs_get(obs: Any, *names: str) -> Any:
    for name in names:
        if isinstance(obs, dict):
            if name in obs and obs[name] is not None:
                return obs[name]
            continue
        value = getattr(obs, name, None)
        if value is not None:
            return value
    return None

def extract_rollout_attempt(obs) -> Tuple[Optional[str], Optional[str]]:
    rid_raw = _obs_get(obs, "sessionId", "session_id")
    tags_raw = _obs_get(obs, "tags")
    rid = str(rid_raw) if rid_raw is not None else None

    aid: Optional[str] = None
    if isinstance(tags_raw, list) and tags_raw:
        aid = str(tags_raw[0])
    elif isinstance(tags_raw, str) and tags_raw:
        aid = tags_raw

    if rid is None or aid is None:
        md = _obs_get(obs, "metadata") or {}
        if isinstance(md, dict):
            rid = rid or _find_in_dict(md, "agentlightning.rollout_id")
            aid = aid or _find_in_dict(md, "agentlightning.attempt_id")

    return rid, aid


def print_obs(obs):
    rid, aid = extract_rollout_attempt(obs)
    md = _obs_get(obs, "metadata") or {}
    print({
        "id": _obs_get(obs, "id"),
        "traceId": _obs_get(obs, "traceId", "trace_id"),
        "type": _obs_get(obs, "type"),
        "name": _obs_get(obs, "name"),
        "startTime": _obs_get(obs, "startTime", "start_time"),
        "endTime": _obs_get(obs, "endTime", "end_time"),
        "rollout_id": rid,
        "attempt_id": aid,
        "reward_name": _find_in_dict(md, "agentlightning.reward.0.name") if isinstance(md, dict) else None,
        "reward_value": _find_in_dict(md, "agentlightning.reward.0.value") if isinstance(md, dict) else None,
        "span_sequence_id": _find_in_dict(md, "agentlightning.span_sequence_id") if isinstance(md, dict) else None,
    })


def _flatten_nested_dict(src: Any, *, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dict values into dotted keys.

    Existing dotted keys are preserved as-is; nested keys are appended with dots.
    """
    out: dict[str, Any] = {}
    if not isinstance(src, dict):
        return out
    for k, v in src.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_nested_dict(v, prefix=key))
        else:
            out[key] = v
    return out


def _parse_time_for_sort(value: Any) -> float:
    """Normalize possible timestamp representations for deterministic sorting."""
    if value is None:
        return float("inf")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return float("inf")
        # Langfuse commonly returns RFC3339 strings ending with Z.
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(text).timestamp()
        except ValueError:
            return float("inf")
    return float("inf")


@dataclass
class LangfuseSpanLike:
    rollout_id: str
    attempt_id: str
    trace_id: str
    span_id: str
    parent_id: Optional[str]
    name: str
    attributes: dict[str, Any]
    start_time: Optional[float]
    end_time: Optional[float]
    sequence_id: int


async def _langfuse_fetch_observations_for_rollout(
    rollout_id: str,
    *,
    max_pages: int = 100,
    limit: int = LANGFUSE_API_MAX_LIMIT,
) -> list[Any]:
    """Fetch observations for a rollout via trace.list(session_id) -> observations.get_many(trace_id)."""
    async def _fetch_once() -> list[Any]:
        hits: list[Any] = []
        trace_page = 1
        for _ in range(max_pages):
            trace_resp = await asyncio.to_thread(
                lf_fetch_obs_page_by_session_id,
                rollout_id=rollout_id,
                limit=limit,
                page=trace_page,
            )
            traces = getattr(trace_resp, "data", None) or []
            if not traces:
                break

            for trace_item in traces:
                trace_id = _obs_get(trace_item, "id")
                if trace_id is None:
                    continue
                obs_page = 1
                for _ in range(max_pages):
                    obs_resp = await asyncio.to_thread(
                        lf_fetch_obs_page,
                        page=obs_page,
                        limit=limit,
                        trace_id=str(trace_id),
                    )
                    obs_data = getattr(obs_resp, "data", None) or []
                    if not obs_data:
                        break
                    hits.extend(obs_data)
                    obs_next_page = lf_next_page(getattr(obs_resp, "meta", None))
                    if obs_next_page is None:
                        break
                    obs_page = obs_next_page

            next_trace_page = lf_next_page(getattr(trace_resp, "meta", None))
            if next_trace_page is None:
                break
            trace_page = next_trace_page

        # De-dup by observation id while preserving order.
        # kh: page에서 가져오는 중복되는 것들 제거하라는데 필요한지 모르겠음.  TEST 필요!!!
        deduped: list[Any] = []
        seen: set[str] = set()
        for item in hits:
            oid = str(_obs_get(item, "id") or "")
            if oid and oid in seen:
                # kh: 이게 진짜 발생하는지 TEST 필요!!!
                continue
            if oid:
                seen.add(oid)
            deduped.append(item)
        return deduped

    return await _fetch_once()


def _obs_to_langfuse_span(obs: Any, fallback_rollout_id: str, fallback_attempt_id: str) -> LangfuseSpanLike:
    rid, aid = extract_rollout_attempt(obs)
    rid = rid or fallback_rollout_id
    aid = aid or fallback_attempt_id

    metadata = _obs_get(obs, "metadata") or {}
    attributes: dict[str, Any] = {}
    if isinstance(metadata, dict):
        # Langfuse stores OTel span attrs under metadata.attributes in many setups.
        # Use only span-level attrs to avoid duplicated payloads in LF-SPAN print.
        nested_span_attrs = metadata.get("attributes")
        if isinstance(nested_span_attrs, dict):
            attributes.update(nested_span_attrs)
        else:
            # Fallback for payloads that don't expose metadata.attributes.
            attributes.update(_flatten_nested_dict(metadata))

    start_time_raw = _obs_get(obs, "startTime", "start_time")
    end_time_raw = _obs_get(obs, "endTime", "end_time")
    start_time = _parse_time_for_sort(start_time_raw)
    end_time = _parse_time_for_sort(end_time_raw)
    if start_time == float("inf"):
        start_time = None
    if end_time == float("inf"):
        end_time = None

    seq_raw = attributes.get("agentlightning.span_sequence_id")
    if seq_raw is None and isinstance(metadata, dict):
        seq_raw = (
            _find_in_dict(metadata, "agentlightning.span_sequence_id")
            or _find_in_dict(metadata, "attributes.agentlightning.span_sequence_id")
        )
    try:
        sequence_id = int(str(seq_raw)) if seq_raw is not None else -1
    except ValueError:
        sequence_id = -1

    parent_id = cast(
        Optional[str],
        _obs_get(obs, "parent_observation_id"),
    )

    return LangfuseSpanLike(
        rollout_id=rid,
        attempt_id=aid,
        trace_id=str(_obs_get(obs, "traceId", "trace_id") or ""),
        span_id=str(_obs_get(obs, "id") or ""),
        parent_id=parent_id,
        name=str(_obs_get(obs, "name") or ""),
        attributes=attributes,
        start_time=start_time,
        end_time=end_time,
        sequence_id=sequence_id,
    )


async def langfuse_query_spans(
    rollout_id: str,
    attempt_id: str | Literal["latest"] | None = None,
    *,
    trace_id: Optional[str] = None,
    trace_id_contains: Optional[str] = None,
    span_id: Optional[str] = None,
    span_id_contains: Optional[str] = None,
    parent_id: Optional[str] = None,
    parent_id_contains: Optional[str] = None,
    name: Optional[str] = None,
    name_contains: Optional[str] = None,
    filter_logic: Literal["and", "or"] = "and",
    limit: int = -1,
    offset: int = 0,
    sort_by: Optional[str] = "sequence_id",
    sort_order: Literal["asc", "desc"] = "asc",
    max_pages: int = 10,
    page_limit: int = LANGFUSE_API_MAX_LIMIT,
) -> agl.PaginatedResult[LangfuseSpanLike]:
    """Langfuse-backed query that mirrors CollectionBasedLightningStore.query_spans semantics."""
    observations = await _langfuse_fetch_observations_for_rollout(rollout_id, max_pages=max_pages, limit=page_limit)
    if not observations:
        return agl.PaginatedResult(items=[], limit=limit, offset=offset, total=0)

    resolved_attempt_id: Optional[str]
    if attempt_id is None:
        resolved_attempt_id = None
    elif attempt_id == "latest":
        # Approximate "latest attempt" by latest observation timestamp for the rollout.
        latest_obs = max(
            observations,
            key=lambda o: _parse_time_for_sort(_obs_get(o, "endTime", "end_time", "startTime", "start_time")),
        )
        _, latest_attempt = extract_rollout_attempt(latest_obs)
        if not latest_attempt:
            return agl.PaginatedResult(items=[], limit=limit, offset=offset, total=0)
        resolved_attempt_id = latest_attempt
    else:
        resolved_attempt_id = attempt_id

    candidate_spans: list[LangfuseSpanLike] = []
    for obs in observations:
        rid, aid = extract_rollout_attempt(obs)
        if rid != rollout_id:
            continue
        if resolved_attempt_id is not None and aid != resolved_attempt_id:
            continue
        candidate_spans.append(_obs_to_langfuse_span(obs, rollout_id, resolved_attempt_id or "attempt-dummy"))

    def _match_field(value: Any, exact: Optional[str], contains: Optional[str]) -> bool:
        checks: list[bool] = []
        if exact is not None:
            checks.append(str(value or "") == exact)
        if contains is not None:
            checks.append(contains in str(value or ""))
        if not checks:
            return True
        return all(checks) if filter_logic == "and" else any(checks)

    filtered: list[LangfuseSpanLike] = []
    for span_item in candidate_spans:
        field_checks = [
            _match_field(span_item.trace_id, trace_id, trace_id_contains),
            _match_field(span_item.span_id, span_id, span_id_contains),
            _match_field(span_item.parent_id, parent_id, parent_id_contains),
            _match_field(span_item.name, name, name_contains),
        ]
        if all(field_checks) if filter_logic == "and" else any(field_checks):
            filtered.append(span_item)

    total = len(filtered)
    if sort_by:
        def _sort_key(span_item: LangfuseSpanLike) -> tuple[int, Any]:
            value = getattr(span_item, sort_by, None)
            return (1, "") if value is None else (0, value)

        filtered.sort(key=_sort_key, reverse=(sort_order == "desc"))

    if limit == -1:
        items = filtered[offset:]
    else:
        items = filtered[offset : offset + limit]

    return agl.PaginatedResult(items=items, limit=limit, offset=offset, total=total)


async def langfuse_dump_for_rollouts_by_session_id(
    rollout_ids: set[str],
    *,
    max_pages: int = 10,
    limit: int = LANGFUSE_API_MAX_LIMIT,
):
    """Dump observations for target rollout IDs using a session-based lookup path.

    Why this function exists:
    - We want a `rollout_id`-centric query path.
    - In our tracing setup, `rollout_id` is written to Langfuse `sessionId`.
    - Langfuse v1 observations endpoint does not reliably support direct filter by
      `sessionId` in our local/self-host setup.

    Strategy used here (v1-compatible):
    1) Query traces by `session_id=rollout_id` via `trace.list(...)`.
    2) For each returned trace, query observations by `trace_id` via
       `observations.get_many(trace_id=...)`.

    Pagination model:
    - Outer pagination: trace pages (`trace_page`)
    - Inner pagination: observation pages (`obs_page`) for each trace
    - Both loops are bounded by `max_pages` for safety.

    Notes:
    - This is more efficient than global page-scan because we avoid scanning all
      observations first.
    - `limit` applies to each page request (both trace and observation calls).
    """
    hits: list[Any] = []

    # Iterate target rollout IDs. Each rollout_id is treated as Langfuse sessionId.
    for rollout_id in rollout_ids:
        trace_page = 1
        # First stage: list traces that belong to this session (rollout).
        for _ in range(max_pages):
            trace_resp = await asyncio.to_thread(
                lf_fetch_obs_page_by_session_id,
                rollout_id=rollout_id,
                limit=limit,
                page=trace_page,
            )
            print(trace_resp)
            traces = getattr(trace_resp, "data", None) or []
            if not traces:
                # No traces found (or no more traces). Move to next rollout_id.
                break

            # Second stage: for each trace, fetch observations by trace_id.
            for trace_item in traces:
                trace_id = _obs_get(trace_item, "id")
                if trace_id is None:
                    # Defensive skip: trace payload without id should not happen,
                    # but we avoid failing the whole dump.
                    continue

                obs_page = 1
                # Page through observations for this specific trace.
                for _ in range(max_pages):
                    obs_resp = await asyncio.to_thread(
                        lf_fetch_obs_page,
                        page=obs_page,
                        limit=limit,
                        trace_id=str(trace_id),
                    )
                    obs_data = getattr(obs_resp, "data", None) or []
                    if not obs_data:
                        # No more observations for this trace.
                        break
                    hits.extend(obs_data)

                    # Follow observation pagination metadata when available.
                    obs_next_page = lf_next_page(getattr(obs_resp, "meta", None))
                    if obs_next_page is None:
                        break
                    obs_page = obs_next_page

            # Follow trace pagination metadata when available.
            np = lf_next_page(getattr(trace_resp, "meta", None))
            if np is None:
                break
            trace_page = np

    # Print matched observations in a human-readable debug form.
    print(f"[langfuse-v1/sessionId] matched {len(hits)} observations for {len(rollout_ids)} rollouts")
    for obs in hits:
        print_obs(obs)
    print("\n\n\n\n\n\n")

async def langfuse_dump_for_rollouts(
    rollout_ids: set[str],
    *,
    max_pages: int = 10,
    limit: int = LANGFUSE_API_MAX_LIMIT,
):
    page = 1
    hits = []

    for _ in range(max_pages):
        resp = await asyncio.to_thread(lf_fetch_obs_page, page=page, limit=limit)
        data = resp.data or []

        # seesionID랑 tags 확인
        # o = resp.data[0]
        # print([a for a in dir(o) if "session" in a.lower() or "tag" in a.lower()])
        # print(getattr(o, "sessionId", None), getattr(o, "session_id", None))
        # print(getattr(o, "tags", None))

        for obs in data:
            rid, _ = extract_rollout_attempt(obs)
            if rid in rollout_ids:
                hits.append(obs)

        if not data:
            break
        np = lf_next_page(resp.meta)
        if np is None:
            break
        page = np

    print(f"[langfuse-v1] matched {len(hits)} observations for {len(rollout_ids)} rollouts")

    # for obs in hits[:200]:
    for obs in hits:
        print_obs(obs)
    print("\n\n\n\n\n\n")

    # print("\n\n\n\n\n\n")
    # print(hits)




# kh: Langfuse API
def _lf_auth() -> str:
    pk = os.environ["LANGFUSE_PUBLIC_KEY"]
    sk = os.environ["LANGFUSE_SECRET_KEY"]
    token = base64.b64encode(f"{pk}:{sk}".encode()).decode()
    return f"Basic {token}"

def langfuse_fetch_recent_observations(limit: int = 200, page: int = 1) -> list[dict]:
    base = os.environ.get("LANGFUSE_BASE_URL", "http://localhost:3000").rstrip("/")
    url = f"{base}/api/public/observations"
    params = {"limit": limit, "page": page}
    r = requests.get(url, headers={"Authorization": _lf_auth()}, params=params, timeout=30)

    if r.status_code >= 400:
        raise RuntimeError(f"Langfuse {r.status_code}: {r.text[:500]}")

    payload = r.json()
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    if isinstance(payload, list):
        return payload
    return payload.get("items", [])


class RolloutProgressTracker:
    """Helper for tracking rollout progress and surfacing stale worker states."""

    def __init__(self, max_stale_seconds: float = MAX_STALE_SECONDS) -> None:
        self._max_stale_seconds = max_stale_seconds
        self._last_progress = time.perf_counter()

    def record_progress(self) -> None:
        self._last_progress = time.perf_counter()

    async def handle_progress(
        self,
        *,
        progress_made: bool,
        pending_rollout_ids: Sequence[str],
        store: agl.LightningStore,
    ) -> None:
        if progress_made:
            self.record_progress()
            return
        await self._check_for_stale(pending_rollout_ids=pending_rollout_ids, store=store)

    async def _check_for_stale(self, *, pending_rollout_ids: Sequence[str], store: agl.LightningStore) -> None:
        if not pending_rollout_ids:
            return
        elapsed = time.perf_counter() - self._last_progress
        if elapsed <= self._max_stale_seconds / 2:
            return
        console.print(f"Stale rollouts: {pending_rollout_ids}")
        if elapsed > self._max_stale_seconds:
            current_workers = await store.query_workers()
            console.print("Stalled. Current worker status shown below:")
            for worker in current_workers:
                console.print(f"  Worker: {worker}", no_wrap=True, overflow="ignore", crop=False)
            raise RuntimeError("Rollout progress has stalled for too long")


def _abort_due_to_timeout() -> None:
    sys.stderr.write(f"[benchmark] Exiting after exceeding the {MAX_RUNTIME_SECONDS // 60} minute timeout.\n")
    sys.stderr.flush()
    os._exit(1)


def _start_timeout_guard(timeout_seconds: float) -> threading.Timer:
    timer = threading.Timer(timeout_seconds, _abort_due_to_timeout)
    timer.daemon = True
    timer.start()
    return timer


def generate_attributes() -> Dict[str, Any]:
    return flatten_dict(
        random_dict(
            depth=(1, 3),
            breadth=(2, 6),
            key_length=(3, 20),
            value_length=(5, 300),
        )
    )


def make_agent(max_rounds: int, sleep_seconds: float) -> agl.LitAgent[str]:
    @agl.rollout
    async def agent(task: str, llm: agl.LLM):
        tracer = get_tracer()
        rounds = random.randint(1, max_rounds)
        selected_round = random.randint(0, rounds - 1)

        # kh: 각 span이 with block에서 종료되면, otlp SDK가 SpanProcessor.on_end(span) 호출,
        # kh: 내부적으로 span_exporter.export, 미리 지정한 endpoint(collector)로 span 전송 시도.
        for i in range(rounds):
            with tracer.start_as_current_span(f"agent{i}") as span:
                # Nested Span
                with tracer.start_as_current_span(f"round{i}_1") as span:
                    await asyncio.sleep(random.uniform(0.0, sleep_seconds))
                    span.set_attributes(generate_attributes())
                    if i == selected_round:
                        span.set_attribute("task", task)

                # Nested Span
                with tracer.start_as_current_span(f"round{i}_2") as span:
                    await asyncio.sleep(random.uniform(0.0, sleep_seconds))
                    span.set_attributes(generate_attributes())

            if random.uniform(0, 1) < 0.5:
                agl.emit_reward(random.uniform(0.0, 1.0))

        # Final Span
        with tracer.start_as_current_span("final") as span:
            await asyncio.sleep(random.uniform(0.0, sleep_seconds))
            span.set_attributes(generate_attributes())

        agl.emit_reward(random.uniform(1.0, 2.0))

    return agent


def check_spans(spans: Sequence[agl.Span], task: str) -> None:
    """Check if the spans contain the task."""
    found_task = any(span.attributes.get("task") == task for span in spans)

    final_reward = agl.find_final_reward(spans)
    print(f"Final reward found: {final_reward}")
    if final_reward is None:
        raise ValueError("Final reward is not found")
    if not (final_reward >= 1 and final_reward <= 2):
        raise ValueError(f"Final reward {final_reward} is not in the range of 1 to 2")
    if not found_task:
        raise ValueError(f"Task {task} is not found in the spans")


class AlgorithmBatch(agl.Algorithm):
    def __init__(
        self,
        mode: Literal["batch", "batch_partial", "single"],
        total_tasks: int,
        batch_size: Optional[int] = None,
        remaining_tasks: Optional[int] = None,
        concurrency: Optional[int] = None,
    ):
        self.mode = mode
        self.total_tasks = total_tasks
        self.batch_size = batch_size
        self.remaining_tasks = remaining_tasks
        self.concurrency = concurrency

    async def run(
        self, train_dataset: Optional[agl.Dataset[Any]] = None, val_dataset: Optional[agl.Dataset[Any]] = None
    ):
        if self.mode == "batch":
            assert self.batch_size is not None
            await self.algorithm_batch(self.total_tasks, self.batch_size)
        elif self.mode == "batch_partial":
            assert self.batch_size is not None
            assert self.remaining_tasks is not None
            await self.algorithm_batch_with_completion_threshold(
                self.total_tasks, self.batch_size, self.remaining_tasks
            )
        elif self.mode == "single":
            assert self.concurrency is not None
            await self.algorithm_batch_single(self.total_tasks, self.concurrency)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    async def algorithm_batch(self, total_tasks: int, batch_size: int):
        """
        At each time, the algorithm will enqueue a batch of rollouts of size `batch_size`.
        The algorithm will use wait_for_rollouts to wait for all rollouts to complete.
        It then checks whether all rollouts are successful and check the spans to ensure the task is found
        and the last reward is in the range of 1 to 2.
        After that, the algorithm will enqueue a new batch of new tasks, until the total number of tasks is reached.
        """
        store = self.get_store()
        tracker = RolloutProgressTracker()
        submitted = 0

        # kh
        # completed_buffer: list[tuple[str, str]] = []  # (rollout_id, task_name)

        while submitted < total_tasks:
            print(f"Submitting batch {submitted} of {total_tasks}")
            batch_count = min(batch_size, total_tasks - submitted)
            batch_rollouts: List[Tuple[str, str]] = []
            await store.add_resources(
                {
                    "llm": agl.LLM(
                        endpoint=f"http://localhost:{submitted}/v1",
                        model=f"test-model-{submitted}",
                    )
                }
            )
            for _ in range(batch_count):
                task_name = f"task-{submitted}-generated"
                # kh: enqueue_rollout -> store.wait_for_rollouts -> dequeue (attempt_id 생성)
                #       -> trace_context -> with_context 활성화?
                rollout = await store.enqueue_rollout(input=task_name, mode="train")
                batch_rollouts.append((rollout.rollout_id, task_name))
                submitted += 1

            pending = {rollout_id: task_name for rollout_id, task_name in batch_rollouts}
            completed_ids: Set[str] = set()
            tracker.record_progress()
            while len(completed_ids) < len(batch_rollouts):
                finished_rollouts = await store.wait_for_rollouts(
                    rollout_ids=[rollout_id for rollout_id, _ in batch_rollouts],
                    timeout=0.0,
                )
                print("[KHHHK] finished_rollouts:", len(finished_rollouts), [r.rollout_id for r in finished_rollouts], flush=True)
                complete_ids_updated: bool = False
                for rollout in finished_rollouts:
                    rollout_id = rollout.rollout_id
                    if rollout_id in completed_ids:
                        continue
                    if rollout.status != "succeeded":
                        raise RuntimeError(f"Rollout {rollout_id} finished with status {rollout.status}")

                    if not os.environ.get("AGL_OTLP_ENDPOINT"):
                        spans = await store.query_spans(rollout_id=rollout_id, attempt_id="latest")
                        for span in spans:
                            print(f"SPPPAN: {span}")
                        check_spans(spans, pending[rollout_id])
                    else:
                        # kh: 실패를 많이 함
                        # spans = await langfuse_query_spans(rollout_id=rollout_id, attempt_id="latest")
                        # for span in spans:
                        #     print(f"LF-SPAN: {span}")
                        # check_spans(cast(Sequence[agl.Span], spans), pending[rollout_id])

                        """Langfuse 경로에서도 in-memory query_spans와 동일한 시맨틱으로 조회/검증."""
                        spans: Sequence[agl.Span] = []
                        last_error: Optional[Exception] = None
                        for attempt in range(LANGFUSE_QUERY_MAX_RETRIES):
                            queried = await langfuse_query_spans(rollout_id=rollout_id, attempt_id="latest")
                            spans = list(cast(Sequence[agl.Span], queried))
                            # for span in spans:
                            #     print(f"LF-SPAN: {span}")

                            try:
                                check_spans(spans, pending[rollout_id])
                                last_error = None
                                break
                            except ValueError as exc:
                                last_error = exc
                                if attempt == LANGFUSE_QUERY_MAX_RETRIES - 1:
                                    raise
                                delay = min(
                                    LANGFUSE_QUERY_RETRY_INITIAL_DELAY_SECONDS * (2**attempt),
                                    LANGFUSE_QUERY_RETRY_MAX_DELAY_SECONDS,
                                )
                                print(
                                    "Langfuse spans appear incomplete for rollout "
                                    f"{rollout_id}; retrying in {delay:.2f}s "
                                    f"({attempt + 1}/{LANGFUSE_QUERY_MAX_RETRIES}) due to: {exc}"
                                )
                                await asyncio.sleep(delay)

                        if last_error is not None:
                            raise last_error

                    completed_ids.add(rollout_id)
                    complete_ids_updated = True

                    # kh
                    # kh
                    # completed_buffer.append((rollout_id, pending[rollout_id]))

                    # if os.environ.get("AGL_OTLP_ENDPOINT") and len(completed_buffer) >= 2:
                    #     tasks10 = {t for _, t in completed_buffer}
                    #     print(f"[langfuse] dumping observations for last {len(tasks10)} tasks...")

                    #     # kh: HTTP 버전
                    #     # obs = []
                    #     # page = 1
                    #     # while len(obs) < 5000:  # 필요량까지만
                    #     #     chunk = langfuse_fetch_recent_observations(limit=100, page=page)
                    #     #     if not chunk:
                    #     #         break
                    #     #     obs.extend(chunk)
                    #     #     page += 1

                    #     # rollout_ids = {rid for rid, _ in completed_buffer}
                    #     # hits = []
                    #     # for o in obs:
                    #     #     md = o.get("metadata") or {}
                    #     #     rattrs = (md.get("resourceAttributes") or {})
                    #     #     if rattrs.get("agentlightning.rollout_id") in rollout_ids:
                    #     #         hits.append(o)

                    #     # print("[langfuse] matched", len(hits))
                    #     # for h in hits:
                    #     #     print_obs(h)

                    #     # kh: HTTP 버전 + raw print
                    #     # print("[debug] sample observation keys:", list(obs[0].keys()) if obs else None)
                    #     # print("[debug] sample observation:", obs[0] if obs else None)
                    #     # print("[debug] sample observation keys:", list(obs[1].keys()) if obs else None)
                    #     # print("[debug] sample observation:", obs[1] if obs else None)
                    #     # print("[debug] sample observation keys:", list(obs[2].keys()) if obs else None)
                    #     # print("[debug] sample observation:", obs[2] if obs else None)


                    #     # kh: SDK v1
                    #     rollout_ids = {rid for rid, _ in completed_buffer}

                    #     print("[langfuse-v1/sessionId] dumping observations ...")
                    #     try:
                    #         await langfuse_dump_for_rollouts_by_session_id(
                    #             rollout_ids,
                    #             max_pages=10,
                    #             limit=100,
                    #         )
                    #     except Exception as exc:
                    #         raise ValueError("NOOOOOOOOOOOOOOOOOOOOOOO") from exc
                    #         print(f"[langfuse-v1/sessionId] failed ({exc}), fallback to page-scan")
                    #         await langfuse_dump_for_rollouts(
                    #             rollout_ids,
                    #             max_pages=10,
                    #             limit=100,
                    #         )

                    #     completed_buffer.clear()
                    # kh
                    # kh

                unfinished_ids = [rollout_id for rollout_id, _ in batch_rollouts if rollout_id not in completed_ids]
                await tracker.handle_progress(
                    progress_made=complete_ids_updated,
                    pending_rollout_ids=unfinished_ids,
                    store=store,
                )

                await asyncio.sleep(5.0)

    async def algorithm_batch_with_completion_threshold(self, total_tasks: int, batch_size: int, remaining_tasks: int):
        """Different from `algorithm_batch`, this algorithm will use query_rollouts to get rollouts' status.
        It will enqueue a new batch of new tasks when the number of running rollouts is less than the remaining tasks threshold.
        """
        store = self.get_store()
        tracker = RolloutProgressTracker()
        submitted = 0
        completed = 0
        active_rollouts: Dict[str, str] = {}

        while completed < total_tasks:
            console.print(f"Completed {completed} of {total_tasks} rollouts")
            if submitted < total_tasks and len(active_rollouts) < remaining_tasks:
                batch_count = min(batch_size, total_tasks - submitted)
                await store.add_resources(
                    {
                        "llm": agl.LLM(
                            endpoint=f"http://localhost:{submitted}/v1",
                            model=f"test-model-{submitted}",
                        )
                    }
                )
                for _ in range(batch_count):
                    task_name = f"task-{submitted}"
                    rollout = await store.enqueue_rollout(input=task_name, mode="train")
                    active_rollouts[rollout.rollout_id] = task_name
                    submitted += 1
                continue

            if not active_rollouts:
                await asyncio.sleep(0.01)
                continue

            rollouts = await store.query_rollouts(rollout_id_in=list(active_rollouts.keys()))
            # kh
            # status_counts = Counter([r.status for r in rollouts])
            # print("[KKKKKKK] status_counts:", status_counts, flush=True)

            # for r in rollouts[:20]:
            #     print("[KKKKKKK] sample:", r.rollout_id, r.status, getattr(r, "error", None), flush=True)
            # kh

            newly_completed = 0
            for rollout in rollouts:
                rollout_id = rollout.rollout_id
                if rollout_id not in active_rollouts:
                    continue
                if rollout.status in ("queuing", "preparing", "running", "requeuing"):
                    continue
                if rollout.status != "succeeded":
                    raise RuntimeError(f"Rollout {rollout_id} finished with status {rollout.status}")
                # spans = await store.query_spans(rollout_id=rollout_id, attempt_id="latest")
                # check_spans(spans, active_rollouts.pop(rollout_id))
                completed += 1
                newly_completed += 1

            await tracker.handle_progress(
                progress_made=newly_completed > 0,
                pending_rollout_ids=list(active_rollouts.keys()),
                store=store,
            )

            if newly_completed == 0:
                await asyncio.sleep(5.0)

    async def algorithm_batch_single(self, total_tasks: int, concurrency: int):
        """Different from `algorithm_batch`, this algorithm will use one async function to enqueue one rollout at a time.
        The function only cares about the rollout it's currently processing.
        It waits for the rollouts with `get_rollout_by_id` and check the spans to ensure the rollout is successful.
        The concurrency is managed via a asyncio semaphore.
        """
        store = self.get_store()
        semaphore = asyncio.Semaphore(concurrency)
        tracker = RolloutProgressTracker()
        active_rollouts: Set[str] = set()
        active_lock = asyncio.Lock()

        async def emit_progress(progress_made: bool) -> None:
            if progress_made:
                async with active_lock:
                    pending_ids = list(active_rollouts)
                await tracker.handle_progress(progress_made=True, pending_rollout_ids=pending_ids, store=store)
                return
            async with active_lock:
                pending_ids = list(active_rollouts)
            await tracker.handle_progress(progress_made=False, pending_rollout_ids=pending_ids, store=store)

        async def handle_single(task_index: int) -> None:
            task_name = f"task-{task_index}"
            async with semaphore:
                console.print(f"Submitting task {task_index} of {total_tasks}")
                await store.add_resources(
                    {
                        "llm": agl.LLM(
                            endpoint=f"http://localhost:{task_index}/v1",
                            model=f"test-model-{task_index}",
                        )
                    }
                )
                rollout = await store.enqueue_rollout(input=task_name, mode="train")
                rollout_id = rollout.rollout_id
                async with active_lock:
                    active_rollouts.add(rollout_id)
                try:
                    while True:
                        current = await store.get_rollout_by_id(rollout_id)
                        if current is not None and current.status in ("failed", "succeeded", "cancelled"):
                            if current.status != "succeeded":
                                raise RuntimeError(f"Rollout {rollout_id} finished with status {current.status}")
                            break
                        await emit_progress(progress_made=False)
                        await asyncio.sleep(5.0)
                    # spans = await store.query_spans(rollout_id=rollout_id, attempt_id="latest")
                    # check_spans(spans, task_name)
                    await emit_progress(progress_made=True)
                finally:
                    async with active_lock:
                        active_rollouts.discard(rollout_id)

        all_tasks = [handle_single(i) for i in range(total_tasks)]
        await asyncio.gather(*all_tasks)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark LightningStore implementations with synthetic rollouts.")
    parser.add_argument("--store-url", default="http://localhost:4747", help="Lightning Store endpoint base URL.")
    parser.add_argument(
        "--mode",
        choices=("batch", "batch_partial", "single"),
        default="batch",
        help="Algorithm mode to exercise different submission patterns.",
    )
    parser.add_argument("--total-tasks", type=int, default=128 * 128, help="Total number of rollouts to submit.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for batch-style modes.")
    parser.add_argument(
        "--remaining-tasks",
        type=int,
        default=512,
        help="Target number of in-flight rollouts before submitting more (batch_partial mode).",
    )
    parser.add_argument("--concurrency", type=int, default=32, help="Maximum concurrent rollouts for single mode.")
    parser.add_argument("--n-runners", type=int, default=32, help="Number of runner processes to launch.")
    parser.add_argument("--max-rounds", type=int, default=10, help="Maximum number of rounds for each rollout.")
    parser.add_argument("--sleep-seconds", type=float, default=1.0, help="Sleep seconds for each rollout.")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging.")
    parser.add_argument("--debug-otel", action="store_true", help="Enable verbose debug logging for OTel.")
    args = parser.parse_args(argv)

    if args.total_tasks <= 0:
        parser.error("--total-tasks must be positive")
    if args.n_runners <= 0:
        parser.error("--n-runners must be positive")
    if args.mode in {"batch", "batch_partial"} and (args.batch_size is None or args.batch_size <= 0):
        parser.error("--batch-size must be positive for batch modes")
    if args.mode == "batch_partial" and (args.remaining_tasks is None or args.remaining_tasks <= 0):
        parser.error("--remaining-tasks must be positive for batch_partial mode")
    if args.mode == "single" and (args.concurrency is None or args.concurrency <= 0):
        parser.error("--concurrency must be positive for single mode")
    if args.max_rounds <= 0:
        parser.error("--max-rounds must be positive")
    if args.sleep_seconds <= 0:
        parser.error("--sleep-seconds must be positive")

    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    agl.setup_logging(
        "DEBUG" if args.debug else "INFO",
        submodule_levels={"agentlightning.utils.otel": "DEBUG" if args.debug_otel else "INFO"},
    )
    store = agl.LightningStoreClient(args.store_url)
    timeout_guard = _start_timeout_guard(MAX_RUNTIME_SECONDS)
    try:
        trainer = agl.Trainer(
            store=store,
            algorithm=AlgorithmBatch(
                mode=cast(Literal["batch", "batch_partial", "single"], args.mode),
                total_tasks=args.total_tasks,
                batch_size=args.batch_size,
                remaining_tasks=args.remaining_tasks,
                concurrency=args.concurrency,
            ),
            n_runners=args.n_runners,
            strategy={
                "type": "cs",
                "managed_store": False,
            },
        )
        # kh: make_agent에서 정의한 agent가 들어감. 그리고 이 agent는 매 rollout 마다 round, span, attribute 생성하고 sleep도 호출.
        trainer.fit(make_agent(max_rounds=args.max_rounds, sleep_seconds=args.sleep_seconds))
    finally:
        timeout_guard.cancel()
        asyncio.run(store.close())


if __name__ == "__main__":
    main()
