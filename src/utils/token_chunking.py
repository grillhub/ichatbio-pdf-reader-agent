"""
Token-overlap chunking for PDF quote extraction using tiktoken.

Builds windows over a linear tokenization of page blocks (same ``Page number`` /
``Page text`` framing as character-based chunking) so downstream code can still
resolve quotes to page numbers via span overlap.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def estimate_llm_api_cost(
    input_tokens: int,
    estimated_output_tokens: int,
    input_price_per_1m_tokens: float,
    output_price_per_1m_tokens: float,
) -> dict[str, float]:
    """
    Estimate API cost from token counts and per-million token prices.

    Formula::
        input_cost = input_tokens / 1_000_000 * input_price_per_1m_tokens
        output_cost = estimated_output_tokens / 1_000_000 * output_price_per_1m_tokens
        total_cost = input_cost + output_cost
    """
    scale = 1_000_000.0
    in_cost = (float(input_tokens) / scale) * float(input_price_per_1m_tokens)
    out_cost = (float(estimated_output_tokens) / scale) * float(
        output_price_per_1m_tokens
    )
    return {
        "input_cost_usd": round(in_cost, 8),
        "output_cost_usd": round(out_cost, 8),
        "total_cost_usd": round(in_cost + out_cost, 8),
    }


def get_tiktoken_encoder(model_name: str) -> Any:
    """
    Return a tiktoken ``Encoding`` for ``model_name``.

    Falls back to ``cl100k_base`` if the model name is unknown to tiktoken.
    Raises if tiktoken is missing or neither model nor fallback can load.
    """
    try:
        import tiktoken
    except ImportError as exc:
        raise RuntimeError(
            "The tiktoken package is required for token-based PDF chunking "
            '(install tiktoken or use PDF_QUOTES_STRATEGY="chunked").'
        ) from exc

    name = (model_name or "").strip() or "gpt-4o-mini"
    try:
        return tiktoken.encoding_for_model(name)
    except KeyError:
        logger.warning(
            "tiktoken has no mapping for model %r; using cl100k_base fallback",
            name,
        )
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception as exc:
            raise RuntimeError(
                "Failed to load tiktoken cl100k_base fallback encoding"
            ) from exc


def _format_page_block(page_num: int, page_text: str) -> str:
    """Same framing as ``split_page_texts_into_quote_llm_chunks`` for consistency."""
    return f"Page number: {page_num}\n\nPage text:\n{page_text}\n\n"


def _page_order_from_texts(
    page_texts: dict[int, str],
    page_iteration_order: list[int] | None,
) -> list[int]:
    if page_iteration_order is not None:
        return [p for p in page_iteration_order if p in page_texts]
    return sorted(page_texts.keys())


def prepare_pdf_page_texts_token_chunks_for_llm(
    page_texts: dict[int, str],
    *,
    max_chars_per_page: int,
    page_iteration_order: list[int] | None,
    chunk_size_tokens: int,
    overlap_tokens: int,
    tiktoken_model: str,
    system_message: str,
    user_message_prefix: str,
    estimated_output_tokens_per_chunk: int,
    input_price_per_1m_tokens: float,
    output_price_per_1m_tokens: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Build overlapping token windows over concatenated page blocks, compute usage
    estimates, and return chunks ready for the quote LLM (``text`` + ``pages``).

    Each chunk dict includes:
    ``chunk_id``, ``text``, ``token_count``, ``start_token``, ``end_token`` (inclusive
    global indices in the full document token stream), ``pages`` (sorted page ints).

    Returns ``(chunks, process_log_dict)`` where ``process_log_dict`` is suitable
    for ``process.log(..., data=...)``.
    """
    if overlap_tokens >= chunk_size_tokens:
        raise ValueError(
            f"overlap_tokens ({overlap_tokens}) must be less than chunk_size_tokens "
            f"({chunk_size_tokens}) so each step advances the window."
        )

    enc = get_tiktoken_encoder(tiktoken_model)

    order = _page_order_from_texts(page_texts, page_iteration_order)
    if not order:
        return [], _empty_prep_log("No pages in page_texts after ordering.")

    # Per-page blocks and cumulative token spans for page attribution.
    spans: list[tuple[int, int, int]] = []  # token_start inclusive, token_end exclusive, page
    all_ids: list[int] = []

    for p in order:
        raw = page_texts.get(p, "")
        if not isinstance(raw, str):
            raw = ""
        if max_chars_per_page > 0:
            raw = raw[:max_chars_per_page]
        if not raw.strip():
            continue
        block = _format_page_block(p, raw)
        ids = enc.encode(block)
        t0 = len(all_ids)
        all_ids.extend(ids)
        t1 = len(all_ids)
        spans.append((t0, t1, p))

    if not all_ids:
        return [], _empty_prep_log("All page texts empty after trimming.")

    step = chunk_size_tokens - overlap_tokens
    if step <= 0:
        raise ValueError("step = chunk_size_tokens - overlap_tokens must be positive")

    system_tok = len(enc.encode(system_message or ""))
    prefix_tok = len(enc.encode(user_message_prefix or ""))

    chunks: list[dict[str, Any]] = []
    start_idx = 0
    chunk_id = 0

    while start_idx < len(all_ids):
        end_idx = min(start_idx + chunk_size_tokens, len(all_ids))
        slice_ids = all_ids[start_idx:end_idx]
        body = enc.decode(slice_ids)

        pages_in: set[int] = set()
        for t0, t1, pg in spans:
            if end_idx <= t0 or start_idx >= t1:
                continue
            pages_in.add(pg)
        pages_sorted = sorted(pages_in)

        tok_count = len(slice_ids)
        chunk_id += 1
        end_token_inclusive = start_idx + tok_count - 1
        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": body,
                "token_count": tok_count,
                "start_token": start_idx,
                "end_token": end_token_inclusive,
                "pages": pages_sorted,
            }
        )

        if end_idx >= len(all_ids):
            break
        start_idx += step

    per_chunk_tokens = [c["token_count"] for c in chunks]
    total_chunk_tokens = sum(per_chunk_tokens)
    n = len(chunks)
    # Each LLM call adds system + user prefix + chunk body (multimodal extras not counted).
    estimated_prompt_tokens_per_request = system_tok + prefix_tok
    estimated_input_tokens_total = n * estimated_prompt_tokens_per_request + total_chunk_tokens
    estimated_output_tokens_total = n * max(0, int(estimated_output_tokens_per_chunk))

    cost = estimate_llm_api_cost(
        estimated_input_tokens_total,
        estimated_output_tokens_total,
        input_price_per_1m_tokens,
        output_price_per_1m_tokens,
    )

    process_log: dict[str, Any] = {
        "chunking_mode": "tiktoken_overlap",
        "tiktoken_model": tiktoken_model,
        "chunk_size_tokens": chunk_size_tokens,
        "overlap_tokens": overlap_tokens,
        # "step_tokens": step,
        # "document_total_tokens": len(all_ids),
        "chunk_count": n,
        "token_count_per_chunk": per_chunk_tokens,
        "total_input_tokens_from_chunks": total_chunk_tokens,
        # "system_message_tokens_est": system_tok,
        # "user_message_prefix_tokens_est": prefix_tok,
        # "estimated_prompt_tokens_per_request": estimated_prompt_tokens_per_request,
        "estimated_total_prompt_tokens_all_requests": estimated_input_tokens_total,
        # "estimated_output_tokens_per_chunk": int(estimated_output_tokens_per_chunk),
        "estimated_total_output_tokens_all_requests": estimated_output_tokens_total,
        "estimated_total_tokens": estimated_input_tokens_total
        # + estimated_output_tokens_total,
        # "pricing_input_per_1m_usd": float(input_price_per_1m_tokens),
        # "pricing_output_per_1m_usd": float(output_price_per_1m_tokens),
        # "estimated_api_cost_usd": cost["total_cost_usd"],
        # "estimated_api_cost_breakdown_usd": cost,
    }
    return chunks, process_log


def _empty_prep_log(reason: str) -> dict[str, Any]:
    z = estimate_llm_api_cost(0, 0, 0.0, 0.0)
    return {
        "chunking_mode": "tiktoken_overlap",
        "chunk_count": 0,
        "token_count_per_chunk": [],
        "total_input_tokens_from_chunks": 0,
        "estimated_total_prompt_tokens_all_requests": 0,
        "estimated_total_output_tokens_all_requests": 0,
        "estimated_total_tokens": 0,
        "estimated_api_cost_usd": z["total_cost_usd"],
        "estimated_api_cost_breakdown_usd": z,
        "note": reason,
    }
