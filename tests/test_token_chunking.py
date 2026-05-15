"""Unit tests for tiktoken overlap chunking."""

import pytest

from utils.token_chunking import (
    estimate_llm_api_cost,
    get_tiktoken_encoder,
    prepare_pdf_page_texts_token_chunks_for_llm,
)


def test_estimate_llm_api_cost() -> None:
    c = estimate_llm_api_cost(1_000_000, 500_000, 1.0, 2.0)
    assert c["input_cost_usd"] == 1.0
    assert c["output_cost_usd"] == 1.0
    assert c["total_cost_usd"] == 2.0


def test_get_tiktoken_encoder_fallback_unknown_model() -> None:
    enc = get_tiktoken_encoder("not-a-real-openai-model-xyz")
    assert enc.name == "cl100k_base"


def test_prepare_chunks_overlap_windows() -> None:
    # Two short pages so concatenated token stream is non-trivial.
    pages = {1: "alpha " * 200, 2: "beta " * 200}
    chunks, log = prepare_pdf_page_texts_token_chunks_for_llm(
        pages,
        max_chars_per_page=0,
        page_iteration_order=None,
        chunk_size_tokens=100,
        overlap_tokens=20,
        tiktoken_model="gpt-4o-mini",
        system_message="sys",
        user_message_prefix="prefix:\n",
        estimated_output_tokens_per_chunk=10,
        input_price_per_1m_tokens=0.15,
        output_price_per_1m_tokens=0.60,
    )
    assert log["chunk_count"] == len(chunks)
    assert len(chunks) >= 1
    for i, ch in enumerate(chunks):
        assert ch["chunk_id"] == i + 1
        assert "text" in ch and ch["text"]
        assert ch["token_count"] <= 100
        assert ch["end_token"] == ch["start_token"] + ch["token_count"] - 1
        assert isinstance(ch["pages"], list)
    # Step 80: second chunk should start at token 80 if document is long enough.
    if len(chunks) >= 2:
        assert chunks[1]["start_token"] == 80


def test_overlap_must_be_lt_chunk_size() -> None:
    with pytest.raises(ValueError, match="overlap_tokens"):
        prepare_pdf_page_texts_token_chunks_for_llm(
            {1: "hello world"},
            max_chars_per_page=0,
            page_iteration_order=None,
            chunk_size_tokens=100,
            overlap_tokens=100,
            tiktoken_model="gpt-4o-mini",
            system_message="",
            user_message_prefix="",
            estimated_output_tokens_per_chunk=1,
            input_price_per_1m_tokens=0.0,
            output_price_per_1m_tokens=0.0,
        )


def test_empty_pages_returns_empty() -> None:
    chunks, log = prepare_pdf_page_texts_token_chunks_for_llm(
        {1: "   ", 2: ""},
        max_chars_per_page=0,
        page_iteration_order=None,
        chunk_size_tokens=100,
        overlap_tokens=10,
        tiktoken_model="gpt-4o-mini",
        system_message="",
        user_message_prefix="",
        estimated_output_tokens_per_chunk=1,
        input_price_per_1m_tokens=0.0,
        output_price_per_1m_tokens=0.0,
    )
    assert chunks == []
    assert log["chunk_count"] == 0
