import asyncio
import functools
import json
import os
import pathlib
import re

import ichatbio.types
import pytest
import yaml
from ichatbio.agent_response import DirectResponse, ProcessLogResponse

from deepeval.evaluate import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from src.agent import PDFReaderAgent, PDFReaderParams


file = pathlib.Path(__file__).parent / "test_sets" / "ground_truth.yaml"
with open(file) as f:
    tests = yaml.safe_load(f)["test_cases"]


@functools.lru_cache(maxsize=1)
def _equivalence_metric() -> GEval:
    # Call-time construction: GEval.__init__ touches the OpenAI client, so module-level breaks collection without a key.
    return GEval(
        name="Equivalence",
        criteria="Determine if the 'actual output' is semantically equivalent to 'expected output'. Cosmetic differences"
        " are okay.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        model="gpt-oss-120b",
        # model="gpt-4o-mini",
        async_mode=False,
    )


def _get_response_text(messages) -> str:
    for m in reversed(messages):
        if isinstance(m, DirectResponse):
            return m.text or ""
    return ""


def _get_log_text(messages) -> str:
    logs = [m.text for m in messages if isinstance(m, ProcessLogResponse) and m.text]
    return " ".join(logs)


_TESTS_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
_RESULT_PATH = pathlib.Path(__file__).resolve().parent / "output" / "result.json"
_SUMMARY_PATH = pathlib.Path(__file__).resolve().parent / "output" / "summary.json"
_RETRIEVAL_TOP_K = 5
_GEVAL_PASS_THRESHOLD = 0.5


@pytest.fixture(scope="session", autouse=True)
def _clear_eval_output_files() -> None:
    """Reset eval outputs at the start of each pytest session."""
    _RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RESULT_PATH.write_text("[]\n", encoding="utf-8")
    _SUMMARY_PATH.write_text("{}\n", encoding="utf-8")


def _test_request_delay_seconds() -> float:
    """Sleep between repeated eval runs to reduce API burst/rate-limit risk."""
    raw = os.getenv("PDF_READER_EVAL_DELAY_SECONDS", "0.5").strip()
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.5


def _pdf_path_for_artifact(artifact_name: str) -> pathlib.Path:
    direct = _TESTS_DIR / "resources" / artifact_name
    if direct.is_file():
        return direct
    underscored = _TESTS_DIR / "resources" / artifact_name.replace(" ", "_")
    if underscored.is_file():
        return underscored
    raise FileNotFoundError(
        f"Test PDF not found for artifact={artifact_name!r}. Tried {direct} and {underscored}"
    )


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _f1_score(precision: float, recall: float) -> float:
    return _safe_div(2 * precision * recall, precision + recall)


def _gold_pages(expected) -> set[int]:
    pages: set[int] = set()
    if isinstance(expected, list):
        for row in expected:
            if not isinstance(row, dict):
                continue
            page = row.get("page")
            try:
                pages.add(int(page))
            except (TypeError, ValueError):
                continue
    return pages


def _expected_to_string(expected) -> str:
    if isinstance(expected, str):
        return expected
    if isinstance(expected, list):
        lines: list[str] = []
        for row in expected:
            if not isinstance(row, dict):
                continue
            page = row.get("page")
            sentence = str(row.get("sentence", "")).strip()
            if not sentence:
                continue
            lines.append(f"[page {page}] {sentence}")
        return "\n".join(lines).strip()
    return str(expected)


def _extract_selected_pages_with_scores(messages, actual_response: str) -> list[dict]:
    for m in reversed(messages):
        if not isinstance(m, ProcessLogResponse):
            continue
        data = getattr(m, "data", None)
        if isinstance(data, dict):
            rows = data.get("selected_pages_with_scores")
            if isinstance(rows, list):
                return rows

    marker = '"selected_pages_with_scores":'
    idx = actual_response.find(marker)
    if idx == -1:
        return []
    arr_start = actual_response.find("[", idx)
    if arr_start == -1:
        return []
    depth = 0
    arr_end = -1
    for i in range(arr_start, len(actual_response)):
        if actual_response[i] == "[":
            depth += 1
        elif actual_response[i] == "]":
            depth -= 1
            if depth == 0:
                arr_end = i
                break
    if arr_end == -1:
        return []
    json_blob = actual_response[arr_start : arr_end + 1]
    try:
        parsed = json.loads(json_blob)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _retrieval_precision_recall(expected, selected_pages_with_scores: list[dict]) -> tuple[float, float]:
    gold = _gold_pages(expected)
    top_pages: list[int] = []
    for row in selected_pages_with_scores[:_RETRIEVAL_TOP_K]:
        if not isinstance(row, dict):
            continue
        page = row.get("page")
        try:
            top_pages.append(int(page))
        except (TypeError, ValueError):
            continue
    predicted = set(top_pages)
    tp = len(predicted & gold)
    precision = _safe_div(tp, len(predicted))
    recall = _safe_div(tp, len(gold))
    return precision, recall


def _extract_json_array_after_marker(text: str, marker: str) -> list[dict]:
    idx = (text or "").find(marker)
    if idx == -1:
        return []
    arr_start = text.find("[", idx)
    if arr_start == -1:
        return []
    depth = 0
    arr_end = -1
    for i in range(arr_start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                arr_end = i
                break
    if arr_end == -1:
        return []
    try:
        parsed = json.loads(text[arr_start : arr_end + 1])
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _extract_quote_findings(actual_response: str) -> list[dict]:
    return _extract_json_array_after_marker(actual_response, "Quote findings detail:")


def _write_eval_result(
    user_message: str,
    selected_pages_with_scores: list[dict],
    fuzzy_precision: float,
    fuzzy_recall: float,
    quote_findings: list[dict],
    geval_score: float,
) -> None:
    _RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if _RESULT_PATH.exists():
        try:
            rows = json.loads(_RESULT_PATH.read_text(encoding="utf-8") or "[]")
        except json.JSONDecodeError:
            rows = []
    if not isinstance(rows, list):
        rows = []
    selected_pages_key = f"selected_pages_with_scores_top_{_RETRIEVAL_TOP_K}"
    precision_key = f"Precision@{_RETRIEVAL_TOP_K}"
    recall_key = f"Recall@{_RETRIEVAL_TOP_K}"
    f1_key = f"F1@{_RETRIEVAL_TOP_K}"
    f1_value = _f1_score(fuzzy_precision, fuzzy_recall)
    new_row = {
        "user_message": user_message,
        "fuzzysearch": {
            selected_pages_key: selected_pages_with_scores[:_RETRIEVAL_TOP_K],
            precision_key: round(fuzzy_precision, 6),
            recall_key: round(fuzzy_recall, 6),
            f1_key: round(f1_value, 6),
        },
        "llm": {
            "quote_findings": quote_findings,
            "geval_score": round(geval_score, 6),
        },
    }
    replaced = False
    for i, row in enumerate(rows):
        if isinstance(row, dict) and row.get("user_message") == user_message:
            rows[i] = new_row
            replaced = True
            break
    if not replaced:
        rows.append(new_row)
    _RESULT_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_eval_summary(rows)


def _write_eval_summary(rows: list[dict]) -> None:
    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []
    geval_scores: list[float] = []
    precision_key = f"Precision@{_RETRIEVAL_TOP_K}"
    recall_key = f"Recall@{_RETRIEVAL_TOP_K}"
    f1_key = f"F1@{_RETRIEVAL_TOP_K}"

    for row in rows:
        if not isinstance(row, dict):
            continue
        fuzzy = row.get("fuzzysearch")
        if isinstance(fuzzy, dict):
            p = fuzzy.get(precision_key)
            r = fuzzy.get(recall_key)
            f1 = fuzzy.get(f1_key)
            if isinstance(p, (int, float)):
                precisions.append(float(p))
            if isinstance(r, (int, float)):
                recalls.append(float(r))
            if isinstance(f1, (int, float)):
                f1_scores.append(float(f1))
        llm = row.get("llm")
        if isinstance(llm, dict):
            g = llm.get("geval_score")
            if isinstance(g, (int, float)):
                geval_scores.append(float(g))

    macro_precision = _safe_div(sum(precisions), len(precisions))
    macro_recall = _safe_div(sum(recalls), len(recalls))
    macro_f1 = _safe_div(sum(f1_scores), len(f1_scores))
    macro_geval = _safe_div(sum(geval_scores), len(geval_scores))
    passed_count = sum(1 for score in geval_scores if score >= _GEVAL_PASS_THRESHOLD)
    failed_count = max(0, len(geval_scores) - passed_count)

    summary = {
        "query_count": len(rows),
        "fuzzysearch": {
            f"Average_{precision_key}": round(macro_precision, 6),
            f"Average_{recall_key}": round(macro_recall, 6),
            f"Average_{f1_key}": round(macro_f1, 6),
        },
        "llm": {
            "average_geval_score": round(macro_geval, 6),
            "passed_count": passed_count,
            "failed_count": failed_count,
        },
    }
    _SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


@pytest.mark.httpx_mock(should_mock=lambda request: str(request.url) == "https://artifact.test")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_message,expected,artifact",
    [(t["user_message"], t["expected"], t["artifact"]) for t in tests],
)
async def test_pdf_reader_agent_with_artifact(context, messages, httpx_mock, user_message, expected, artifact):
    delay_s = _test_request_delay_seconds()
    if delay_s > 0:
        await asyncio.sleep(delay_s)

    artifact_url = "https://artifact.test"
    pdf_path = _pdf_path_for_artifact(artifact)
    pdf_bytes = pdf_path.read_bytes()
    httpx_mock.add_response(url=artifact_url, content=pdf_bytes)

    pdf_artifact = ichatbio.types.Artifact(
        local_id="#eval-pdf",
        description=f"User upload: {artifact}",
        mimetype="application/pdf",
        uris=[artifact_url],
        metadata={"original_filename": artifact},
    )
    params = PDFReaderParams(pdf_artifact=pdf_artifact, pdf_url=None)

    await PDFReaderAgent().run(context, user_message, "read_pdf", params)

    actual_response = _get_response_text(messages)
    actual_logs = _get_log_text(messages)
    actual_output = actual_response if actual_response else actual_logs

    selected_pages_with_scores = _extract_selected_pages_with_scores(messages, actual_response)
    fuzzy_precision, fuzzy_recall = _retrieval_precision_recall(expected, selected_pages_with_scores)
    quote_findings = _extract_quote_findings(actual_response)

    metric = _equivalence_metric()
    test_case = LLMTestCase(
        input=user_message,
        expected_output=_expected_to_string(expected),
        actual_output=actual_output,
    )
    assert_test(test_case, [metric], run_async=False)
    geval_score = float(getattr(metric, "score", 0.0) or 0.0)
    _write_eval_result(
        user_message=user_message,
        selected_pages_with_scores=selected_pages_with_scores,
        fuzzy_precision=fuzzy_precision,
        fuzzy_recall=fuzzy_recall,
        quote_findings=quote_findings,
        geval_score=geval_score,
    )
