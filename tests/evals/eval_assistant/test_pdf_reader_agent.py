import asyncio
import functools
import os
import pathlib

import ichatbio.types
import pytest
import yaml
from ichatbio.agent_response import DirectResponse, ProcessLogResponse

from deepeval.evaluate import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from src.agent import PDFReaderAgent, PDFReaderParams


file = pathlib.Path(__file__).parent / "test_sets" / "generate_expert_requests_pdf_reader.yaml"
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

    test_case = LLMTestCase(
        input=user_message,
        expected_output=expected,
        actual_output=actual_output,
    )
    assert_test(test_case, [_equivalence_metric()], run_async=False)
