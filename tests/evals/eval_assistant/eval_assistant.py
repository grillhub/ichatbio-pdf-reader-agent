import pathlib

import pytest
import yaml
from deepeval.evaluate import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from chat.agent_adapter import EntrypointAdapter
from chat.assistant import ToolCall
from evals.mocks import MockAgent

file = pathlib.Path(__file__).parent / "test_sets" / "generate_expert_requests.yaml"
with open(file) as f:
    tests = yaml.safe_load(f)["test_cases"]

equivalence = GEval(
    name="Equivalence",
    criteria="Determine if the 'actual output' is semantically equivalent to 'expected output'. Cosmetic differences"
             " are okay.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    model="gpt-4.1-mini"
)


@pytest.mark.asyncio
@pytest.mark.parametrize("user_message,expected", [(test["user_message"], test["expected"]) for test in tests])
async def test_generate_requests_for_expert_agents(chat, user_message, expected):
    agents = {
        "idigbio": MockAgent(entrypoints=[
            EntrypointAdapter("run", "Counts and retrieves biodiversity occurrence records using the iDigBio API.",
                              None)])
    }

    messages = await chat(agents, user_message)
    tool_message = messages[-1]
    assert isinstance(tool_message, ToolCall)

    test_case = LLMTestCase(
        input=user_message,
        expected_output=expected,
        actual_output=tool_message.arguments["request"]
    )
    assert_test(test_case, [equivalence])
