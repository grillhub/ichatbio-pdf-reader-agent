import itertools
from typing import AsyncIterator, AsyncGenerator

import pytest
import pytest_asyncio

from chat import assistant
from chat.assistant import LLMAssistant
from chat.conversation import Conversation
from chat.messages import MessageHolder, UserMessage


async def _consume_message(message: assistant.AsyncChatLoopItem):
    match message:
        case assistant.TextStream(text=text) if isinstance(text, AsyncGenerator):
            return assistant.TextStream(text="".join([t async for t in text]))
        case assistant.ToolCall() | assistant.TextStream():
            return message
        case _:
            raise ValueError()


async def _read_chat_loop(chat_loop: AsyncIterator[assistant.AsyncChatLoopItem]):
    return [await _consume_message(message) async for message in chat_loop]


@pytest.fixture
def llm_assistant():
    return LLMAssistant(model="gpt-oss-120b")
    # return LLMAssistant(model="gpt-4o-mini")


@pytest_asyncio.fixture
async def chat(llm_assistant):
    async def chat(agents, request):
        conversation = Conversation(
            history=[MessageHolder(message_id="test_request", message=UserMessage(text=request))],
            message_id_generator=(str(i) for i in itertools.count(start=1))
        )

        return await _read_chat_loop(llm_assistant.chat(conversation, user_message=request, agents=agents))

    return chat