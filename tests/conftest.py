import pytest

from ichatbio.agent_response import ResponseChannel, ResponseContext, ResponseMessage


class InMemoryResponseChannel(ResponseChannel):
    """
    Useful for interacting with agents locally (e.g., unit tests, command line interfaces) instead of sending responses
    over the network. The `message_buffer` is populated by running an agent.

    Example:

        messages = list()
        channel = InMemoryResponseChannel(messages)
        context = ResponseContext(channel)

        # `messages` starts empty
        agent = HelloWorldAgent()
        await agent.run(context, "Hi", "hello", None)
        # `messages` should now be populated

        assert messages[1].text == "Hello world!"
    """

    def __init__(self, message_buffer: list):
        self.message_buffer = message_buffer

    async def submit(self, message: ResponseMessage, context_id: str):
        self.message_buffer.append(message)


TEST_CONTEXT_ID = "617727d1-4ce8-4902-884c-db786854b51c"


@pytest.fixture(scope="function")
def messages() -> list[ResponseMessage]:
    """During unit tests, the agent's replies to iChatBio will be stored in this list"""
    return list()


@pytest.fixture(scope="function")
def context(messages) -> ResponseContext:
    """
    A special test context which gathers agent response messages as they are generated. Messages that do not occur
    within a process block are assigned the context_id ``"617727d1-4ce8-4902-884c-db786854b51c"``.
    """
    return ResponseContext(InMemoryResponseChannel(messages), TEST_CONTEXT_ID)
