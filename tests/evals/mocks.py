from typing import Optional, AsyncIterator

from httpx import AsyncClient

from chat.agent_adapter import AgentAdapter, EntrypointAdapter
from chat.artifact import ArtifactRegistry
from chat.messages import AgentResponseMessage
from storage.content_manager import ContentManager


class MockAgent(AgentAdapter):
    def __init__(self, entrypoints: list[EntrypointAdapter]):
        super().__init__(
            name="agent",
            description="Just for testing",
            icon="",
            card_url="http://localhost/.well-known/agent.json",
            agent_url="http://localhost/",
            entrypoints=entrypoints
        )
        self.mock_messages = []

    def set_response(self, *messages: AgentResponseMessage):
        self.mock_messages = messages

    async def query(self, httpx_client: AsyncClient, content_manager: ContentManager,
                    artifact_registry: ArtifactRegistry, context_id: str, request: str, entrypoint: str,
                    params: Optional[dict]) -> AsyncIterator[AgentResponseMessage]:
        for message in self.mock_messages:
            yield message
