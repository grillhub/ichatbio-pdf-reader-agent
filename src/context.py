from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Optional

from ichatbio.agent_response import ResponseContext

# Same pattern as ``langchain_weather.context``: tools resolve the active iChatBio
# request without threading ``ResponseContext`` through LangChain internals.
current_context: ContextVar[Optional[ResponseContext]] = ContextVar(
    "current_context",
    default=None,
)


@asynccontextmanager
async def response_context_bind(ctx: ResponseContext):
    """``async with response_context_bind(context):`` → tools may use ``current_context.get()``."""
    token = current_context.set(ctx)
    try:
        yield
    finally:
        current_context.reset(token)
