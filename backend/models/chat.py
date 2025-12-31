from typing import List, Optional
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage


class ChatState(TypedDict):
    """State for chatbot conversation."""

    # Conversation history tracked by LangGraph's add_messages reducer
    messages: Annotated[List[AnyMessage], add_messages]
    # Latest user question the graph is handling
    question: str
    # Routing decision (e.g., "simple" or "complex")
    route: Optional[str]