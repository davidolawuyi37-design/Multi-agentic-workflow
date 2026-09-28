from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class AssistantState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

    user_input: str
    
    route: str
    
    specialist_response: str
    
    final_response: str