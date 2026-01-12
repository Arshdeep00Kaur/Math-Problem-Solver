from models.chat import ChatState
from typing import Literal
def routing_query(state:ChatState)->Literal["complex","simple"]:
    is_complex=state["is_complex"]
    if is_complex:
        return "complex"
    else:
        return "simple"