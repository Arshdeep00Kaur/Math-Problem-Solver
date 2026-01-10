from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.prebuilt import ToolNode
# import models
from models.chat import ChatState
from nodes import classifier,complex,simple
from nodes.complex import universal_math_solver

def build_graph():
    # Tools
    tools = [universal_math_solver]
    tool_node = ToolNode(tools)
    
    graph = StateGraph(ChatState)
    
    graph.add_node('simple',simple)
    graph.add_node('classifier',classifier)
    graph.add_node('complex',complex)
    graph.add_node('tools',tool_node)
    
    graph.add_edge(START, "classifier")
    graph.add_edge("classifier", "simple")
    graph.add_edge("classifier", "complex")
    graph.add_edge("complex", "tools")
    graph.add_edge("tools", END)
    graph.set_entry_point("classifier")