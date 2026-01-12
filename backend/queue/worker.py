import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.prebuilt import ToolNode
# import models
from models.chat import ChatState
from nodes.classifier import query_classifier
from nodes.complex import complex
from nodes.routing import routing_query
from nodes.simple import simple
from nodes.complex import universal_math_solver

def build_graph():
    # Tools
    tools = [universal_math_solver]
    tool_node = ToolNode(tools)
    
    graph_builder = StateGraph(ChatState)
    
    graph_builder.add_node('simple',simple)
    graph_builder.add_node('classifier',query_classifier)
    graph_builder.add_node('routing_query',routing_query)
    graph_builder.add_node('complex',complex)
    graph_builder.add_node('tools',tool_node)
    
    graph_builder.add_edge(START, "classifier")
    graph_builder.add_conditional_edges("classifier", "routing_query")
    graph_builder.add_edge("routing_query","simple")
    graph_builder.add_edge("simple","End")
    graph_builder.add_edge("routing_query", "complex")
    graph_builder.add_edge("complex", "tools")
    graph_builder.add_edge("tools", END)
    
    return graph_builder.compile()
    
    
def main():
    graph=build_graph()
    user=input(">")
    _state=ChatState(
        query=None,
        is_complex=None,
        llm_response=None
    )
    response=graph.invoke(_state)
    return response

main()
    
    