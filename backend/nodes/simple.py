from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from models.chat import ChatState
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def simple(state:ChatState)->ChatState:
    system_prompt=f"""
    you are expert in mathematical problem solving.
    whenever questions are related to -
    - Basic calculations
    - Symbolic reasoning
    - Language-to-math translation
    - Conceptual or formula-based problems
    - Procedural or step-by-step reasoning

    Examples include:
    - Arithmetic problems
    - Algebraic problems
    - Word problems
    - Geometry problems
    - Probability and statistics
    - Calculus (conceptual or procedural)
    - Logical or puzzle-based problems
    - Optimization and decision problems
    - Step-by-step solution generation
    
    solve the question step by step .
    
    rules:
    1. Answer the question briefly.
    2. show clear reasoning
    3. Always provide final answer.
    4. Use proper mathematical notation where applicable.
    5. Do not reveal internal chain-of-thought.
    6. solve mathematical problems clearly and accurately.

    """
    query=state["query"]
    response = client.chat.completions.create(
    model="gemini-2.5-flash",
    reasoning_effort="low",
    messages=[
        {   "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": query
        }
    ]
)

    llm_response=response.choices[0].message
    state=state["llm_response"]
    return state
    
    