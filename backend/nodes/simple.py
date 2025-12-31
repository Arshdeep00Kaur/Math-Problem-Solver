from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from models.chat import ChatState
import os
from dotenv import load_dotenv
load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
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
    ai_msg = llm.invoke(system_prompt)
    