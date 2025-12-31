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

def query_classifier(state: ChatState) -> ChatState:
    classification_prompt = f"""
    You are a classifier.

    Analyze the user's query and classify whether it is:
    - simple
    - complex

    Guidelines:

    1. Output **simple** if the query involves:
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

    2. Output **complex** if the query involves:
    - Large or long numerical calculations
    - Very complex integrals
    - High-precision engineering or scientific math
    - Problems requiring complex diagrams or exact computation

    Rules:
    - Respond with **only one of these two outputs**:
    simple
    complex
    - Do NOT provide explanations.

    User query:
    {state["question"]}
    """
    response = llm.invoke([HumanMessage(content=classification_prompt)])
    classification = response.content.strip().lower()

    # ✅ Update state
    state["route"] = classification

    return state

    
    
