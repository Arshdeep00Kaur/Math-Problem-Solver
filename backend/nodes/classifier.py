from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from models.chat import ChatState
from typing import Literal
import os
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

class queryclassifier(BaseModel):
    is_complex:bool

def query_classifier(state: ChatState) ->ChatState:
    classification_prompt = f"""
    You are a classifier.

    Analyze the user's query and classify whether it is:
    {{ "is_complex": true }} or {{ "is_complex": false }}

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
    {{ "is_complex": true }} or {{ "is_complex": false }}
    - Do NOT provide explanations.

    User query:
    {state["query"]}
    """
    response = client.beta.chat.completions.parse(
    model="gemini-2.5-flash",
    response_format=queryclassifier,
    messages=[
        {   "role": "system",
            "content": classification_prompt
        },
        {
            "role": "user",
            "content": state["query"]
        }
    ]
)

    is_complex=response.choices[0].message.parsed.is_complex
    state["is_complex"]=is_complex
    return state

   

    
    
