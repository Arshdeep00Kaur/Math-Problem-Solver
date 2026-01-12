from sympy import symbols, solve, diff, integrate, simplify, limit, Matrix
from sympy.parsing.sympy_parser import parse_expr
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from langchain.tools import tool
from typing import Optional, List, Tuple, Callable, Dict, Any
from models.chat import ChatState
from langgraph.prebuilt import ToolNode 
import httpx 
import sympy as sp
from openai import OpenAI
import os


async def search_knowledgebase(query: str) -> str:
    """Search the knowledgebase for relevant context"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/query",
            json={"query": query}
        )
    result = response.json()
    return result.get("context", "No results found")

@tool
def universal_math_solver(
    problem_type: str,
    expression: Optional[str] = None,
    variable: str = "x",
    matrix: Optional[List[List[float]]] = None,
    A: Optional[List[List[float]]] = None,
    b: Optional[List[float]] = None,
    bounds: Optional[Tuple[float, float]] = None,
    initial_guess: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Universal math computation tool (symbolic + numerical).
    """

    try:
        var = symbols(variable)

        # ---------- SYMBOLIC ----------
        if problem_type == "solve":
            expr =sp.sympify(expression)
            sol = solve(expr, var)
            return {"type": "algebra", "solution": sol}

        if problem_type == "derivative":
            expr = sp.sympify(expression)
            return {"type": "derivative", "result": diff(expr, var)}

        if problem_type == "integral":
            expr = sp.sympify(expression)
            return {"type": "integral", "result": integrate(expr, var)}

        if problem_type == "limit":
            expr = sp.sympify(expression)
            point = bounds[0] if bounds else 0
            return {"type": "limit", "result": limit(expr, var, point)}

        if problem_type == "simplify":
            expr = sp.sympify(expression)
            return {"type": "simplify", "result": simplify(expr)}

        if problem_type == "matrix":
            M = Matrix(matrix)
            return {
                "type": "matrix",
                "determinant": M.det(),
                "rank": M.rank(),
                "inverse": M.inv() if M.det() != 0 else "Not invertible"
            }

        # ---------- NUMERICAL ----------
        if problem_type == "linear_system":
            A_np = np.array(A, dtype=float)
            b_np = np.array(b, dtype=float)
            sol = np.linalg.solve(A_np, b_np)
            return {"type": "linear_system", "solution": sol.tolist()}

        if problem_type == "numerical_integral":
            expr = sp.sympify(expression)
            f = lambda x: float(expr.subs(var, x))
            result, error = quad(f, bounds[0], bounds[1])
            return {"type": "numerical_integral", "result": result, "error": error}

        if problem_type == "optimization":
            expr = sp.sympify(expression)
            f = lambda x: float(expr.subs(var, x))
            result = minimize(f, x0=initial_guess)
            return {
                "type": "optimization",
                "minimum_value": result.fun,
                "at": result.x.tolist()
            }

        return {"error": "Unsupported or ambiguous problem type"}

    except Exception as e:
        return {"error": str(e)}
    
universal_math_solver_node = ToolNode([universal_math_solver])
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def complex(state:ChatState)->ChatState:
    system_prompt=f""" 
    You are expert in solving math problems.
    whenever you have given complex problem use the knowledge base first to find the solution availabe.
    if the solution is not available then use the tool available for problem computation.
    use knowledge base for reasoning . 
    You have available tool universal_math_solver.
    if universal_math_solver returns "unsupported or ambiguuous proble type." recommend some resources  that have solution to  similar  problem or solve it by reasoning .
    
    Follow these rules strictly:

1. First, search the knowledge base for similar solved problems.
2. Use retrieved knowledge to explain concepts and reasoning steps.
3. If exact computation is required, call the tool `universal_math_solver`.
4. NEVER guess numerical or symbolic results.
5. If the tool returns an error or unsupported type:
   - Solve symbolically if possible
   - Or recommend authoritative resources (textbooks, lectures).

Explain the solution step by step in a clear and logical manner.
    """
    response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {   "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content":state["query"]
        }
    ]
)

    llm_response=response.choices[0].message
    state=state["llm_response"]
    return state


