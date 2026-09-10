import os
import sys
import io
import traceback

from typing import TypedDict, List, Optional

import uvicorn
from fastapi import FastAPI
from langserve import add_routes

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END


# ==========================================
# 1. GEMINI API KEY
# ==========================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. "
        "Please set the GEMINI_API_KEY environment variable."
    )


# ==========================================
# 2. LLM INITIALIZATION
# ==========================================

llm_flash = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    api_key=GEMINI_API_KEY,
    temperature=0
)

llm = llm_flash


# ==========================================
# 3. STATE DEFINITION
# ==========================================

class CrewState(TypedDict):
    messages: List[BaseMessage]
    next_step: Optional[str]
    code: Optional[str]
    report: Optional[str]


# ==========================================
# 4. TOOLS
# ==========================================

@tool
def run_python_code(code: str) -> str:
    """Execute Python code and return the output or error."""

    if not isinstance(code, str):
        code = str(code)

    clean_code = (
        code
        .replace("```python", "")
        .replace("```", "")
        .strip()
    )

    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    try:
        local_scope = {}
        exec(clean_code, {}, local_scope)
        result = new_stdout.getvalue()

    except Exception:
        result = "Execution Error:\n" + traceback.format_exc()

    finally:
        sys.stdout = old_stdout

    if result.strip():
        return result.strip()

    return "Success (no terminal output)"


@tool
def generate_test_cases(task_description: str) -> str:
    """Generate 3 to 5 test scenarios for a coding task."""

    prompt = f"""
You are a Senior QA Engineer.

Generate 3 to 5 highly specific test scenarios
for the following Python coding task:

{task_description}

Include:
1. Normal test cases
2. Edge cases
3. Boundary cases

Return only a numbered list.
"""

    response = llm.invoke(prompt)

    if hasattr(response, "content"):
        return str(response.content)

    return str(response)


# ==========================================
# 5. GRAPH NODES
# ==========================================

def task_input_node(state: CrewState):
    return {
        "next_step": "developer"
    }


def real_time_developer(state: CrewState):

    print("\n[Developer] Writing dynamic code using LLM...")

    task = state["messages"][-1].content

    dev_prompt = f"""
Write a clean Python script to solve this coding task:

{task}

Requirements:
- Return only Python code.
- Do not provide explanations.
- Do not use Markdown.
"""

    response = llm_flash.invoke(dev_prompt)
    content = response.content

    if isinstance(content, list):
        if len(content) > 0:
            if isinstance(content[0], dict):
                code_str = content[0].get("text", "")
            else:
                code_str = str(content[0])
        else:
            code_str = ""
    else:
        code_str = str(content)

    print("\nGenerated Code:")
    print(code_str)

    return {
        "code": code_str,
        "next_step": "tester"
    }


def real_time_tester(state: CrewState):

    print("\n[Tester] Generating dynamic tests...")

    task = state["messages"][-1].content

    test_cases = generate_test_cases.invoke(task)
    cases_str = str(test_cases)

    execution_result = run_python_code.invoke(
        {"code": state["code"]}
    )

    report = f"""
### EXECUTION OUTPUT

{execution_result}

### TEST SCENARIOS EVALUATED

{cases_str}
"""

    print("\nTester Report:")
    print(report)

    return {
        "report": report,
        "next_step": "manager_decision"
    }


def manager_decision_node(state: CrewState):

    print("\n[Manager] Reviewing tester report...")

    report = state.get(
        "report",
        "No report available."
    )

    return {
        "report": report,
        "next_step": "archiver"
    }


def archiver_node(state: CrewState):

    print("\n[Archiver] Task stored successfully.")

    return {
        "next_step": "exit"
    }


# ==========================================
# 6. ROUTING FUNCTIONS
# ==========================================

def route_from_input(state: CrewState):

    next_step = state.get("next_step")

    print(f"\n[Input Router] Next step: {next_step}")

    if next_step == "exit":
        return END

    if next_step == "developer":
        return "developer"

    if next_step == "tester":
        return "tester"

    if next_step == "manager_decision":
        return "manager_decision"

    if next_step == "archiver":
        return "archiver"

    return "developer"


def route_from_decision(state: CrewState):

    next_step = state.get("next_step")

    print(f"\n[Manager Router] Next step: {next_step}")

    if next_step == "archiver":
        return "archiver"

    if next_step == "exit":
        return END

    return "task_input"


# ==========================================
# 7. BUILD LANGGRAPH
# ==========================================

rt_workflow = StateGraph(CrewState)

rt_workflow.add_node("task_input", task_input_node)
rt_workflow.add_node("developer", real_time_developer)
rt_workflow.add_node("tester", real_time_tester)
rt_workflow.add_node("manager_decision", manager_decision_node)
rt_workflow.add_node("archiver", archiver_node)

rt_workflow.add_edge(START, "task_input")

rt_workflow.add_conditional_edges(
    "task_input",
    route_from_input
)

rt_workflow.add_edge(
    "developer",
    "tester"
)

rt_workflow.add_edge(
    "tester",
    "manager_decision"
)

rt_workflow.add_conditional_edges(
    "manager_decision",
    route_from_decision
)

rt_workflow.add_edge(
    "archiver",
    END
)

rt_app = rt_workflow.compile()

print("LangGraph workflow compiled successfully.")


# ==========================================
# 8. API INPUT FORMAT
# ==========================================

class AgentInput(TypedDict):
    input: str


# ==========================================
# 9. FORMAT INPUT FOR LANGGRAPH
# ==========================================

def format_for_agent(x):

    if isinstance(x, dict):
        user_input = x["input"]
    else:
        user_input = x.input

    return {
        "messages": [
            HumanMessage(content=user_input)
        ],
        "next_step": "developer",
        "code": None,
        "report": None
    }


# ==========================================
# 10. EXTRACT FINAL RESPONSE
# ==========================================

def extract_text_response(graph_output):

    if not isinstance(graph_output, dict):
        return str(graph_output)

    report = graph_output.get("report")

    if report:
        return str(report)

    code = graph_output.get("code")

    if code:
        return str(code)

    messages = graph_output.get("messages")

    if messages:
        last = messages[-1]
        return getattr(last, "content", str(last))

    return str(graph_output)


# ==========================================
# 11. CREATE RUNNABLE CHAIN
# ==========================================

formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | rt_app
    | RunnableLambda(extract_text_response)
)


# ==========================================
# 12. FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="AI Developer Tester Workflow",
    description=(
        "LangGraph based AI Developer, Tester, "
        "Manager and Archiver workflow"
    ),
    version="1.0.0"
)

add_routes(
    app,
    formatted_agent_chain,
    path="/agent",
    playground_type="default"
)


# ==========================================
# 13. ROOT ENDPOINT
# ==========================================

@app.get("/")
def root():

    return {
        "message": "AI Developer Tester Workflow is running",
        "endpoint": "/agent"
    }


# ==========================================
# 14. RUN SERVER
# ==========================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
