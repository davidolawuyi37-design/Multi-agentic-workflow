from typing import Literal

from llm import llm
from state import AssistantState

from langchain_core.messages import AIMessage

def get_conversation(state: AssistantState):

    conversation = ""

    for message in state["message"]:
        conversation += f"{message.type}: {message.content}\n"

    return conversation

def supervisor_agent(state: AssistantState):
    user_input = state["user_input"]

    prompt = f"""
You are a supervisor for an AI assistant.

Your job is to decide which specialist should handle the user's request.

Available specialists:

explain
plan
write
general

User request:
{user_input}

Return only one word:

explain
plan
write
general
"""

    response = llm.invoke(prompt)

    return {
        "route": (
            response.content.strip()
            if isinstance(response.content, str)
            else "".join(str(part) for part in response.content).strip()
        ).lower()
    }


def explain_agent(state: AssistantState):
    user_input = state["user_input"]

    prompt = f"""
You are an expert teacher.

Explain the user's request clearly and simply.

Use examples where useful.

User request:
{user_input}
"""
    response = llm.invoke(prompt)
    clean_response = response.content

    return {
        "specialist_response": clean_response
    }

def planner_agent(state: AssistantState):
    user_input = state["user_input"]

    prompt = f"""
You are a planning assistant.

Turn the user's request into a practical step-by-step plan.

Keep the steps clear and actionable.

User request:
{user_input}
"""
    response = llm.invoke(prompt)
    clean_response = response.content

    return {
        "specialist_response": clean_response
    }

def writer_agent(state: AssistantState):
    user_input = state["user_input"]

    prompt = f"""
You are a professional writing assistant.

Create the content requested by the user.

Make it natural, clear and appropriate for the request.

User request:
{user_input}
"""
    response = llm.invoke(prompt)
    clean_response = response.content

    return {
        "specialist_response": clean_response
    }

def general_agent(state: AssistantState):
    user_input = state["user_input"]

    prompt = f"""
You are a helpful AI assistant.

Answer the following request clearly:
User request:
{user_input}
"""
    response = llm.invoke(prompt)
    clean_response = response.content

    return {
        "specialist_response": clean_response
    }

def review_agent(state: AssistantState):
    user_input = state["user_input"]
    draft = state["specialist_response"]
    prompt = f"""
You are the final reviewer in an AI assistant system.

The user asked:

{user_input}

Another agent produced this answer:

{draft}

Review the answer.

Fix:
- unclear wording
- missing important information
- unnecessary repetition
- obvious mistakes

Return only the improved final answer.
"""
    response = llm.invoke(prompt)
    clean_response = response.content

    return {
        "final_response": clean_response
    }

def route_request(state: AssistantState) -> Literal["explain", "plan", "write", "general"]:

    route = state["route"]

    if route == "explain":
        return "explain"

    if route == "plan":
        return "plan"

    if route == "write":
        return "write"

    if route == "general":
        return "general"

    return "general"